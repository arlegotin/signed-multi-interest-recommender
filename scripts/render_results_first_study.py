"""Aggregate saved native evidence only; no arrays, fits or bootstrap reruns."""
import argparse
import csv
import hashlib
import json
import math
import shutil
import os
import tempfile
from pathlib import Path
from statistics import fmean

import numpy as np

from signed_study import confirmation, diagnostics, protocol_lock
from signed_study.budget import Ledger, atomic_write_json
from signed_study.protocol import canonical_sha256
from signed_study.publication import publish_files, recover_publication
from signed_study.report import report_config, inspection_publication, verify_report_publication, _actual_work
from signed_study.stage_evidence import file_reference, stage_path, verify_record, verify_reference


def checked(reference):
    verify_reference(reference)
    return json.loads(Path(reference['path']).read_text())


def effect(row):
    return {key:row.get(key) for key in ('sample_mean_low', 'sample_mean_high',
            'low', 'high', 'n_users', 'n_complete', 'n_missing', 'standard_error')}


def build(root, ledger):
    lock = root / 'protocol.lock.json'
    digest, payload = protocol_lock.read_lock(lock)
    saved = confirmation.load_historical_confirmation(lock, ledger)
    events = confirmation.read_events(lock)
    inputs = diagnostics._diagnostic_inputs(digest, payload, saved, events)
    diagnostic_path = root / f"diagnostics-{saved['run_id']}.json"
    diagnostic = diagnostics._qualify(
        diagnostics._load_saved_diagnostics(diagnostic_path, ledger, inputs), ledger, saved)
    selected = payload['development']
    config = report_config(root)
    receipt = json.loads(stage_path(config, 'develop').read_text())
    verify_record(receipt)
    for reference in receipt['artifacts'].values():
        verify_reference(reference)
    development = checked(receipt['artifacts']['development'])
    verify_record(development)
    state = ledger.snapshot()
    components = {}
    for index, batch in enumerate(development['batches']):
        workload = checked(receipt['artifacts'][f'batch_{index}'])
        measured = workload['measurements']
        if canonical_sha256(measured) != workload['output_sha256']:
            raise ValueError('Bound workload summary has a changed content digest')
        sums = {key:sum(row.get(key, 0.) for row in measured['user_measurements'])
                for key in ('fit_seconds', 'score_seconds', 'write_seconds', 'overhead_seconds')}
        actual = state['attempts'][batch['attempt_id']]['elapsed_seconds']
        components[batch['kind']] = dict(
            evaluations=measured['actual_evaluations'], status=measured['status'],
            failures=sum(row['status'] != 'completed' for row in measured['attempts']),
            prototype_cap_user_fraction=measured['prototype_cap_user_fraction'],
            elapsed_seconds=actual, measured_components=sums,
            remaining_boundary_seconds=actual-sum(sums.values()))
    metrics = {}
    for slot in saved['comparison_schedule']:
        rows = [row for row in saved['attempts']
                if row['slot'] == slot['name'] and row['status'] == 'completed']
        metrics[slot['name']] = dict(n_complete=len(rows), n_scheduled=len(saved['scheduled_user_ids']),
            **{key:fmean(row['metrics'][key] for row in rows) if rows else None
               for key in ('auc', 'ndcg10', 'negative_at_10', 'tie_fraction')})
    numerical = {}
    for slot in dict.fromkeys(row['slot'] for row in diagnostic['numerical']):
        rows = [row for row in diagnostic['numerical'] if row['slot'] == slot]
        complete = [row for row in rows if row['status'] == 'completed']
        numerical[slot] = dict(n_complete=len(complete), n_other=len(rows)-len(complete),
            mean_absolute_auc_change=fmean(row['absolute_auc_change'] for row in complete) if complete else None,
            max_absolute_auc_change=max((row['absolute_auc_change'] for row in complete), default=None),
            max_absolute_score_difference=max((row['max_absolute_score_difference'] for row in complete), default=None),
            rank_positions_changed=sum(row['rank_positions_changed'] for row in complete))
    strata = {kind:{name:dict(n_users=group['n_users'], imprecise=group['imprecise'],
                comparisons={name:effect(value) for name,value in group['comparisons'].items()})
                for name,group in diagnostic['strata'][kind].items()}
              for kind in ('H_O', 'negative_count')}
    timing = {name:dict(n_completed=row['n_completed'], n_failed=row['n_failed'],
                **{key:{stat:row[key][stat] for stat in ('median', 'p95', 'repeat_sd')}
                   for key in ('fit_seconds', 'score_seconds', 'rank_seconds',
                               'total_evaluation_seconds', 'final_write_seconds', 'pipeline_overhead_seconds')})
              for name,row in diagnostic['timing']['methods'].items()}
    resources = [event['measurements'] for event in state['events'] if event.get('measurements')]
    extrema = {}
    for name in ('rss_bytes', 'os_max_rss_bytes', 'project_disk_bytes',
                 'available_memory_bytes', 'free_disk_bytes'):
        values = [row[name] for row in resources if row.get(name, 0)>0]
        extrema[name] = dict(minimum=min(values), maximum=max(values)) if values else None
    calibration_receipt = json.loads(stage_path(config, 'calibrate').read_text())
    verify_record(calibration_receipt)
    native_start = min(event['recorded_at'] for event in state['events']
                       if event['event'] == 'start'
                       and event['attempt_id'] in calibration_receipt['attempt_ids'])
    native_resources = [event['measurements'] for event in state['events']
                        if event.get('measurements') and event.get('recorded_at', 0)>=native_start]
    native_extrema = {}
    for name in extrema:
        values = [row[name] for row in native_resources if row.get(name, 0)>0]
        native_extrema[name] = dict(minimum=min(values), maximum=max(values)) if values else None
    contamination = diagnostic['contamination']
    conditions_path = root / 'operator_logs' / 'task2-timing-operating-conditions.json'
    conditions = json.loads(conditions_path.read_text())
    verify_record(conditions)
    conditions_hash = file_reference(conditions_path)['sha256']
    anchors = [a['outputs']['operating_conditions'] for a in state['attempts'].values()
               if a.get('status') == 'completed' and a.get('outputs', {}).get('operating_conditions')
               and a['outputs']['operating_conditions']['sha256'] == conditions_hash
               and Path(a['outputs']['operating_conditions']['path']).resolve() == conditions_path.resolve()]
    if len(anchors) != 1:
        raise ValueError('Operating conditions require one exact completed publication')
    conditions_ref = anchors[0]
    verify_reference(conditions_ref)
    environment_path = root / 'environment.json'
    environment=json.loads(environment_path.read_text());verify_record(environment)
    environment_owner=inspection_publication(environment_path,environment,state,require_target=True)
    if environment_owner is None:
        raise ValueError('Inspection requires completed compatible G0 publication')
    work_records=[]
    for stage in ('calibrate','smoke','develop'):
        record=json.loads(stage_path(config,stage).read_text())
        seen=set()
        while True:
            verify_record(record)
            if record['output_sha256'] in seen:raise ValueError('Cyclic work receipt chain')
            seen.add(record['output_sha256']);work_records.append((stage,record))
            previous=record.get('supersedes')
            if not previous:break
            record=checked(previous)
    work_problems=[]
    actual_work=_actual_work(root,ledger,payload,work_records,work_problems,state)
    if work_problems:raise ValueError('Bound work evidence invalid: '+'; '.join(work_problems))
    return dict(schema_version=1, lock_sha256=digest,
        source=payload['identities']['source'], execution_policy=payload['identities']['execution_policy'],
        correctness_sha256=payload['identities']['correctness_sha256'],
        cohort_counts={role:len(selected[f'{role}_user_ids']) for role in ('development', 'confirmation')},
        selection={name:{key:value[key] for key in ('parameters', 'macro_auc')} for name,value in selected['families'].items()},
        primary_comparator=saved['primary_comparator'], primary=effect(saved['primary']),
        confirmation_status=saved['status'], interpretation=saved['interpretation'],
        confirmation_evaluations=saved['actual_evaluations'], confirmation_attempt_ids=saved['attempt_ids'],
        metric_means=metrics, comparisons={name:effect(row) for name,row in saved['comparisons'].items()},
        development_configurations=len(development['configurations']), development_components=components,
        endpoint_coverage=selected['endpoint_coverage'], boundary_winners=selected['boundary_winners'],
        complete_suite_eligible=selected['complete_suite_eligible'], planned_precision=selected['planned_precision'],
        diagnostic_status=diagnostic['status'], diagnostic_attempt_ids=diagnostic['attempt_ids'],
        numerical=numerical, strata=strata,
        negative_mean_norm={key:diagnostic['strata']['negative_mean_norm'][key]
                            for key in ('quantiles', 'paired_users', 'signed_minus_matched_mean_correlation')},
        timing=timing, timing_threadpools=diagnostic['timing']['threadpools'],
        numpy_blas_build=getattr(np.__config__, 'CONFIG', {}).get('Build Dependencies', {}).get('blas', {}),
        thread_environment={key:os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                            'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')},
        cold_shared_seconds=diagnostic['timing']['cold_shared_seconds'],
        contamination_applicability=diagnostic['contamination_applicability'],
        contamination_status=contamination['status'], contamination_reason=contamination.get('reason'),
        contamination_by_seed={seed:{name:effect(row) for name,row in comparisons.items()}
                               for seed,comparisons in contamination.get('paired_by_seed', {}).items()},
        operating_conditions_reference=conditions_ref,
        environment_publication=environment_owner,
        actual_work=actual_work,
        comparison_diagnostics={name:{key:row.get('complete_case',{}).get(key)
            for key in ('mean_difference','standard_error','n_users')} for name,row in saved['comparisons'].items()},
        operating_conditions={key:conditions[key] for key in ('captured_at', 'commands', 'load_average', 'system_cpu_percent_one_second')},
        cumulative_stage_seconds=state['stage_seconds'], setup_seconds=state['setup_seconds'],
        attempted_evaluations=state['attempted_evaluations'], resource_extrema=extrema,
        native_continuation_started_at=native_start, resource_extrema_native_continuation=native_extrema,
        ledger_as_of_event_count=len(state['events']),
        ledger_as_of_events_sha256=canonical_sha256(dict(events=state['events'])),
        confirmation_reference=file_reference(Path(saved['artifact_path'])),
        diagnostics_reference=file_reference(diagnostic_path), development_reference=receipt['artifacts']['development'])



STABLE_KEYS = (
    'lock_sha256', 'source', 'execution_policy', 'correctness_sha256',
    'cohort_counts', 'selection', 'primary_comparator', 'primary',
    'confirmation_status', 'interpretation', 'confirmation_evaluations',
    'confirmation_attempt_ids', 'metric_means', 'comparisons',
    'development_configurations', 'development_components',
    'endpoint_coverage', 'boundary_winners', 'complete_suite_eligible',
    'planned_precision', 'diagnostic_status', 'diagnostic_attempt_ids',
    'numerical', 'strata', 'negative_mean_norm', 'timing', 'timing_threadpools',
    'cold_shared_seconds', 'contamination_applicability',
    'contamination_status', 'contamination_reason', 'contamination_by_seed',
    'operating_conditions_reference', 'operating_conditions',
    'environment_publication', 'attempted_evaluations',
    'confirmation_reference', 'diagnostics_reference', 'development_reference',
)
ADDENDUM = '\n\n## Native continuation evidence addendum\n\n'
RECIPE = [
    '.venv/bin/python -m signed_study report --results results',
    '.venv/bin/python scripts/render_results_first_study.py --results results',
]


def summary_snapshot(root, ledger):
    path = root / 'operator_logs' / 'task2-scientific-summary.json'
    saved = json.loads(path.read_text())
    verify_record(saved)
    if (saved['lock_sha256'] != '7e13cb9bfbb6ddbaa27aa9d39c6291304f3120c74a107ce5eccdf1cb75902eca'
            or saved['source']['commit'] != '5a0595d29869678c9b424ae0a9df42bfcee7f8f9'
            or saved['interpretation'] != 'confirmatory' or saved['confirmation_status'] != 'completed'):
        raise ValueError('This packaging recipe is scoped to the completed frozen native study')
    reference = file_reference(path)
    owner = ledger.snapshot()['attempts'][saved['ledger_attempt_id']]
    published = owner.get('outputs', {}).get('scientific_summary', {})
    if (owner['status'] != 'completed' or published.get('sha256') != reference['sha256']
            or Path(published.get('path', '')).resolve() != path.resolve()):
        raise ValueError('Native aggregate summary lacks its exact completed publication')
    verify_reference(published)
    current = build(root, ledger)
    for key in STABLE_KEYS:
        if saved[key] != current[key]:
            raise ValueError(f'Saved scientific summary differs from current bound evidence: {key}')
    # Historical summary bytes stay fixed. Work counts and comparison metadata
    # are independently rebound to their saved evidence for export validation.
    return dict(saved,actual_work=current['actual_work'],
                comparison_diagnostics=current['comparison_diagnostics']), reference


def verify_csvs(root, summary):
    """Require exact exported identities and values from bound saved evidence."""
    def rows(name,fields,identity,expected):
        with (root/name).open(newline='',encoding='utf-8') as handle:
            reader=csv.DictReader(handle)
            if reader.fieldnames!=fields:raise ValueError(f'{name}: unexpected CSV schema')
            values=list(reader)
        ids=[row[identity] for row in values]
        if len(ids)!=len(set(ids)) or set(ids)!=set(expected):
            raise ValueError(f'{name}: duplicate, missing or extra {identity} identities')
        if any(None in row or any(v is None for v in row.values()) for row in values):
            raise ValueError(f'{name}: malformed CSV row')
        return values
    def same_number(value,expected,*,integer=False):
        if expected is None:
            if value!='':raise ValueError('Unavailable export value must remain unavailable')
            return
        try:number=int(value) if integer else float(value)
        except (ValueError,TypeError) as error:raise ValueError('Invalid exported number') from error
        if not math.isfinite(number) or number!=expected or integer and (number<0 or str(number)!=value):
            raise ValueError('Exported numeric value differs from bound finite evidence')
    metrics=rows('metrics.csv',['method','n_completed','n_scheduled','auc','ndcg10','negative_at_10','tie_fraction'],
                 'method',summary['metric_means'])
    comparisons=rows('comparisons.csv',['comparator','sample_mean_low','sample_mean_high','low','high',
                    'n_users','n_complete','n_missing','standard_error','complete_case'],'comparator',summary['comparisons'])
    for row in metrics:
        expected = summary['metric_means'][row['method']]
        same_number(row['n_completed'],expected['n_complete'],integer=True)
        same_number(row['n_scheduled'],expected['n_scheduled'],integer=True)
        for key in ('auc', 'ndcg10', 'negative_at_10', 'tie_fraction'):
            same_number(row[key],expected[key])
    for row in comparisons:
        expected = summary['comparisons'][row['comparator']]
        for key in ('sample_mean_low', 'sample_mean_high', 'low', 'high', 'standard_error'):
            same_number(row[key],expected[key])
        for key in ('n_users', 'n_complete', 'n_missing'):
            same_number(row[key],expected[key],integer=True)
        diagnostic=json.loads(row['complete_case'])
        if set(diagnostic)!={'label','mean_difference','standard_error','n_users'} or diagnostic['label']!='complete_case_diagnostic':
            raise ValueError('Invalid paired complete-case metadata')
        bound=summary.get('comparison_diagnostics',{}).get(row['comparator'])
        if bound is None:
            if expected['n_missing']!=0:raise ValueError('Missing bound complete-case metadata')
            bound=dict(mean_difference=expected['sample_mean_low'],standard_error=expected['standard_error'],n_users=expected['n_complete'])
        for key,value in bound.items():
            if diagnostic[key]!=value or value is not None and not math.isfinite(value):
                raise ValueError('Paired complete-case metadata differs from saved evidence')
    bound_work=summary.get('actual_work')
    if bound_work is None:raise ValueError('Actual work export requires independently bound work evidence')
    expected_work={row['work']:row for row in bound_work}
    if len(expected_work)!=len(bound_work):raise ValueError('Duplicate bound work identities')
    fields=['work','reserved','attempted','completed','failed','unfinished','unavailable','pilot_attempted','status','evidence']
    work=rows('actual-work.csv',fields,'work',expected_work)
    for row in work:
        expected=expected_work[row['work']]
        for key in fields[1:8]:same_number(row[key],expected[key],integer=True)
        if any(row[key]!=expected[key] for key in ('status','evidence')):
            raise ValueError('Work status/evidence differs from saved evidence')
        total=expected['attempted'];counts=[expected[key] for key in ('completed','failed','unfinished')]
        if (total is not None and all(v is not None for v in counts)
                and (sum(counts)>total or expected['status']=='available' and sum(counts)!=total)):
            raise ValueError('Impossible actual-work terminal counters')
        if expected['pilot_attempted'] is not None and (total is None or expected['pilot_attempted']>total):
            raise ValueError('Pilot attempts must be an actual-work subset')
    return metrics, comparisons


def public_exports(root, docs, comparisons):
    os.environ['MPLCONFIGDIR'] = str((root.parent / '.cache' / 'matplotlib').resolve())
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(9, 5.4), layout='constrained')
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    labels = {
        'LinearLogistic':'Logistic C=1 (primary B*)', 'RBFSVM':'RBF SVM',
        'matched_MeanNegative':'Matched negative mean',
        'matched_K1_SignedMax':'Matched K=1 SignedMax',
        'matched_PositiveMax':'Matched positive-only',
        'MeanNegative':'Tuned negative mean', 'PositiveMax':'Tuned positive-only',
        'RocchioDirection':'Rocchio direction', 'AllExemplarMargin':'All-exemplar margin',
        'PrototypeRatio':'Prototype ratio', 'ExemplarRatio':'Exemplar ratio', 'Centroid':'Centroid',
    }
    for index, row in enumerate(comparisons):
        color = '#8e3039' if row['comparator'] == 'LinearLogistic' else '#254b65'
        axis.plot([float(row['low']), float(row['high'])], [index, index], color=color, lw=2)
        axis.plot(float(row['sample_mean_low']), index, 'o', color=color, ms=5)
    axis.set_yticks(range(len(comparisons)), [labels[row['comparator']] for row in comparisons])
    axis.tick_params(labelsize=10)
    axis.axvline(0, color='#b9b9b9', lw=.8)
    axis.axvline(.01, color='#787878', linestyle='--', lw=1.2, label='delta = 0.01')
    axis.set_xlabel('AUC difference: SignedMax − comparator (95% user-bootstrap CI)', fontsize=11)
    axis.set_title('Frozen confirmation: 1776 users · SignedMax K=2, lambda=2', fontsize=12, pad=12)
    axis.legend(loc='lower right', frameon=False, fontsize=9)
    target = docs / 'figures' / 'native-paired-intervals.png'
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=150, metadata={'Software':'signed-study'})
    shutil.copyfile(root / 'metrics.csv', docs / 'native-confirmation-metrics.csv')
    shutil.copyfile(root / 'comparisons.csv', docs / 'native-confirmation-comparisons.csv')
    return [target, docs / 'native-confirmation-metrics.csv', docs / 'native-confirmation-comparisons.csv']


def report_addendum(root, summary):
    primary = summary['primary']
    baseline = summary['primary_comparator']
    metrics = summary['metric_means']
    conditions = summary['operating_conditions']
    lines = [
        f"Frozen primary: SignedMax AUC {metrics['SignedMax']['auc']:.9f}; "
        f"{baseline} B* AUC {metrics[baseline]['auc']:.9f}; "
        f"difference {primary['sample_mean_low']:.9f}, 95% user-bootstrap CI "
        f"[{primary['low']:.9f}, {primary['high']:.9f}], delta=0.01. "
        f"All {primary['n_users']} users completed; interpretation={summary['interpretation']}.",
        'The user-authorized pre-freeze results_first amendment made elapsed limits, forecasts '
        'and ordinary material-repair quota advisory. Scientific settings, full cohorts, seeds, '
        'delta, evaluation bound and non-time resource guards were preserved. Original stop '
        'publications and the 20-file historical archive remain intact; saved calibration '
        'adoption added zero model evaluations.',
        'Calibration actual-work status retains the original terminal forecast failure. '
        'All 1056 measurements completed and the separate successful zero-evaluation adoption '
        'qualified them; the original failed owner was never relabeled.',
        '| Comparator/control | AUC | Signed difference | 95% interval |',
        '| --- | ---: | ---: | --- |',
    ]
    for name in ('LinearLogistic', 'MeanNegative', 'matched_MeanNegative',
                 'matched_K1_SignedMax', 'matched_PositiveMax'):
        row = summary['comparisons'][name]
        lines.append(f"| {name} | {metrics[name]['auc']:.9f} | {row['sample_mean_low']:.9f} "
                     f"| [{row['low']:.9f}, {row['high']:.9f}] |")
    lines += [
        'Secondary controls/mechanism intervals are frozen and unadjusted. Replacing the '
        'negative maximum with a raw mean also changes user-specific penalty scale; these '
        'controls do not uniquely identify a geometric causal mechanism.',
        f"Coverage: 418 development / 1776 confirmation users, {summary['development_configurations']} "
        f"complete development configurations, thirteen confirmation slots, "
        f"{summary['confirmation_evaluations']} confirmation evaluations; "
        f"complete-suite coverage eligible={summary['complete_suite_eligible']}. "
        'Coverage eligibility is not complete-suite superiority or unrestricted tuning.',
        'All 704 float32 checks completed; maximum AUC perturbation was zero despite score '
        'and rank perturbations. Canonical accuracy remains float64. Contamination status: '
        f"{summary['contamination_status']}; {summary['contamination_reason']}. This was a "
        'scientific applicability decision, not a time skip.',
    ]
    high = summary['strata']['H_O']['high_H_high_O']
    row = high['comparisons']['matched_MeanNegative']
    lines.append(f"Frozen high-H/high-O subgroup n={high['n_users']}: SignedMax minus matched "
                 f"mean {row['sample_mean_low']:.9f}, CI [{row['low']:.9f}, {row['high']:.9f}]; "
                 'secondary and unadjusted.')
    lines += [
        'Actual development workload boundaries (seconds):',
        '| Batch | Fit | Score/metrics | Journal writes | Broader pipeline overhead | Remaining boundary | Elapsed |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for name, batch in summary['development_components'].items():
        row = batch['measured_components']
        lines.append(f"| {name} | {row['fit_seconds']:.6f} | {row['score_seconds']:.6f} "
                     f"| {row['write_seconds']:.6f} | {row['overhead_seconds']:.6f} "
                     f"| {batch['remaining_boundary_seconds']:.6f} | {batch['elapsed_seconds']:.6f} |")
    lines += [
        'Broader overhead includes accumulating evidence serialization/publication, resource '
        'checks and other pipeline work; it does not isolate physical I/O. Initial grid '
        '2711.963s, endpoint-campaign pricing 375.467s and final readiness/serialization 82.257s '
        'are measured boundaries. The advisory full-cohort G3 forecast 83643.394s priced '
        'the bounded maximum schedule; actual coverage used 83 configurations plus its pilot.',
        'Full-catalog timing: 3883 items, 32 prespecified users, one warmup and five timed repeats. '
        'These are performance repetitions, not accuracy seeds. Configured policy: one worker '
        'and one BLAS thread; detected libomp: one thread. NumPy uses Accelerate; actual '
        'Accelerate thread count was not independently verified and thread-control environment '
        'variables were unset. Timing is qualified separately from frozen canonical accuracy.',
        f"Fresh pre-G5 conditions at {conditions['captured_at']}: "
        f"{conditions['commands']['power_source']['stdout'].splitlines()[0]}; "
        'pmset power mode 0; no recorded thermal/performance warning; '
        f"system CPU {conditions['system_cpu_percent_one_second']}% over one second; "
        f"load averages {conditions['load_average']}. This is a point-in-time snapshot with "
        'background contention, not continuous isolation or catalog extrapolation.',
        'Native-continuation resource extrema (absolute bytes): '
        +json.dumps(summary['resource_extrema_native_continuation'], sort_keys=True)+'. '
        'GB in the readable outcome is decimal; hardware RAM GiB is binary.',
        'Cumulative stage seconds at the preserved native-summary snapshot: '
        +json.dumps(summary['cumulative_stage_seconds'], sort_keys=True)
        +f"; setup separately {summary['setup_seconds']:.6f}s. "
        'Later reporting extends the ledger; historical failures and metadata-audit costs remain included.',
        'Inspection publication binding independently verified by same resolved file and exact SHA: '
        +json.dumps(summary['environment_publication'], sort_keys=True)+'. '
        'The generic relative/absolute path-spelling caveat is resolved; hardware values remain '
        'historical inspection evidence supplemented by the fresh conditions snapshot.',
        'Reproduction uses the tracked scripts/render_results_first_study.py after the generic '
        'report CLI, with original bound local artifacts preserved. No workflow scratch, '
        'new outcomes, fitting, encoding or bootstrap rerun is required.',
    ]
    path = root / 'report.md'
    text = path.read_text().split(ADDENDUM, 1)[0]
    path.write_text(text.rstrip()+ADDENDUM+'\n\n'.join(lines)+'\n')


def package(root, docs, ledger, attempt_id):
    """Validate prospective exports, then publish one recoverable saved bundle.

    A durable bundle retains new files and prior projections. The selector is
    updated only after every replacement; success additionally requires the
    exact completed G6 receipt owner. Interrupted selection is rolled back on
    the next invocation unless that owner durably completed. No scientific
    stages or inference are performed here.
    """
    root=Path(root).resolve();docs=Path(docs).resolve()
    recover_publication(root,'report')
    recover_publication(root,'results-first',ledger=ledger,allowed_roots=[root,docs])
    verify_report_publication(root)
    summary, summary_ref = summary_snapshot(root, ledger)
    pointer_path = root / 'results-first-reproduction.json'
    previous=None
    state=ledger.snapshot() if ledger is not None else dict(events=[],attempts={})
    published=[event['outputs']['results_first_packaging'] for event in state['events']
        if event.get('event')=='finish' and state['attempts'][event['attempt_id']]['stage']=='G6'
        and event.get('status')=='completed' and event.get('outputs',{}).get('results_first_packaging')]
    if pointer_path.exists():
        pointer = json.loads(pointer_path.read_text());verify_record(pointer)
        if published and pointer['receipt']!=published[-1]:
            raise ValueError('Stale native packaging selector; preserve receipts and recover explicitly')
        previous = checked(pointer['receipt']);verify_record(previous)
        previous_owner = ledger.snapshot()['attempts'][previous['ledger_attempt_id']]
        if (previous_owner.get('event')!='finish' or previous_owner.get('stage')!='G6'
                or previous_owner['status'] != 'completed'
                or previous_owner.get('outputs', {}).get('results_first_packaging') != pointer['receipt']):
            raise ValueError('Previous packaging receipt lacks its completed publication')
        verify_reference(previous['inventory_snapshot'])
        if previous['native_summary']!=summary_ref:raise ValueError('Previous packaging uses different saved evidence')
        # Validate preservation before running any renderer or changing a file.
        for name,digest in previous['scientific_hashes'].items():
            path=root.parent/name
            if not any(path.resolve().is_relative_to(scope) for scope in (root,docs)):
                raise ValueError('Previous scientific export path outside declared scope')
            if file_reference(path)['sha256']!=digest:raise ValueError('Prior scientific export changed')
    elif published:
        raise ValueError('Missing native packaging selector for completed publication; recover explicitly')
    _, comparisons = verify_csvs(root, summary)
    with tempfile.TemporaryDirectory(prefix='.results-first-stage-',dir=root.parent) as directory:
        staging=Path(directory);staged_root=staging/'results';staged_docs=staging/'docs'
        staged_root.mkdir();staged_docs.mkdir()
        names=('comparisons.csv','metrics.csv','actual-work.csv','paired-intervals.png',
               'paired-differences.png','report.md','artifact-inventory.json')
        for name in names:shutil.copyfile(root/name,staged_root/name)
        outputs=public_exports(staged_root,staged_docs,comparisons)
        report_addendum(staged_root,summary)
        verify_csvs(staged_root,summary)
        files={root/name:staged_root/name for name in names}
        files.update({docs/path.relative_to(staged_docs):path for path in outputs})
        science_paths=[root/name for name in names[:5]]+[docs/path.relative_to(staged_docs) for path in outputs]
        def key(path):return str(path.relative_to(root.parent)) if path.is_relative_to(root.parent) else str(path)
        hashes={key(path):file_reference(files[path])['sha256'] for path in science_paths}
        if previous is not None and previous['scientific_hashes']!=hashes:
            raise ValueError('Repeated packaging changed a scientific export')
        repeated=previous is not None
        script_ref=file_reference(Path(__file__).resolve())
        ledger_prefix=ledger.path.read_bytes()
        prefix=dict(byte_length=len(ledger_prefix),sha256=hashlib.sha256(ledger_prefix).hexdigest())
        inventory_path=root/'artifact-inventory.json'
        inventory=json.loads((staged_root/inventory_path.name).read_text())
        for row in inventory['artifacts']:
            if row['name']=='report.md':row['sha256']=file_reference(staged_root/'report.md')['sha256']
        inventory['results_first_packaging']=dict(script=script_ref,recipe=RECIPE,native_summary=summary_ref,
            ledger_prefix=prefix,scientific_hashes=hashes,inspection_qualification=summary['environment_publication'])
        atomic_write_json(staged_root/inventory_path.name,inventory)
        inventory_snapshot=root/'operator_logs'/f'results-first-inventory-{attempt_id}.json'
        files[inventory_snapshot]=staged_root/inventory_path.name
        def final_reference(source,target):return dict(path=str(target),sha256=file_reference(source)['sha256'])
        receipt=dict(schema_version=1,ledger_attempt_id=attempt_id,native_summary=summary_ref,
            script=script_ref,recipe=RECIPE,ledger_prefix=prefix,scientific_hashes=hashes,
            repeated_scientific_hashes_equal=repeated,
            inventory_snapshot=final_reference(files[inventory_snapshot],inventory_snapshot),
            report_sha256=file_reference(staged_root/'report.md')['sha256'],
            interpretation='saved-evidence packaging only; no new experiment or bootstrap inference')
        receipt['output_sha256']=canonical_sha256(receipt)
        staged_receipt=staged_root/'receipt.json';atomic_write_json(staged_receipt,receipt)
        receipt_path=root/'operator_logs'/f'results-first-packaging-{attempt_id}.json'
        files[receipt_path]=staged_receipt
        reference=final_reference(staged_receipt,receipt_path)
        pointer=dict(schema_version=1,receipt=reference)
        pointer['output_sha256']=canonical_sha256(pointer)
        report_selector=dict(schema_version=1,report_sha256=receipt['report_sha256'],
            inventory_sha256=file_reference(staged_root/'artifact-inventory.json')['sha256'],
            receipt=reference,qualification='saved native packaging; exact completed G6 owner required')
        report_selector['output_sha256']=canonical_sha256(report_selector)
        staged_selector=staged_root/'report.active.json';atomic_write_json(staged_selector,report_selector)
        files[root/'report.active.json']=staged_selector
        publish_files(root,'results-first',files,pointer_path,pointer,ledger=ledger,owner=attempt_id,
                      outputs={'results_first_packaging':reference},allowed_roots=[root,docs])
    return dict(status='completed', receipt=str(receipt_path), report=str(root / 'report.md'),
                repeated_scientific_hashes_equal=repeated, scientific_hashes=hashes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results')
    parser.add_argument('--docs', default='docs')
    parser.add_argument('--recover-only',action='store_true',
                        help='Restore/project a pending validated bundle without admitting or completing new G6 work')
    args = parser.parse_args(argv)
    root = Path(args.results).resolve()
    docs = Path(args.docs).resolve()
    ledger = Ledger(root / 'budget_ledger.jsonl', report_config(root))
    try:
        if args.recover_only:
            report_state=recover_publication(root,'report')
            publication_state=recover_publication(root,'results-first',ledger=ledger,allowed_roots=[root,docs])
            verify_report_publication(root)
            print(json.dumps(dict(status='recovered',publication_state=publication_state or 'nothing_pending',
                report_state=report_state or 'nothing_pending',
                qualification='Bundle recovery only; interrupted ledger owners remain unfinished; no scientific work executed'),sort_keys=True))
            return
        with ledger.work('G6', 1, 0) as attempt_id:
            result = package(root, docs, ledger, attempt_id)
        print(json.dumps(result, sort_keys=True))
    finally:
        ledger.close()


if __name__ == '__main__':
    main()
