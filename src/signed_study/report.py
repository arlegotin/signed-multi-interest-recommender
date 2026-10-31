"""Deterministic public aggregates from saved evidence; no study execution.

The library renderer never admits work or repairs evidence. The report CLI owns
its G6 accounting. Historical validation cannot authorize opening outcomes.
"""
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
from statistics import fmean

from . import confirmation, diagnostics, protocol_lock
from .budget import Ledger
from .protocol import ProtocolMismatch, canonical_sha256, config_from_record, load_config
from .stage_evidence import file_reference, verify_record, verify_reference


def report_config(root):
    root=Path(root)
    lock=root/'protocol.lock.json'
    config=None
    try:
        if lock.exists(): config=config_from_record(protocol_lock.read_lock(lock)[1]['invocation_config'])
        elif (root/'resolved_config.json').exists(): config=load_config(root/'resolved_config.json')
    except (ValueError,OSError,KeyError,TypeError):
        # Broken evidence must still permit a scoped report and ledger costs.
        pass
    if config is None: config=load_config(Path(__file__).resolve().parents[2]/'config/study.json')
    return replace(config,paths={**config.paths,'results':root})


def _table(path,fields,rows):
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    return str(path)


def _json(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=True,allow_nan=False)


def _outcome(saved,invalid,resource_limited,delta):
    if invalid or saved and saved['interpretation']=='invalid_confirmatory':
        return 'invalid experiment'
    if resource_limited:
        return 'practical infeasibility'
    if (not saved or saved['interpretation']!='confirmatory' or saved['status']!='completed'
            or not saved.get('complete_suite_eligible') or not saved.get('primary')):
        return 'inconclusive evidence'
    primary=saved['primary']
    if primary.get('n_missing',0): return 'inconclusive evidence'
    if primary['low']>delta: return 'supported contribution'
    if primary['high']<delta: return 'informative negative result'
    return 'inconclusive evidence'


def _aggregate(comparison):
    # Explicit export allowlist: never copy individual users/bounds/labels.
    return {k:comparison.get(k) for k in ('sample_mean_low','sample_mean_high','low','high',
                                         'n_users','n_complete','n_missing')}


def _figures(root,comparisons):
    intervals=[(name,r) for name,r in comparisons.items() if r.get('low') is not None and r.get('high') is not None]
    if not intervals: return []
    cache=Path('.cache')/'matplotlib'; cache.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(cache.resolve())
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig=Figure(figsize=(7,max(2,len(intervals)*.3)),layout='constrained'); FigureCanvasAgg(fig)
    ax=fig.subplots()
    for i,(name,r) in enumerate(intervals):
        ax.plot([r['low'],r['high']],[i,i],color='#254b65',linewidth=2)
        if r.get('sample_mean_low')==r.get('sample_mean_high') and r.get('sample_mean_low') is not None:
            ax.plot(r['sample_mean_low'],i,'o',color='#254b65')
    ax.set_yticks(range(len(intervals)),[n for n,_ in intervals]); ax.axvline(.01,color='gray',linestyle='--')
    ax.set_xlabel('Saved paired AUC difference / whole-cohort interval (SignedMax − comparator)')
    path=root/'paired-intervals.png'; fig.savefig(path,dpi=140,metadata={'Software':'signed-study'})
    paths=[str(path)]
    primary=next(iter(comparisons.values()),{})
    lo,hi=primary.get('lower_bounds',[]),primary.get('upper_bounds',[])
    if lo and lo==hi:
        fig=Figure(figsize=(6,3),layout='constrained'); FigureCanvasAgg(fig); ax=fig.subplots()
        ax.hist(lo,bins=min(20,max(1,len(lo))),color='#254b65')
        ax.set_xlabel('Saved per-user paired AUC difference (primary B*)'); ax.set_ylabel('Users')
        path=root/'paired-differences.png';fig.savefig(path,dpi=140,metadata={'Software':'signed-study'});paths.append(str(path))
    return paths


def render_report(results_directory: Path) -> dict:
    """Render all ten report sections and safe aggregates, including early stops.

    No fits, encoding, downloads, new inference or private outcome-array access.
    Missing/invalid files remain explicit; no stage is run to complete them.
    """
    root=Path(results_directory).resolve(); root.mkdir(parents=True,exist_ok=True)
    problems=[]; inventory=[]; stage_rows=[]; saved=None; diagnostic=None; payload={}; invalid=False
    qualifications=[]; failed_records=[]
    config=None; state={}; ledger=None
    def note(path,status,**extra):
        row={'name':str(Path(path).resolve().relative_to(root)),'status':status,**extra}
        if Path(path).is_file(): row['sha256']=file_reference(path)['sha256']
        inventory.append(row)
    try:
        config=report_config(root)
        ledger=Ledger(root/'budget_ledger.jsonl',config)
        state=ledger.snapshot()
    except (ValueError,OSError,KeyError,TypeError) as error:
        problems.append(f'Configuration/accounting invalid: {error}'); invalid=True
    attempts=state.get('attempts',{})
    if (root/'budget_ledger.jsonl').exists(): note(root/'budget_ledger.jsonl','permanent accounting; snapshot identity in report')
    def read(path):
        if not path.exists(): note(path,'missing'); return None
        try:
            result=json.loads(path.read_text()); verify_record(result)
            note(path,'self-checksummed saved metadata')
            return result
        except (ValueError,OSError,KeyError,TypeError) as error:
            note(path,'invalid');problems.append(f'{path.name}: {error}');return None
    environment=read(root/'environment.json') or {}
    environment_ref=file_reference(root/'environment.json') if environment else None
    environment_bound=bool(environment_ref and any(a.get('outputs',{}).get('environment')==environment_ref for a in attempts.values()))
    if environment and not environment_bound:
        problems.append('Inspection metadata is self-checksummed but lacks matching external ledger publication; hardware is unverified historical metadata')
    artifacts={}
    for name in ('acquire','audit','calibrate','smoke','develop'):
        path=root/'stages'/f'{name}.json'; row=read(path)
        if row is None:
            stage_rows.append(f'{name}: missing or invalid; not executed/verified')
            continue
        try:
            for ref in row['artifacts'].values(): verify_reference(ref)
            ids=row['attempt_ids']
            if not ids or any(a not in attempts for a in ids): raise ProtocolMismatch('Missing stage ledger attempts')
            bad=[attempts[a]['status'] for a in ids if attempts[a]['status']!='completed']
            status='incomplete current accounting' if bad else row['status']
            stage_rows.append(f'{name}: {status}; stop reason: {row.get("stop_reason")}')
            artifacts.update(row['artifacts'])
        except (ValueError,OSError,KeyError,TypeError) as error:
            problems.append(f'{name}: {error}');stage_rows.append(f'{name}: invalid evidence')
    for path in sorted((root/'stages').glob('*.failed-*.json')):
        failed=read(path)
        if failed:
            failed_records.append(failed)
            ids=failed.get('attempt_ids',[])
            qualifier='ledger attempts present' if ids and all(a in attempts for a in ids) else 'accounting incomplete/unverified'
            stage_rows.append(f'{failed.get("stage")}: saved failed invocation ({qualifier}); {failed.get("status")}; stop reason: {failed.get("stop_reason")}')
    lock=root/'protocol.lock.json'
    if lock.exists() and ledger:
        try:
            _,payload=protocol_lock.read_lock(lock)
            saved=confirmation.load_historical_confirmation(lock,ledger)
            qualifications=[{k:e.get(k) for k in ('kind','reason','interpretation')}
                            for e in confirmation.read_events(lock) if e['kind'] in ('invalidated','correction_authorized')]
            note(lock,'historically verified freeze and ledger anchor')
            note(Path(saved['artifact_path']),'historically verified; current accounting qualified')
        except (ValueError,OSError,KeyError,TypeError) as error:
            message=str(error); problems.append(f'Confirmation: {message}')
            if 'No finished saved confirmation' in message:
                stage_rows.append('confirmation: analysis pending or not executed; no paired inference available')
            else: invalid=True
            note(lock,'incomplete' if not invalid else 'invalid')
        if saved:
            diagnostic_path=root/f"diagnostics-{saved['run_id']}.json"
            if diagnostic_path.exists():
                try:
                    inputs=diagnostics._diagnostic_inputs(saved['lock_sha256'],payload,saved,confirmation.read_events(lock))
                    diagnostic=diagnostics._qualify(diagnostics._load_saved_diagnostics(diagnostic_path,ledger,inputs),ledger,saved)
                    note(diagnostic_path,'externally ledger-bound; current accounting qualified')
                except (ValueError,OSError,KeyError,TypeError) as error:
                    note(diagnostic_path,'invalid');problems.append(f'Diagnostics invalid: {error}')
            else: note(diagnostic_path,'missing; diagnostics not executed')
    else: note(lock,'missing; confirmation not executed')
    for path in (lock.with_suffix('.sha256'),confirmation._events_path(lock),lock.with_name('confirmation.exposure.json')):
        if path.exists(): note(path,'retained provenance; validation/limitations reported above')
    # Only verified stage references supply acquisition/audit/calibration details.
    details={}
    for name,ref in artifacts.items():
        if Path(ref['path']).name in ('data_manifest.json','encoder_acquisition.json','cohort_audit.json','calibration.json'):
            try: details[Path(ref['path']).name]=json.loads(Path(ref['path']).read_text())
            except (ValueError,OSError) as error: problems.append(f'{name}: {error}')
    calibration=details.get('calibration.json')
    if calibration is None and (root/'calibration.json').exists():
        calibration=read(root/'calibration.json')
    calibration=calibration or {}
    development={}
    if 'development' in artifacts:
        try:
            development=json.loads(Path(artifacts['development']['path']).read_text());verify_record(development)
        except (ValueError,OSError,KeyError,TypeError) as error:
            problems.append(f'Saved development selection invalid: {error}');development={}
    selected=payload.get('development',development)
    delta=payload.get('config',{}).get('delta',.01)
    relevant_work=[a for a in attempts.values() if a['stage'] in ('G2','G3','G4')]
    resource_limited=bool((not saved and relevant_work and relevant_work[-1]['status']=='budget_exceeded')
                          or saved and saved['status']=='budget_exceeded')
    forecast=calibration.get('forecast') or {}
    scaling_stop=bool(not saved and calibration.get('status')=='completed' and calibration.get('actual_evaluations',0)>0
        and forecast.get('status')=='infeasible' and forecast.get('tested_scales')
        and any(a['stage']=='G2' and a['status']=='failed' and a.get('measurements',{}).get('exception')==forecast.get('stop_reason') for a in attempts.values())
        and any(r.get('stage')=='calibrate' and r.get('stop_reason')==forecast.get('stop_reason') for r in failed_records))
    resource_limited=resource_limited or scaling_stop
    outcome=_outcome(saved,invalid,resource_limited,delta)
    comparisons=saved.get('comparisons',{}) if saved else {}
    # Put frozen B* first; another comparator's confirmation mean never reselects it.
    primary_name=selected.get('B_star',{}).get('family',saved.get('primary_comparator') if saved else 'not selected')
    comparisons={n:comparisons[n] for n in [primary_name,*sorted(comparisons)] if n in comparisons}
    rows=[dict(comparator=name,**_aggregate(value)) for name,value in comparisons.items()]
    tables=[]
    if rows:
        tables.append(_table(root/'comparisons.csv',['comparator',*list(_aggregate({}))],rows))
    cost_rows=[dict(stage=stage,seconds=seconds,limit_seconds=state.get('stage_limits',{}).get(stage),
                    failed_seconds=sum(a['elapsed_seconds'] for a in attempts.values() if a['stage']==stage and a['status']!='completed'))
               for stage,seconds in state.get('stage_seconds',{}).items()]
    cost_rows.append(dict(stage='setup',seconds=state.get('setup_seconds',0),limit_seconds=2700,
                          failed_seconds=sum(a['elapsed_seconds'] for a in attempts.values() if a['stage']=='setup' and a['status']!='completed')))
    tables.append(_table(root/'costs.csv',['stage','seconds','limit_seconds','failed_seconds'],cost_rows))
    metric_rows=[]
    if saved and saved.get('attempts'):
        from .metrics import METRIC_KEYS
        for slot in saved.get('comparison_schedule',[]):
            completed=[r for r in saved['attempts'] if r['slot']==slot['name'] and r['status']=='completed']
            if completed:
                metric_rows.append(dict(method=slot['name'],n_completed=len(completed),
                    n_scheduled=len(saved['scheduled_user_ids']),
                    **{k:fmean(r['metrics'][k] for r in completed if k in r['metrics'])
                       if any(k in r['metrics'] for r in completed) else None for k in METRIC_KEYS}))
        if metric_rows: tables.append(_table(root/'metrics.csv',['method','n_completed','n_scheduled',*METRIC_KEYS],metric_rows))
    figures=_figures(root,comparisons)
    fitting_costs={}
    for row in calibration.get('user_measurements',[]):
        family=row.get('family','unknown')
        fitting_costs[family]=fitting_costs.get(family,0.)+row.get('fit_seconds',0.)+row.get('score_seconds',0.)
    method_failures={}
    for row in saved.get('attempts',[]) if saved else []:
        if row['status']!='completed':
            key=(row['slot'],row['status'],row.get('exception'))
            method_failures[key]=method_failures.get(key,0)+1
    failure_rows=[dict(method=name,status=status,reason=reason,n_attempts=count)
                  for (name,status,reason),count in method_failures.items()]
    # Remove only renderer-owned stale exports when the current evidence no
    # longer supports them; an old benchmark must not survive an early stop.
    current={Path(p).name for p in [*tables,*figures]}
    for name in ('comparisons.csv','metrics.csv','paired-intervals.png','paired-differences.png'):
        if name not in current: (root/name).unlink(missing_ok=True)
    lines=['# Signed multi-interest recommendation study','',
        '## 1. Question, outcome and recommendation',
        'Does signed multi-interest ranking improve observed-rating user AUC over the frozen development-selected comparator by delta=0.01?',
        f'Outcome: **{outcome}**. Scope: this metadata/encoder setting and frozen cohort only.',
        {'supported contribution':'The primary result supports the specified advantage over B* within this scope; assess complete-suite and mechanism evidence separately.',
         'informative negative result':'The valid primary interval rules out the predefined useful advantage; nonsignificance alone is not this conclusion.',
         'practical infeasibility':'The recorded resource stop limits the specified schedule under the caps; a timeout is not a scientific refutation.',
         'invalid experiment':'Comparative conclusions are not valid. Preserve attempted evidence and resolve the recorded validity failure.',
         'inconclusive evidence':'Evidence does not justify a practical comparative recommendation.'}[outcome],
        '', '## 2. Execution and hardware',
        'Historical evidence validation only; no fresh authorization to expose labels or execute stages. Current source HEAD, encoder installation and caches are not revalidated.',
        'Hardware (saved inspection; '+('ledger publication verified' if environment_bound else 'not externally verified')+'): '+_json({k:environment.get('hardware',{}).get(k) for k in ('system','machine','chip','installed_ram_bytes','logical_cpus')}),
        'Inspection blockers: '+_json(environment.get('blockers',[])),*stage_rows,
        'Saved measured forecast: '+_json({k:forecast.get(k) for k in ('status','tested_scales','scale','support_model','stop_reason','seconds','safety_multiplier')}),
        'Scaling qualification: '+('measured saved calibration plus matching permanent G2 failure and failed-stage reason; smallest tested scale shown above' if scaling_stop else 'no terminal measured scaling stop established'),
        f'Confirmation: {saved["status"] if saved else "not executed or no verified finished summary"}. Diagnostics: {diagnostic["status"] if diagnostic else "not executed or unavailable"}.',
        '', '## 3. Provenance and cohort',
        'Sources: MovieLens 1M; access-only MovieLens 100K fallback is separately scoped. Fixed sentence-transformers/all-MiniLM-L6-v2. No representation-universal claim.',
        'Frozen identities: '+_json({k:payload.get('identities',{}).get(k) for k in ('source','data_sha256','cache_sha256','correctness_sha256','metric_sha256')}),
        'Frozen encoder identity: '+_json({k:payload.get('config',{}).get(k) for k in ('dataset','model_id','model_revision','environment_hash')}),
        'Dataset/model acquisition: '+_json({name:{k:r.get(k) for k in ('dataset','archive_sha256','model_id','model_revision','license')} for name,r in details.items() if name in ('data_manifest.json','encoder_acquisition.json')}),
        'Eligible-user funnel: '+_json(details.get('cohort_audit.json',{}).get('funnel','unavailable')),
        'Saved burst/cap audit: '+_json({k:details.get('cohort_audit.json',{}).get(k) for k in ('bursts','caps','medians')}),
        'Temporal split j=(7*n)//10; burst ratings limit temporal interpretation. Eligibility [10,5,5,5]; history support caps [128,64]. Missing interactions are not evaluation negatives.',
        'Frozen protocol: '+(saved['lock_sha256'] if saved else 'not verified/reached'),
        '', '## 4. Primary effect and required comparisons',f'Frozen primary B*: {primary_name}; delta=0.01. Confirmation never reselects B*.',
        'Interpretation: '+(saved['interpretation'].replace('_',' ') if saved else 'no confirmation interpretation'),
        'Primary saved aggregate: '+_json(_aggregate(saved.get('primary',{})) if saved else {}),
        'All saved comparisons: '+_json(rows) if rows else 'No benchmark observations/inference available; no benchmark table invented.',
        'Descriptive completed-case saved metric means (not full-cohort inference): '+_json(metric_rows),
        'Whole frozen denominator: '+str(len(saved.get('scheduled_user_ids',[])) if saved else len(selected.get('confirmation_user_ids',[])))+'. Missing rows retain worst-case AUC difference bounds [-1,1]; no interval is computed by this renderer.',
        'Complete-suite superiority is not established unless every required comparison and validity condition supports it. Primary and complete-suite claims are separate.',
        '', '## 5. Matched controls and attribution',
        'Selected parameters and controls: '+_json([{k:s.get(k) for k in ('name','fit_family','parameters')} for s in selected.get('comparison_schedule',[])]),
        'K=1 SignedMax, positive-only, matched-K/lambda MeanNegative and independently tuned MeanNegative comparisons are required controls; missing controls remain unavailable.',
        'Selection of lambda>0 or K>1 alone is not evidence that the component helps. SignedMax minus matched controls is secondary to the primary comparison; geometry attribution requires the appropriate controls.',
        '', '## 6. Secondary strata and robustness',
        'History H/O and negative-count strata, numerical float32 checks and contamination are secondary/unadjusted; small strata (<100) are imprecise. No retuning or automatic routing claim.',
    ]
    if diagnostic:
        strata=[]
        for grouping,groups in diagnostic.get('strata',{}).items():
            if grouping not in ('H_O','negative_count'): continue
            for name,group in groups.items():
                strata.append(dict(grouping=grouping,stratum=name,**{k:group.get(k) for k in ('n_users','status','imprecise')},
                                   comparisons={n:_aggregate(r) for n,r in group.get('comparisons',{}).items()}))
        lines += ['Saved stratum summaries: '+_json(strata),
                  'Contamination: '+_json({k:diagnostic.get('contamination',{}).get(k) for k in ('status','reason','interpretation')}),
                  'Numerical evidence: '+_json([{k:r.get(k) for k in ('slot','status','dtype','backend','refitted','max_absolute_score_difference','rank_positions_changed','absolute_auc_change')} for r in diagnostic.get('numerical',[])]),
                  'Negative-mean norm scale diagnostics: '+_json(dict(quantiles=diagnostic.get('strata',{}).get('negative_mean_norm',{}).get('quantiles'),
                      associations={n:{k:r.get(k) for k in ('n_users','pearson_correlation')} for n,r in diagnostic.get('strata',{}).get('negative_mean_norm',{}).get('associations',{}).items()})),
                  'Separate contamination seeds (corrupted minus clean AUC): '+_json({seed:{name:_aggregate(r) for name,r in values.items()} for seed,values in diagnostic.get('contamination',{}).get('paired_by_seed',{}).items()}),
                  'Timing/amortization: '+_json({name:{**{k:{stat:v.get(stat) for stat in ('median','p95','repeat_sd')} for k,v in r.items() if k.endswith('_seconds') and isinstance(v,dict)},
                       'amortization_1000_requests':r.get('amortization_1000_requests')} for name,r in diagnostic.get('timing',{}).get('methods',{}).items()}),
                  'Shared cold/encoding cost: '+_json({k:diagnostic.get('timing',{}).get(k) for k in ('cold_shared_seconds','cold_total_seconds','shared_item_embedding_seconds','formula_operations','policy','rss_method','peak_process_tree_rss_bytes','os_max_rss_bytes')})]
    else: lines += ['Strata, precision robustness, and contamination evidence unavailable; not executed/verified.']
    lines += ['', '## 7. Full resource accounting',
        'Saved permanent ledger costs include failed and interrupted work, setup separately. Embedding=G2; development fitting/tuning/scoring=G3; confirmation=G4; diagnostics=G5. Stage totals include shared overhead and cannot be presented as isolated fit costs.',
        'If the permanent ledger is absent, actual costs are unavailable: displayed zeros are recorded totals only, not evidence of zero work.',
        'Measured shared cold/encoding costs: '+_json(calibration.get('timings',{}))+'; measured calibration fit-plus-score seconds by family: '+_json(fitting_costs)+'. Dominant saved term: '+str(max({**calibration.get('timings',{}),**fitting_costs},key={**calibration.get('timings',{}),**fitting_costs}.get) if calibration.get('timings') or fitting_costs else 'unavailable')+'. No unmeasured cost breakdown is inferred.',
        'Accounting snapshot as-of ledger event count '+str(len(state.get('events',[])))+'; SHA256 '+canonical_sha256({'events':state.get('events',[])})+'. Repeated CLI reporting extends this ledger; scientific tables retain the same saved values.',
        _json(cost_rows),
        'Cumulative experimental seconds: '+str(state.get('experimental_seconds',0))+'; reserved attempted evaluations: '+str(state.get('attempted_evaluations',0))+'; actual confirmation evaluations: '+str(saved.get('actual_evaluations','unavailable') if saved else 'unavailable')+'. Reservations are not actual executed user/configuration attempts.',
        'Peak process-tree RSS bytes: '+str(state.get('peak_rss_bytes','unavailable'))+'; peak project disk bytes: '+str(state.get('peak_project_disk_bytes','unavailable'))+'; baseline disk bytes: '+str(state.get('baseline_disk_bytes','unavailable')),
        'Effective reallocations: '+_json([{k:r.get(k) for k in ('source_stage','target_stage','seconds','reason')} for r in state.get('reallocations',[])])+'; protected reporting seconds: '+str(state.get('report_reserve_seconds',1800)),
        'Amortization: 1,000 requests, fit each request and fit reused are separate scenarios only when measured diagnostics exist; item encoding is shared cold cost. No unsupported catalog extrapolation.',
        '', '## 8. Correctness, failures and limitations',
        'Evidence qualifications: '+_json(problems),
        'Confirmation stop reason: '+str(saved.get('stop_reason') if saved else 'not executed/unavailable'),
        'Saved method failures (aggregate counts; no automatic retry): '+_json(failure_rows),
        'Persistent journal validity/correction reasons: '+_json(qualifications),
        'Failed/unfinished ledger work: '+_json([{k:a.get(k) for k in ('stage','status','elapsed_seconds','evaluations')} for a in attempts.values() if a['status']!='completed']),
        'Invalid/missing baselines and bounded tuning limit complete-suite claims. Endpoint coverage: '+_json(selected.get('endpoint_coverage',{}))+'; boundary winners: '+_json(selected.get('boundary_winners',[]))+'; missing required families: '+_json(selected.get('missing_required_families',[]))+'; incomplete configurations: '+str(len(selected.get('incomplete_configuration_ids',[]))),
        'Numerical tolerances: '+_json(payload.get('config',{}).get('numerics','not available'))+'; deviations: '+_json(payload.get('deviations',[])),
        'Leakage protection requires frozen development selection, verified lock and exposure journal. Historical reading validates saved anchors, not current implementation correctness. Float32 timing does not change the canonical float64 accuracy backend. Wider data/representation coverage remains unresolved.',
        '', '## 9. Prior-art boundary',
        'The following boundaries and links are inherited from docs/blueprint.md (planning evidence); literature was not freshly verified during reporting.',
        '[Weston et al., 2013](https://doi.org/10.1145/2507157.2507209) precedes max-over-interests; [PinnerSage](https://arxiv.org/abs/2007.03634) uses learned graph vectors; [Giacinto and Roli](https://proceedings.neurips.cc/paper/2004/file/5acdc9ca5d99ae66afdfe1eea0e3b26b-Paper.pdf) precedes nearest relevant/nonrelevant distance ratios; [Science Concierge](https://doi.org/10.1371/journal.pone.0158423) precedes signed centroid feedback. [Wang et al., 2026](https://doi.org/10.24963/ijcai.2026/339) uses paired positive/negative interests within a larger reasoning pipeline.',
        'This study tests separate maxima on capped frozen content histories under matched controls. No first-ever claim or unrestricted maxout guarantee. Observed-rating AUC does not establish online satisfaction or resolve exposure bias.',
        '', '## 10. Unresolved questions and next step',
        'Unresolved: mechanism attribution, wider representation/data coverage, online utility and any missing/invalid evidence identified above.',
        'Next step: '+('resolve the recorded validity failure before comparative interpretation.' if outcome=='invalid experiment' else 'assess the frozen evidence limitations before deciding whether any further study is warranted.'),
        'Local raw evidence stays in ignored results/data/cache directories. Public exports contain aggregates and artifact hashes only; they exclude outcomes, raw MovieLens content, vectors, candidate IDs and per-item scores. See README for recreation and license boundaries.','']
    report_path=root/'report.md';report_path.write_text('\n\n'.join(lines),encoding='utf-8')
    for path in [report_path,*map(Path,tables),*map(Path,figures)]: note(path,'generated aggregate export')
    inventory_path=root/'artifact-inventory.json'
    inventory_path.write_text(_json(dict(schema_version=1,validation='historical saved evidence; no outcome permit',artifacts=inventory))+'\n')
    if ledger: ledger.close()
    return dict(status='completed',outcome=outcome,report=str(report_path),tables=tables,figures=figures,inventory=str(inventory_path),limitations=problems)
