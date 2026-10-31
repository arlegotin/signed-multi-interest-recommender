"""Immutable canonical protocol and independently recomputed live identities."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from . import data, stage_evidence as evidence
from .calibration import planned_comparison_schedule, select_calibration_users
from .embed import load_embedding_cache
from .protocol import ProtocolMismatch, VerifiedProtocol, canonical_sha256, load_config


TABLE_SCHEMAS = {
    'attempt': ['user_id','slot','configuration_id','fit_family','parameters','status',
                'candidate_ids','scores','metrics','diagnostics','exception','elapsed_seconds'],
    'attempt_optional_geometry': ['prototype_identity','fitted_state'],
    'fitted_state': ['schema_version','lock_sha256','run_id','user_id','slot','fit_family','parameters',
                     'support_ids','positive_ids','negative_ids','positive_vectors','prototype_identity',
                     'cache_sha256','diagnostics','output_sha256'],
    'metrics': ['auc','ndcg10','negative_at_10','tie_fraction'],
    'comparison': ['sample_mean_low','sample_mean_high','n_users','seed','resamples','batch_size','standard_error','delta','user_ids','n_complete','n_missing','lower_bounds','upper_bounds',
                   'low','high','labels','complete_case'],
    'qualification': ['comparison_scope','interpretation','complete_suite_eligible','endpoint_coverage','boundary_winners'],
}


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def exclusive_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def source_identity():
    root=Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.run(['git','-C',str(root),*args],check=True,capture_output=True,text=True).stdout.strip()
    if git('status','--porcelain','--untracked-files=all'):
        raise ProtocolMismatch('Freeze/confirmation requires clean committed source')
    return dict(commit=git('rev-parse','HEAD'),tree=git('rev-parse','HEAD^{tree}'),
                clean=True,content=evidence.code_identity())


def _context(config,ledger):
    # All these reads concern provenance/history bytes, never semantic outcomes.
    from .stages import _acquired, _prerequisite
    source=source_identity()
    if source.get('clean') is not True or not source.get('commit') or not source.get('content'):
        raise ProtocolMismatch('Clean committed source/content proof is required')
    correctness=evidence.verify_correctness(config,ledger)
    acquired,_,_,_,_,expected=_acquired(config,ledger)
    audit=_prerequisite(config,'audit',ledger,{'acquire':acquired['output_sha256']})
    calibration=_prerequisite(config,'calibrate',ledger)
    if (calibration['inputs']['acquire']!=acquired['output_sha256'] or
            calibration['inputs']['audit']!=audit['output_sha256']):
        raise ProtocolMismatch('Calibration acquisition/audit changed')
    smoke=_prerequisite(config,'smoke',ledger,{'calibrate':calibration['output_sha256'],'development_users':8})
    stage=_prerequisite(config,'develop',ledger,{'calibrate':calibration['output_sha256'],'smoke':smoke['output_sha256']})
    resolved=load_config(config.paths['results']/'calibration_config.json')
    store=data.load_episode_store(audit['artifacts']['audit']['path'],audit['artifacts']['episodes']['path'])
    cache=load_embedding_cache(resolved.paths['cache']/'embeddings',expected)
    development=json.loads(Path(stage['artifacts']['development']['path']).read_text())
    evidence.verify_record(development)
    measurements=json.loads((config.paths['results']/'calibration.json').read_text())
    evidence.verify_record(measurements)
    if development['input_sha256']!=canonical_sha256(dict(config=evidence.config_record(resolved),calibration=measurements['output_sha256'])):
        raise ProtocolMismatch('Development resolved policy/calibration changed')
    refs={name:evidence.file_reference(config.paths['results']/name) for name in
          ('environment.json','resolved_config.json','calibration_config.json','calibration.json')}
    refs.update({f'stage_{name}':evidence.file_reference(config.paths['results']/'stages'/f'{name}.json')
                 for name in ('acquire','audit','calibrate','smoke','develop')})
    # Verify all selected outcome bytes before allowing even the first array load.
    outcome_refs={str(uid):evidence.file_reference(store.paths['confirmation_outcomes']/f'{uid}.npy')
                  for uid in development['confirmation_user_ids']}
    for uid,ref in outcome_refs.items():
        if ref['sha256']!=store.audit['outputs']['outcome_sha256']['confirmation'][uid]:
            raise ProtocolMismatch('Confirmation outcome provenance changed')
    identities=dict(source=source,config=evidence.config_record(config),resolved_config=evidence.config_record(resolved),
                    environment=evidence.environment_pins(),artifacts=refs,
                    correctness_sha256=correctness['output_sha256'],data_sha256=data.episode_store_digest(store),
                    outcomes=outcome_refs,cache_sha256=cache.manifest['output_sha256'],
                    model=expected['inputs'],development_sha256=development['output_sha256'],
                    selection_sha256=canonical_sha256(_frozen_development(development)),
                    confirmation_user_ids=development['confirmation_user_ids'],
                    table_schemas_sha256=canonical_sha256(TABLE_SCHEMAS),
                    metric_sha256=evidence.file_reference(Path(__file__).with_name('metrics.py'))['sha256'],
                    resources_sha256=ledger.snapshot()['resources_sha256'])
    return dict(identities=identities,ledger_events=ledger.snapshot()['events']),resolved,store,cache,development,stage


def _frozen_development(saved):
    # Raw arrays remain in verified development/batch files, never duplicated in
    # the immutable control document or parsed on every confirmation attempt.
    keys=('status','development_user_ids','confirmation_user_ids','configurations',
          'families','S_star','B_star','matched_controls','comparison_schedule',
          'endpoint_coverage','initial_boundary_winners','boundary_winners',
          'incomplete_configuration_ids','missing_required_families',
          'repair_decisions','complete_suite_eligible','planned_precision',
          'history_statistics','history_thresholds','selected_forecast',
          'confirmation_outcomes_accessed','artifact_name')
    return {key:saved[key] for key in keys}


def live_inputs(config,ledger):
    """Recompute identities from disk/environment without exposing label arrays."""
    try:
        return _context(config,ledger)[0]
    except (ValueError,KeyError,OSError) as error:
        raise ProtocolMismatch(str(error)) from error


def _validate_schedule(development,config,store,ledger):
    schedule=development['comparison_schedule']
    expected=planned_comparison_schedule()
    if (development['status']!='completed' or development['missing_required_families'] or
            development['confirmation_outcomes_accessed'] is not False or
            [s['name'] for s in schedule]!=[s['name'] for s in expected]):
        raise ProtocolMismatch('Required complete selected/control schedule is missing')
    for slot,base in zip(schedule,expected):
        selected=development['matched_controls' if base['role']=='matched_control' else 'families'][base['name']]
        if slot!={**base,**selected} or slot['fit_family']!=base['fit_family']:
            raise ProtocolMismatch('Executable selected schedule differs from saved selections')
        if not slot.get('prototype_identity') or (slot['geometric'] and not slot.get('development_prototype_identities')):
            raise ProtocolMismatch('Missing actual development prototype evidence')
    if development['S_star']!=development['families']['SignedMax'] or development['B_star']!=development['families'].get(development['B_star']['family']) or development['B_star']['family']=='SignedMax':
        raise ProtocolMismatch('Invalid selected primary comparator')
    p=development['S_star']['parameters']
    controls=development['matched_controls']
    for name,parameters in [('matched_MeanNegative',p),('matched_PositiveMax',{'K':p['K'],'lambda':0}),('matched_K1_SignedMax',{'K':1,'lambda':p['lambda']})]:
        if controls[name]['parameters']!=parameters:
            raise ProtocolMismatch('Matched control parameters changed')
    for name in ('matched_MeanNegative','matched_PositiveMax'):
        if controls[name]['development_prototype_identities']!=development['S_star']['development_prototype_identities']:
            raise ProtocolMismatch('Matched actual prototypes differ')
    for role in ('development','confirmation'):
        users=development[f'{role}_user_ids']
        if not users or any(type(u) is not int for u in users) or len(set(users))!=len(users) or not set(users)<={e.history.user_id for e in getattr(store,role)}:
            raise ProtocolMismatch('Invalid planned user IDs')
    forecast=development['selected_forecast']; state=ledger.snapshot(); caps=config.resources
    seconds=forecast['seconds']; evaluations=forecast['evaluations']
    timing=select_calibration_users(list(store.development))
    numerical=development['confirmation_user_ids'][:64]
    diagnostic_count=len(timing)*13*(config.timing['warmups']+config.timing['repetitions'])+len(numerical)*sum(s['geometric'] for s in schedule)
    if (forecast['status']!='feasible' or forecast['comparison_schedule']!=schedule or forecast['comparison_entries']!=13
            or forecast['timing_user_ids']!=timing or forecast['numerical_user_ids']!=numerical
            or evaluations['G5']!=diagnostic_count
            or type(forecast['catalog_items']) is not int or forecast['catalog_items']<=0
            or forecast['cost_chunk_size']!=config.numerics['score_chunk_size']
            or evaluations['G4']!=len(development['confirmation_user_ids'])*13
            or forecast['safety_multiplier']!=caps['safety_multiplier']):
        raise ProtocolMismatch('Selected forecast does not price the executable schedule')
    from .diagnostics import APPLICABILITY, CONTAMINATION_METHODS
    plan=forecast.get('contamination',{})
    if (plan.get('user_ids')!=development['confirmation_user_ids'][:config.contamination['max_users']]
            or plan.get('seeds')!=list(config.contamination_seeds)
            or plan.get('schedule')!=[s for s in schedule if s['name'] in CONTAMINATION_METHODS]
            or plan.get('applicability')!=APPLICABILITY or plan.get('minimum_subgroup_users')!=100
            or plan.get('numerical_auc_threshold')!=config.delta
            or plan.get('minimum_extra_seconds')!=config.contamination['minimum_extra_seconds']
            or plan.get('evaluations')!=len(plan.get('user_ids',[]))*12
            or not isinstance(plan.get('seconds'),(int,float)) or not math.isfinite(plan['seconds']) or plan['seconds']<0):
        raise ProtocolMismatch('Conditional diagnostic plan is not frozen completely')
    if any(not math.isfinite(seconds[s]) or seconds[s]<0 or state['stage_seconds'][s]+caps['safety_multiplier']*seconds[s]>state['stage_limits'][s] for s in ('G4','G5')) or state['attempted_evaluations']+sum(evaluations.values())>caps['evaluation_limit'] or state['experimental_seconds']+caps['safety_multiplier']*sum(seconds.values())>caps['experimental_seconds']-caps['report_reserve_seconds']:
        raise ProtocolMismatch('Selected schedule no longer fits effective resource limits')


def freeze_protocol(config,development,ledger):
    path=config.paths['results']/'protocol.lock.json'
    if path.exists() or path.with_suffix('.sha256').exists():
        raise FileExistsError('Immutable protocol lock already exists')
    with ledger.work('G3',1,0) as aid:
        live,resolved,store,_,saved,stage=_context(config,ledger)
        if development!=stage and development!=saved:
            raise ProtocolMismatch('Freeze requires the saved completed development evidence')
        _validate_schedule(saved,resolved,store,ledger)
        calibration=evidence.read_stage(config,'calibrate')
        if calibration['inputs']['correctness']!=live['identities']['correctness_sha256']:
            raise ProtocolMismatch('Calibration does not bind the current G1 proof')
        payload=dict(schema_version=1,identities=live['identities'],config=evidence.config_record(resolved),
                     invocation_config=evidence.config_record(config),development=_frozen_development(saved),freeze_attempt_id=aid,
                     data_access=dict(confirmation_user_ids=saved['confirmation_user_ids'],episodes_sha256=data.episode_store_digest(store)),
                     table_schemas=TABLE_SCHEMAS,
                     tie_rules=dict(selection='maximum anchored within 1e-6; K,lambda,C,g,lexicographic configuration ID',
                                    comparator='maximum macro development AUC; exact tie lexicographic family ID',
                                    metric='exact stored-score ties',ranking='ascending_item_id'),
                     deviations=evidence.plain(resolved.pre_freeze_repairs),
                     ledger_snapshot=ledger.snapshot(),ledger_path=str(ledger.path))
        exclusive_json(path,payload)
        with path.with_suffix('.sha256').open('x') as handle:
            handle.write(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
            handle.flush(); os.fsync(handle.fileno())
        ledger.finish(aid,'completed',{}, {'protocol_lock':evidence.file_reference(path),
                                        'sidecar':evidence.file_reference(path.with_suffix('.sha256'))})
    return path


def read_lock(path):
    try:
        raw=Path(path).read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        payload=json.loads(raw)
        if (Path(path).with_suffix('.sha256').read_text().strip()!=digest or
                raw!=canonical_bytes(payload) or payload['schema_version']!=1):
            raise ProtocolMismatch('Immutable canonical lock/sidecar mismatch')
        return digest,payload
    except (ValueError,KeyError,OSError) as error:
        raise ProtocolMismatch(str(error)) from error


def verify_lock_accounting(payload,digest,live):
    prefix=payload['ledger_snapshot']['events']
    if live['ledger_events'][:len(prefix)]!=prefix:
        raise ProtocolMismatch('Frozen ledger history was removed or changed')
    finished=[e for e in live['ledger_events'] if e.get('event')=='finish' and e.get('attempt_id')==payload['freeze_attempt_id']]
    if (len(finished)!=1 or finished[0]['status']!='completed' or
            finished[0]['outputs'].get('protocol_lock',{}).get('sha256')!=digest):
        raise ProtocolMismatch('Lock differs from completed permanent freeze evidence')


def verify_protocol(lock_path,live_inputs):
    digest,payload=read_lock(lock_path)
    try:
        if payload['identities']!=live_inputs['identities'] or not live_inputs['identities']['source']['clean']:
            raise ProtocolMismatch('Live protocol identities differ from the immutable lock')
        verify_lock_accounting(payload,digest,live_inputs)
        return VerifiedProtocol(digest,payload)
    except (KeyError,TypeError) as error:
        raise ProtocolMismatch('Incomplete live protocol identities') from error
