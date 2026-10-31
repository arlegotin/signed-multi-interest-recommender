"""Small persisted stage records and current-source correctness evidence."""
from dataclasses import fields
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import uuid
import warnings

from .budget import atomic_write_json
from .protocol import canonical_sha256


def plain(value):
    if hasattr(value, 'items'):
        return {str(k): plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [plain(v) for v in value]
    return str(value) if isinstance(value,Path) else value


def config_record(config):
    return {f.name: plain(getattr(config,f.name)) for f in fields(config)}


def environment_pins():
    """Exact installed native-environment versions; exclude this editable project."""
    return {d.metadata['Name'].lower().replace('_','-'): d.version
            for d in importlib.metadata.distributions()
            if d.metadata['Name'].lower().replace('_','-') != 'signed-study'}


def write_environment_lock(path):
    text=''.join(f'{name}=={version}\n' for name,version in sorted(environment_pins().items()))
    if path.exists() and path.read_text()!=text:
        raise ValueError('Existing environment lock differs from installed native versions')
    temporary=path.with_suffix('.txt.tmp')
    with temporary.open('w') as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return text


def verify_environment_lock(path):
    text=''.join(f'{name}=={version}\n' for name,version in sorted(environment_pins().items()))
    if path.read_text()!=text:
        raise ValueError('Installed environment no longer matches acquisition pins')


def file_reference(path):
    path=Path(path)
    with path.open('rb') as handle:
        digest=hashlib.file_digest(handle,'sha256').hexdigest()
    return {'path':str(path),'sha256':digest}


def verify_reference(reference):
    if file_reference(reference['path']) != reference:
        raise ValueError(f"Artifact integrity discrepancy: {reference['path']}")


def code_identity():
    root=Path(__file__).resolve().parents[2]
    paths=sorted([*root.joinpath('src').rglob('*.py'), *root.joinpath('tests').rglob('*.py'),
                  root/'pyproject.toml',root/'requirements.lock.txt'])
    return {str(p.relative_to(root)):file_reference(p)['sha256'] if p.exists() else None for p in paths}


def correctness_command():
    return [sys.executable,'-m','pytest','-q','--basetemp=.cache/correctness-pytest']


def correctness_inputs(config):
    environment=config.paths['results']/'environment.json'
    lock=Path('requirements.lock.txt')
    return {'code':code_identity(),'config':config_record(config),
            'environment':file_reference(environment) if environment.exists() else None,
            'environment_lock':file_reference(lock) if lock.exists() else None}


def record_correctness(config,ledger):
    """Run the entire synthetic suite once and bind success to live inputs.

    Invoke directly, never inside another ledger-owned command/stage. A plain
    historic pytest ledger entry is intentionally insufficient for G2.
    """
    with ledger.work('G1',1,0) as evidence_attempt:
        before=correctness_inputs(config)
    old=set(ledger.snapshot()['attempts'])
    command=correctness_command()
    rc=ledger.run_command(command,stage='G1',projected_seconds=30)
    attempts=[a for a in ledger.snapshot()['attempts'] if a not in old]
    with ledger.work('G1',1,0) as write_attempt:
        unchanged=before==correctness_inputs(config)
        result=dict(schema_version=1,status='completed' if rc==0 and unchanged else 'failed',
                    input_sha256=canonical_sha256(before),inputs=before,command=command,
                    attempt_ids=[evidence_attempt,*attempts,write_attempt],command_attempt_ids=attempts,
                    returncode=rc,stop_reason=None if rc==0 and unchanged else 'Correctness failed or inputs changed')
        result['output_sha256']=canonical_sha256(result)
        path=config.paths['results']/'correctness.json'
        if path.exists():
            previous=json.loads(path.read_text())
            verify_record(previous)
            archive=path.with_name(f"correctness-{previous['output_sha256']}.json")
            if archive.exists() and archive.read_bytes()!=path.read_bytes():
                raise ValueError('Historical correctness archive differs from original')
            if not archive.exists():
                with archive.open('xb') as handle:
                    handle.write(path.read_bytes()); handle.flush(); os.fsync(handle.fileno())
        atomic_write_json(path,result)
    return result


def verify_correctness(config,ledger):
    result=json.loads((config.paths['results']/'correctness.json').read_text())
    verify_record(result)
    if result['status']!='completed' or result['input_sha256']!=canonical_sha256(correctness_inputs(config)):
        raise ValueError('G1 correctness evidence is stale or unsuccessful')
    state=ledger.snapshot()
    if any(state['attempts'][a]['status']!='completed' or state['attempts'][a]['stage']!='G1'
           for a in result['attempt_ids']):
        raise ValueError('G1 evidence includes an unfinished or failed accounting attempt')
    for aid in result['command_attempt_ids']:
        a=state['attempts'][aid]
        if (a['stage']!='G1' or a['status']!='completed' or a['outputs']['returncode']!=0
                or a['outputs']['command']!=result['command']):
            raise ValueError('G1 correctness command lacks successful permanent ledger evidence')
    if not result['command_attempt_ids'] or result['command']!=correctness_command():
        raise ValueError('G1 evidence requires the complete correctness command')
    return result


def verify_record(record):
    expected=canonical_sha256({k:v for k,v in record.items() if k!='output_sha256'})
    if record.get('schema_version')!=1 or record.get('output_sha256')!=expected:
        raise ValueError('Stage evidence digest/schema mismatch')


_SUPERSEDABLE_STAGES=('calibrate','smoke','develop')


def _receipt_reference(root,stage,reference,*,legacy=False):
    path=Path(reference['path'])
    if (path.parent.resolve()!=root.resolve() or path.suffix!='.json' or
            not (path.name.startswith(stage+'-') or legacy and path.name==stage+'.json')):
        raise ValueError('Invalid pre-freeze active receipt reference')
    verify_reference(reference)
    record=json.loads(path.read_text()); verify_record(record)
    if record.get('stage')!=stage:
        raise ValueError('Pre-freeze receipt stage identity mismatch')
    return path


def _completed_legacy_supersession(config,stage,*,supersedes=None):
    """Detect old completed replacements, never select one by filename or time.

    Before selection witnesses existed, successful terminal accounting plus the
    replacement's own attempt IDs is the durable evidence of supersession.
    Orphaned records without completed accounting do not supersede anything.
    """
    candidates=[]
    for path in (config.paths['results']/'stages').glob(stage+'-*.json'):
        row=json.loads(path.read_text())
        if (row.get('stage')==stage and row.get('status')=='completed' and row.get('supersedes') is not None
                and (supersedes is None or row['supersedes']==supersedes)):
            verify_record(row)
            candidates.append(row)
    ledger=config.paths['results']/'budget_ledger.jsonl'
    if not candidates:
        return False
    if not ledger.exists():
        warnings.warn('Receipt replacement interrupted before selection; retaining the previous publication',RuntimeWarning)
        return False
    terminal={}
    for line in ledger.read_text().splitlines():
        row=json.loads(line)
        if row.get('event')=='finish':
            terminal[row['attempt_id']]=row['status']
    completed=any(row.get('attempt_ids') and all(terminal.get(a)=='completed'
                  for a in row['attempt_ids']) for row in candidates)
    if not completed:
        warnings.warn('Receipt replacement interrupted before selection; retaining the previous publication',RuntimeWarning)
    return completed


def stage_path(config,stage):
    """Resolve only the three mutable pre-freeze receipt roles, with hash checks."""
    root=config.paths['results']/'stages'
    selector=root/f'{stage}.active.json'
    selection=root/f'{stage}.selection.json'
    if stage in _SUPERSEDABLE_STAGES and selection.exists():
        witness=json.loads(selection.read_text()); verify_record(witness)
        if witness.get('stage')!=stage or witness.get('status') not in ('pending','completed'):
            raise ValueError('Invalid pre-freeze selection witness')
        target=_receipt_reference(root,stage,witness['receipt'])
        previous=_receipt_reference(root,stage,witness['previous'],legacy=True)
        if json.loads(target.read_text()).get('supersedes')!=witness['previous']:
            raise ValueError('Receipt selection witness supersession identity mismatch')
        value=json.loads(selector.read_text()) if selector.exists() else None
        if value is not None:
            verify_record(value)
            if value.get('stage')!=stage:
                raise ValueError('Invalid pre-freeze active receipt reference')
        reference=None if value is None else value.get('receipt')
        if witness['status']=='completed':
            if reference!=witness['receipt']:
                raise ValueError('Completed receipt selection has a missing or stale active selector')
        else:
            # The previous publication stays available after a failed pointer
            # replacement. Never treat the not-yet-committed target as active.
            if reference==witness['previous'] or reference is None and previous.name==stage+'.json':
                warnings.warn('Receipt selection interrupted; retaining the previous publication',RuntimeWarning)
                return previous
            raise ValueError('Receipt selection interrupted before commit; previous publication retained in selection witness')
    if stage in _SUPERSEDABLE_STAGES and selector.exists():
        value=json.loads(selector.read_text()); verify_record(value)
        if value.get('stage')!=stage:
            raise ValueError('Invalid pre-freeze active receipt reference')
        if not selection.exists() and _completed_legacy_supersession(config,stage,supersedes=value['receipt']):
            raise ValueError('Historical active selector is stale after completed receipt supersession')
        return _receipt_reference(root,stage,value['receipt'])
    if stage in _SUPERSEDABLE_STAGES and _completed_legacy_supersession(config,stage):
        raise ValueError('Completed receipt supersession has a missing active selector')
    return root/f'{stage}.json'


def read_stage(config,stage,inputs=None):
    path=stage_path(config,stage)
    record=json.loads(path.read_text())
    verify_record(record)
    if record.get('stage')!=stage:
        raise ValueError('Stage receipt identity mismatch')
    if record['status']!='completed':
        raise ValueError(f'{stage} prerequisite is incomplete')
    if inputs is not None and record['input_sha256']!=canonical_sha256(inputs):
        raise ValueError(f'{stage} inputs changed; preserve old evidence and resolve before rerunning')
    for ref in record['artifacts'].values():
        verify_reference(ref)
    return record


def save_stage(config,stage,inputs,artifacts,attempt_ids,*,supersedes=None,**extra):
    record=dict(schema_version=1,stage=stage,status='completed',stop_reason=None,
                input_sha256=canonical_sha256(inputs),inputs=inputs,
                artifacts={name:file_reference(path) for name,path in artifacts.items()},
                attempt_ids=attempt_ids,**extra)
    if supersedes is not None:
        if stage not in _SUPERSEDABLE_STAGES:
            raise ValueError('Only calibration/smoke/development receipts may be superseded')
        root=config.paths['results']
        if any((root/name).exists() for name in ('protocol.lock.json','protocol.lock.sha256',
                'confirmation.exposure.json','confirmation.events.jsonl')):
            raise ValueError('Receipt supersession is prohibited after freeze or exposure')
        verify_reference(supersedes)
        if supersedes!=file_reference(stage_path(config,stage)):
            raise ValueError('Supersession must name the exact current receipt')
        record['supersedes']=supersedes
    record['output_sha256']=canonical_sha256(record)
    root=config.paths['results']/'stages'
    path=root/(f'{stage}-{uuid.uuid4().hex}.json' if supersedes else f'{stage}.json')
    atomic_write_json(path,record)
    if supersedes:
        selector=dict(schema_version=1,stage=stage,receipt=file_reference(path))
        selector['output_sha256']=canonical_sha256(selector)
        # The independent witness distinguishes an orphan replacement from a
        # committed selection, including deletion of the active selector later.
        selection=dict(schema_version=1,stage=stage,status='pending',
                       receipt=selector['receipt'],previous=supersedes)
        selection['output_sha256']=canonical_sha256(selection)
        atomic_write_json(root/f'{stage}.selection.json',selection)
        atomic_write_json(root/f'{stage}.active.json',selector)
        selection['status']='completed'
        selection.pop('output_sha256')
        selection['output_sha256']=canonical_sha256(selection)
        atomic_write_json(root/f'{stage}.selection.json',selection)
    return record
