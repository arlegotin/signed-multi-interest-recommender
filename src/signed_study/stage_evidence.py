"""Small persisted stage records and current-source correctness evidence."""
from dataclasses import fields
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

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
        atomic_write_json(config.paths['results']/'correctness.json',result)
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


def read_stage(config,stage,inputs=None):
    path=config.paths['results']/'stages'/f'{stage}.json'
    record=json.loads(path.read_text())
    verify_record(record)
    if record['status']!='completed':
        raise ValueError(f'{stage} prerequisite is incomplete')
    if inputs is not None and record['input_sha256']!=canonical_sha256(inputs):
        raise ValueError(f'{stage} inputs changed; preserve old evidence and resolve before rerunning')
    for ref in record['artifacts'].values():
        verify_reference(ref)
    return record


def save_stage(config,stage,inputs,artifacts,attempt_ids,**extra):
    record=dict(schema_version=1,stage=stage,status='completed',stop_reason=None,
                input_sha256=canonical_sha256(inputs),inputs=inputs,
                artifacts={name:file_reference(path) for name,path in artifacts.items()},
                attempt_ids=attempt_ids,**extra)
    record['output_sha256']=canonical_sha256(record)
    atomic_write_json(config.paths['results']/'stages'/f'{stage}.json',record)
    return record
