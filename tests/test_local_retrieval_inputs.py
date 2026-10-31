"""Verified local preparation and exact saved-bundle schema; no outcome access."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from signed_study.budget import atomic_write_json
from signed_study.calibration import user_digest
from signed_study.protocol import canonical_sha256
from signed_study.stage_evidence import file_reference, read_stage
from test_stage_pipeline import pipeline


ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundle(tmp_path):
    path = tmp_path/'local.npz'
    ids = np.arange(1,257,dtype=np.int64)
    X = np.random.default_rng(731).normal(size=(len(ids),384))
    X /= np.linalg.norm(X,axis=1,keepdims=True)
    users = np.array(sorted([1,2],key=user_digest),dtype=np.int64)
    arrays = dict(item_ids=ids,vectors=X,user_ids=users,
        positive_ids=np.array([list(ids[:10])+[-1]*118 for _ in users],dtype=np.int64),
        negative_ids=np.array([list(ids[100:105])+[-1]*59 for _ in users],dtype=np.int64),
        positive_counts=np.full(2,10,dtype=np.int64),negative_counts=np.full(2,5,dtype=np.int64))
    source = tmp_path/'original-receipt.json'
    source.write_text('{"synthetic_format_fixture":true}\n')
    descriptor = dict(schema_version=1,kind='verified_local_torch_development',
        source_references=[file_reference(source)],manifest_sha256='a'*64,
        encoder_revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
        encoder_runtime_versions={'torch':'synthetic schema fixture'},
        cohort_order='first development histories in SHA256 order; no outcome arrays opened',user_count=2)
    def save():
        np.savez(path,**arrays)
        descriptor['bundle'] = file_reference(path)
        atomic_write_json(path.with_suffix('.json'),descriptor)
    save()
    return path,arrays,descriptor,save


def test_local_bundle_preserves_exact_geometry_supports_and_privacy(bundle):
    path, arrays, _, _ = bundle
    cache,histories,public = script('benchmark_retrieval').local_inputs(path)
    np.testing.assert_array_equal(cache.vectors,arrays['vectors'])
    np.testing.assert_array_equal(cache.item_ids,arrays['item_ids'])
    assert [h.user_id for h in histories] == arrays['user_ids'].tolist()
    assert all(h.role=='development' and h.positive_ids==tuple(range(1,11))
               and h.negative_ids==tuple(range(101,106)) for h in histories)
    assert 'user_ids' not in public and 'source_references' not in public and 'bundle' not in public


def test_local_bundle_requires_npz_container_before_interpreting_arrays(bundle):
    path,_,descriptor,_=bundle
    with path.open('wb') as handle:
        np.save(handle,np.eye(3),allow_pickle=False)
    descriptor['bundle']=file_reference(path)
    atomic_write_json(path.with_suffix('.json'),descriptor)
    with pytest.raises(ValueError,match='NPZ|archive|container'):
        script('benchmark_retrieval').local_inputs(path)


@pytest.mark.parametrize('damage', ['truncated_counts','float_ids','unsigned_overflow','duplicate_ids',
    'duplicate_users','unordered_users','wrong_user_count','too_many_users','wrong_dimensions','nonunit',
    'nonfinite','float_counts','bool_counts','positive_cap','negative_cap','positive_eligibility',
    'negative_eligibility','missing_support','duplicate_support','overlapping_classes','padding',
    'missing_array','extra_array','wrong_encoder','wrong_kind'])
def test_local_bundle_rejects_schema_or_identity_mismatch(bundle,damage):
    path, arrays, descriptor, save = bundle
    if damage=='truncated_counts': arrays['negative_counts'] = arrays['negative_counts'][:1]
    elif damage=='float_ids': arrays['item_ids'] = arrays['item_ids'].astype(float)
    elif damage=='unsigned_overflow':
        arrays['item_ids'] = arrays['item_ids'].astype(np.uint64); arrays['item_ids'][-1] = 2**63
    elif damage=='duplicate_ids': arrays['item_ids'][1] = arrays['item_ids'][0]
    elif damage=='duplicate_users': arrays['user_ids'][1] = arrays['user_ids'][0]
    elif damage=='unordered_users': arrays['user_ids'] = arrays['user_ids'][::-1]
    elif damage=='wrong_user_count': descriptor['user_count'] = 3
    elif damage=='too_many_users': descriptor['user_count'] = 33
    elif damage=='wrong_dimensions': arrays['vectors'] = arrays['vectors'][:,:32]
    elif damage=='nonunit': arrays['vectors'][0] *= 2
    elif damage=='nonfinite': arrays['vectors'][0,0] = np.nan
    elif damage=='float_counts': arrays['positive_counts'] = arrays['positive_counts'].astype(float)
    elif damage=='bool_counts': arrays['positive_counts'] = arrays['positive_counts'].astype(bool)
    elif damage=='positive_cap': arrays['positive_counts'][0] = 129
    elif damage=='negative_cap': arrays['negative_counts'][0] = 65
    elif damage=='positive_eligibility': arrays['positive_counts'][0] = 9
    elif damage=='negative_eligibility': arrays['negative_counts'][0] = 4
    elif damage=='missing_support': arrays['positive_ids'][0,0] = 999999
    elif damage=='duplicate_support': arrays['positive_ids'][0,0] = arrays['positive_ids'][0,1]
    elif damage=='overlapping_classes': arrays['negative_ids'][0,0] = arrays['positive_ids'][0,0]
    elif damage=='padding': arrays['negative_ids'][0,-1] = 2
    elif damage=='missing_array': arrays.pop('positive_counts')
    elif damage=='extra_array': arrays['confirmation_labels'] = np.array([1])
    elif damage=='wrong_encoder': descriptor['encoder_revision'] = 'b'*40
    elif damage=='wrong_kind': descriptor['kind'] = 'onnx_substitute'
    save()
    with pytest.raises(ValueError): script('benchmark_retrieval').local_inputs(path)


def acquired_audit(pipeline):
    config, ledger, _, _, _, stages, _, _ = pipeline
    assert stages.run_stage('acquire',config,ledger)['status']=='completed'
    assert stages.run_stage('audit',config,ledger)['status']=='completed'
    acquired,resolved,manifest,encoder,metadata,_ = stages._acquired(config,ledger)
    stages.encode_items(metadata,resolved,encoder_provenance=encoder,dataset_manifest=manifest,
                        encoder=stages.load_encoder(resolved,encoder))
    return config,ledger


@pytest.mark.parametrize('damage', ['acquire_failed','acquire_unfinished','audit_failed','audit_unfinished','audit_chain','audit_dataset'])
def test_preparation_rejects_unqualified_original_receipts(pipeline,monkeypatch,damage):
    config,ledger = acquired_audit(pipeline)
    module = script('prepare_retrieval_benchmark')
    monkeypatch.setattr(module,'PINNED_ENCODER_REVISION','a'*40,raising=False)
    role = 'acquire' if damage.startswith('acquire') else 'audit'
    receipt = read_stage(config,role)
    if damage.endswith(('failed','unfinished')):
        rows = [json.loads(line) for line in ledger.path.read_text().splitlines()]
        target = receipt['attempt_ids'][-1]
        if damage.endswith('failed'):
            for row in rows:
                if row.get('event')=='finish' and row.get('attempt_id')==target: row['status']='failed'
        else: rows = [row for row in rows if not(row.get('event')=='finish' and row.get('attempt_id')==target)]
        ledger.path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    elif damage=='audit_chain':
        receipt['inputs']['acquire']='f'*64
        receipt['input_sha256']=canonical_sha256(receipt['inputs'])
        receipt.pop('output_sha256'); receipt['output_sha256']=canonical_sha256(receipt)
        atomic_write_json(config.paths['results']/'stages'/'audit.json',receipt)
    else:
        path = Path(receipt['artifacts']['audit']['path'])
        row = json.loads(path.read_text()); row['dataset']='ml-100k'; atomic_write_json(path,row)
    with pytest.raises(ValueError): module.verified_inputs(config,users=2)


def test_preparation_reads_verified_development_records_without_outcomes(pipeline,monkeypatch):
    config,ledger = acquired_audit(pipeline)
    module = script('prepare_retrieval_benchmark')
    monkeypatch.setattr(module,'PINNED_ENCODER_REVISION','a'*40,raising=False)
    before = ledger.path.read_bytes()
    real_load = np.load
    def load(path,*a,**kw):
        assert 'outcomes' not in Path(path).parts
        return real_load(path,*a,**kw)
    monkeypatch.setattr(np,'load',load)
    cache,histories,expected,acquisition,audit = module.verified_inputs(config,users=2)
    assert len(cache.item_ids)==256
    assert [h.user_id for h in histories] == sorted(range(1,419),key=user_digest)[:2]
    assert acquisition['output_sha256']==audit['inputs']['acquire']
    assert expected['inputs']['model_revision']=='a'*40
    assert ledger.path.read_bytes()==before
