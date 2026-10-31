#!/usr/bin/env python3
"""Read a verified local Torch cache and development histories; never acquire assets."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from signed_study.budget import Ledger
from signed_study.calibration import user_digest, _verify_cohort_binding
from signed_study.data import Movie, load_episode_store, _EXPECTED_AUDIT
from signed_study.embed import expected_embedding_manifest, load_embedding_cache
from signed_study.protocol import load_config
from signed_study.stage_evidence import file_reference, read_stage


PINNED_ENCODER_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


def verified_inputs(config, *, users=32):
    """Read terminal acquisition/audit evidence and earlier histories only.

    Ledger construction/snapshot are read-only: preparation never admits work,
    reconciles interrupted owners, changes the campaign or loads outcome arrays.
    Historical source-bound G1/freeze evidence is not reinterpreted by this tool.
    """
    if type(users) is not int or not 1<=users<=32:
        raise ValueError('users must be between 1 and 32')
    ledger=Ledger(config.paths['results']/'budget_ledger.jsonl',config)
    try:
        state=ledger.snapshot()
    finally:
        ledger.close()
    acquisition=read_stage(config,'acquire')
    audit=read_stage(config,'audit',{'acquire':acquisition['output_sha256']})
    store=load_episode_store(Path(audit['artifacts']['audit']['path']),Path(audit['artifacts']['episodes']['path']))
    _verify_cohort_binding(config,store,state)
    name=store.audit['dataset']
    expected_counts=_EXPECTED_AUDIT[name]
    if any(len(getattr(store,role))!=expected_counts[f'funnel.{role}_users']
           for role in ('development','confirmation')):
        raise ValueError('Comparable local preparation requires the full audited official cohort')
    def value(name):
        return json.loads(Path(acquisition['artifacts'][name]['path']).read_text())
    resolved=load_config(Path(acquisition['artifacts']['resolved_config.json']['path']))
    metadata={row['item_id']:Movie(row['item_id'],row['title'],tuple(row['genres']))
              for row in value('metadata.json')['movies']}
    expected=expected_embedding_manifest(metadata,resolved,
        encoder_provenance=value('encoder_acquisition.json'),dataset_manifest=value('data_manifest.json'))
    if expected['inputs']['model_revision']!=PINNED_ENCODER_REVISION:
        raise ValueError('Comparable mode requires the published pinned encoder revision')
    cache=load_embedding_cache(config.paths['cache']/'embeddings',expected)
    catalog_ids=set(map(int,cache.item_ids))
    histories=[e.history for e in sorted(store.development,key=lambda e:user_digest(e.history.user_id))[:users]]
    if len(histories)!=users:
        raise ValueError('The verified development cohort cannot supply the requested users')
    for history in histories:
        supports=history.positive_ids+history.negative_ids
        if (history.role!='development' or not 10<=len(history.positive_ids)<=128
                or not 5<=len(history.negative_ids)<=64 or len(set(supports))!=len(supports)
                or not set(supports)<=catalog_ids):
            raise ValueError('Verified development support identity/caps differ from the catalog')
    return cache,histories,expected,acquisition,audit


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('config/study.json'))
    parser.add_argument('--output',type=Path,default=Path('.cache/retrieval/local.npz'))
    parser.add_argument('--users',type=int,default=32)
    args=parser.parse_args(argv)
    if args.users < 1 or args.users > 32:
        parser.error('users must be between 1 and 32')
    if args.output.suffix!='.npz':
        parser.error('output must have the .npz suffix')
    # Real history mappings must stay ignored. There is no public export mode.
    if subprocess.run(['git','check-ignore','-q',str(args.output)],check=False).returncode:
        parser.error('output must be a Git-ignored local artifact (for example .cache/retrieval/local.npz)')
    config=load_config(args.config)
    cache,histories,expected,acquisition,audit=verified_inputs(config,users=args.users)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output,item_ids=cache.item_ids,vectors=cache.vectors,
             positive_ids=np.array([list(h.positive_ids)+[-1]*(128-len(h.positive_ids)) for h in histories],dtype=np.int64),
             negative_ids=np.array([list(h.negative_ids)+[-1]*(64-len(h.negative_ids)) for h in histories],dtype=np.int64),
             positive_counts=np.array([len(h.positive_ids) for h in histories]),
             negative_counts=np.array([len(h.negative_ids) for h in histories]),
             user_ids=np.array([h.user_id for h in histories]))
    sources=[config.paths['results']/'stages'/'acquire.json',config.paths['results']/'stages'/'audit.json',
             *[Path(r['path']) for r in acquisition['artifacts'].values()],
             *[Path(r['path']) for r in audit['artifacts'].values()],
             config.paths['cache']/'embeddings'/'manifest.json',
             config.paths['cache']/'embeddings'/'vectors.npy',config.paths['cache']/'embeddings'/'item_ids.npy']
    descriptor=dict(schema_version=1,kind='verified_local_torch_development',
        bundle=file_reference(args.output.resolve()),source_references=[file_reference(p.resolve()) for p in sources],
        manifest_sha256=file_reference(config.paths['cache']/'embeddings'/'manifest.json')['sha256'],
        encoder_revision=expected['inputs']['model_revision'],
        encoder_runtime_versions=expected['inputs']['runtime_versions'],
        cohort_order='first development histories in SHA256 order; no outcome arrays opened',
        user_count=len(histories))
    args.output.with_suffix('.json').write_text(json.dumps(descriptor,indent=2,sort_keys=True)+'\n')
    print(f'Prepared {len(histories)} development histories in ignored local bundle {args.output}; no outcomes opened')


if __name__=='__main__':
    main()
