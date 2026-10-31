#!/usr/bin/env python3
"""Offline source-bound retrieval timing; synthetic by default, no accuracy claims."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import tracemalloc

import numpy as np
import psutil
from threadpoolctl import threadpool_info, threadpool_limits

from signed_study.data import History
from signed_study.embed import EmbeddingCache
from signed_study.models import fit_user, score_user, signed_scores
from signed_study.protocol import canonical_sha256, load_config
from signed_study.retrieval import GramCatalog, score_user_cached
from signed_study.retrieval_benchmark import summarize
from signed_study.stage_evidence import config_record, file_reference, verify_reference


def digest(array):
    return hashlib.sha256(np.asarray(array).tobytes(order='C')).hexdigest()


def source_identity():
    root=Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.run(['git',*args],cwd=root,check=True,capture_output=True).stdout
    paths=sorted([*root.joinpath('src').rglob('*.py'),*root.joinpath('scripts').glob('*.py'),root/'pyproject.toml'])
    diff=git('diff','--binary','HEAD')
    return dict(commit=git('rev-parse','HEAD').decode().strip(),
        dirty_diff_sha256=hashlib.sha256(diff).hexdigest(),
        status_sha256=hashlib.sha256(git('status','--porcelain')).hexdigest(),
        executed_source_sha256={str(p.relative_to(root)):file_reference(p)['sha256'] for p in paths})


def cpu_evidence():
    """Retain available CPU model evidence, including explicit unavailable status."""
    if platform.system()=='Darwin':
        try:
            result=subprocess.run(['sysctl','-n','machdep.cpu.brand_string'],capture_output=True,
                                  text=True,timeout=5,check=False)
            return dict(method='sysctl machdep.cpu.brand_string',returncode=result.returncode,
                        model=result.stdout.strip() or None,error=result.stderr.strip() or None)
        except (OSError,subprocess.TimeoutExpired) as error:
            return dict(method='sysctl machdep.cpu.brand_string',model=None,error=str(error))
    path=Path('/proc/cpuinfo')
    if path.exists():
        try:
            for line in path.read_text().splitlines():
                if line.startswith('model name'):
                    return dict(method='/proc/cpuinfo model name',model=line.split(':',1)[1].strip())
        except OSError as error:
            return dict(method='/proc/cpuinfo model name',model=None,error=str(error))
    return dict(method='platform.processor',model=platform.processor() or None)


class Geometry:
    """Already normalized verified geometry; avoid a second normalization path."""
    def __init__(self,ids,X):
        self.item_ids=np.array(ids,copy=True)
        self.vectors=np.array(X,copy=True)
        self._index={int(item):i for i,item in enumerate(ids)}
    def rows(self,ids):
        requested=list(ids)
        if any(type(item) is not int and not isinstance(item,np.integer) for item in requested):
            raise ValueError('Embedding row lookup requires integer IDs')
        return self.vectors[[self._index[int(i)] for i in requested]]


def local_inputs(path):
    from signed_study.calibration import user_digest
    path=Path(path)
    description=json.loads(path.with_suffix('.json').read_text())
    if description.get('schema_version')!=1 or description.get('kind')!='verified_local_torch_development':
        raise ValueError('Unrecognized local bundle; run prepare_retrieval_benchmark.py')
    if (description.get('encoder_revision')!='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
            or type(description.get('user_count')) is not int or not 1<=description['user_count']<=32
            or not isinstance(description.get('encoder_runtime_versions'),dict)
            or not description['encoder_runtime_versions'].get('torch')
            or not isinstance(description.get('source_references'),list) or not description['source_references']):
        raise ValueError('Local bundle lacks pinned Torch provenance or a valid user count')
    verify_reference(description['bundle'])
    if Path(description['bundle']['path']).resolve()!=path.resolve():
        raise ValueError('Bundle path differs from verified descriptor')
    for reference in description['source_references']:
        verify_reference(reference)
    container=np.load(path,allow_pickle=False)
    if not isinstance(container,np.lib.npyio.NpzFile):
        raise ValueError('Local bundle must be an NPZ archive container')
    with container as arrays:
        fields={'item_ids','vectors','user_ids','positive_ids','negative_ids','positive_counts','negative_counts'}
        if len(arrays.files)!=len(fields) or set(arrays.files)!=fields:
            raise ValueError('Local bundle array schema differs from the verified format')
        values={name:arrays[name] for name in fields}
        for name in fields-{'vectors'}:
            if values[name].dtype!=np.dtype(np.int64):
                raise ValueError('Local bundle IDs and counts must be exact signed int64 arrays')
        ids,X,u=values['item_ids'],values['vectors'],values['user_ids']
        count=description['user_count']
        if (ids.ndim!=1 or not len(ids) or np.any(ids<=0) or len(np.unique(ids))!=len(ids)
                or X.dtype!=np.dtype(np.float64) or X.shape!=(len(ids),384)
                or not np.isfinite(X).all() or np.any(np.abs(np.linalg.norm(X,axis=1)-1)>1e-12)
                or u.shape!=(count,) or np.any(u<=0) or len(np.unique(u))!=count
                or list(map(int,u))!=sorted(map(int,u),key=user_digest)
                or values['positive_ids'].shape!=(count,128) or values['negative_ids'].shape!=(count,64)
                or values['positive_counts'].shape!=(count,) or values['negative_counts'].shape!=(count,)):
            raise ValueError('Local bundle geometry, ordered users or support dimensions are invalid')
        catalog_ids=set(map(int,ids))
        for row in range(count):
            active=[]
            for role,minimum,cap in (('positive',10,128),('negative',5,64)):
                n=int(values[role+'_counts'][row]); support=values[role+'_ids'][row]
                if not minimum<=n<=cap or np.any(support[n:]!=-1):
                    raise ValueError('Local bundle support eligibility, caps or padding is invalid')
                chosen=list(map(int,support[:n]))
                if not set(chosen)<=catalog_ids:
                    raise ValueError('Local bundle support IDs are absent from verified geometry')
                active.extend(chosen)
            if len(set(active))!=len(active):
                raise ValueError('Local bundle support IDs must be distinct across both classes')
        cache=Geometry(ids,X)
        histories=[History(int(uid),0,'development',tuple(map(int,p[:int(pc)])),tuple(map(int,n[:int(nc)])))
            for uid,p,n,pc,nc in zip(u,values['positive_ids'],values['negative_ids'],
                                   values['positive_counts'],values['negative_counts'],strict=True)]
    # Public provenance contains hashes and runtime only, never local path/user mappings.
    public={k:description[k] for k in ('kind','manifest_sha256','encoder_revision','encoder_runtime_versions','cohort_order')}
    public['bundle_sha256']=description['bundle']['sha256']
    return cache,histories,public


def component_probe(method,model,cache,catalog,candidates,args,expected):
    """Diagnostic phase probe; public API timings remain the comparison boundary.

    These probes retain validated chunks until scoring. Their allocation/cache
    behavior differs from streaming APIs; their times are never subtracted from
    public query time or used to claim API acceleration.
    """
    from signed_study.models import _item_ids, _positive_integer, _score_inputs, _unit_rows
    from signed_study.retrieval import _model_query
    started=time.perf_counter_ns()
    P,N=model.state['P'],model.state['N']
    query=None
    if method in ('gram_fast','adapter_cached'):
        if method=='adapter_cached':
            P,negative_ids,penalty=_model_query(model,catalog)
        else:
            negative_ids=model.support_ids[-len(N):] if len(N) else ()
            penalty=2
        query=catalog._query(P,negative_ids,penalty,candidates,args.chunk)
    else:
        _positive_integer(args.chunk,'chunk_size')
        if method=='adapter_reference':
            ids=_item_ids(candidates)
            if model.diagnostics['status']!='ok':raise ValueError('Failed model')
            blocks=[]
            for start in range(0,len(ids),args.chunk):
                X=np.ascontiguousarray(_unit_rows(cache.rows(ids[start:start+args.chunk]),'candidates'))
                X,P,N=_score_inputs(X,P,N)
                blocks.append(X)
        else:
            if method=='gram_strict':
                q=catalog._query(P,model.support_ids[-len(N):] if len(N) else (),2,candidates,args.chunk)
                P,_,ni,_,_,ci,_=q
                X,P,N=_score_inputs(catalog.vectors[ci],P,catalog.vectors[ni])
            else:
                X,P,N=_score_inputs(cache.rows(candidates),P,N)
            chunk=len(candidates) if method=='dense' else args.chunk
            blocks=[X[start:start+chunk] for start in range(0,len(X),chunk)]
    validated=time.perf_counter_ns()
    if query is not None:
        P,_,ni,penalty,ids,ci,chunk=query
        scores=catalog._scores(P,ni,penalty,ids,ci,chunk)
    else:
        scores=np.empty(len(candidates),dtype=np.float64)
        offset=0
        for X in blocks:
            values=np.max(X@P.T,axis=1)
            if len(N):values-=2*np.max(X@N.T,axis=1)
            scores[offset:offset+len(X)]=values
            offset+=len(X)
        if not np.isfinite(scores).all():raise ValueError('Nonfinite component scores')
    scored=time.perf_counter_ns()
    if method in ('gram_fast','gram_strict'):
        catalog._rank(candidates,scores,args.k,'fast' if method=='gram_fast' else 'strict')
    else:
        np.lexsort((candidates,-scores))[:args.k]
    ranked=time.perf_counter_ns()
    error=float(np.max(np.abs(scores-expected),initial=0.))
    tolerance=16*cache.vectors.shape[1]*np.finfo(float).eps*3
    if error>tolerance:raise ValueError('Component kernel differs from public score API')
    return dict(status='ok',validation_gather_seconds=(validated-started)/1e9,
        score_kernel_seconds=(scored-validated)/1e9,ranking_seconds=(ranked-scored)/1e9,
        max_score_error=error)


def run(args):
    source_before=source_identity()
    preparation_start=time.perf_counter()
    if args.local_bundle:
        cache,histories,provenance=local_inputs(args.local_bundle)
        if args.users > len(histories):
            raise ValueError('Local bundle has fewer users than requested; prepare enough users or set --users explicitly')
        histories=histories[:args.users]
    else:
        rng=np.random.default_rng(args.seed)
        raw=rng.normal(size=(args.items,384))
        cache=EmbeddingCache(np.arange(1,args.items+1),raw,{})
        histories=[]
        for ordinal in range(args.users):
            support=rng.permutation(cache.item_ids)
            n=7+(ordinal*13)%58
            histories.append(History(ordinal+1,0,'development',tuple(map(int,support[:128])),
                                     tuple(map(int,support[128:128+n]))))
        provenance=dict(kind='synthetic',seed=args.seed,encoder='none')
    X,ids=cache.vectors,cache.item_ids
    config=load_config(args.config)
    preparation_seconds=time.perf_counter()-preparation_start
    process=psutil.Process()
    rss_before=process.memory_info().rss
    tracemalloc.start()
    build_start=time.perf_counter()
    catalog=GramCatalog(ids,X,build_block_size=args.build_block,
                        manifest_sha256=provenance.get('manifest_sha256'))
    build_seconds=time.perf_counter()-build_start
    _,build_peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    rss_after=process.memory_info().rss
    methods=('reference','dense','gram_fast','gram_strict','adapter_reference','adapter_cached')
    observations=[];checks=[];workloads=[];fits=[];errors=[];components=[]
    query_peaks={method:0 for method in methods}
    for ordinal,history in enumerate(histories):
        started=time.perf_counter()
        model=fit_user('SignedMax',history,cache,{'K':2,'lambda':2},config=config)
        fits.append(dict(ordinal=ordinal,seconds=time.perf_counter()-started,status=model.diagnostics['status']))
        if model.diagnostics['status']!='ok':
            errors.append(dict(ordinal=ordinal,stage='fit',diagnostics=model.diagnostics))
        for workload in ('full','capped_support_excluded'):
            excluded=set(history.positive_ids+history.negative_ids) if workload!='full' else set()
            candidates=np.array([i for i in ids if int(i) not in excluded],dtype=np.int64)
            P=model.state.get('P');N=model.state.get('N')
            def ranked(scores):
                order=np.lexsort((candidates,-scores))[:args.k]
                return candidates[order],scores[order]
            def invoke(method):
                if method=='reference':
                    return ranked(signed_scores(cache.rows(candidates),P,N,2,chunk_size=args.chunk))
                if method=='dense':
                    return ranked(signed_scores(cache.rows(candidates),P,N,2,chunk_size=max(1,len(candidates))))
                if method in ('gram_fast','gram_strict'):
                    result=catalog.topk(P,history.negative_ids,2,candidate_ids=candidates,k=args.k,
                                        chunk_size=args.chunk,mode='fast' if method=='gram_fast' else 'strict')
                    return result.item_ids,result.scores
                if method=='adapter_reference':
                    return ranked(score_user(model,candidates,cache,chunk_size=args.chunk))
                return ranked(score_user_cached(model,candidates,catalog,chunk_size=args.chunk))
            def score_array(method):
                if method=='reference':return signed_scores(cache.rows(candidates),P,N,2,chunk_size=args.chunk)
                if method=='dense':return signed_scores(cache.rows(candidates),P,N,2,chunk_size=max(1,len(candidates)))
                if method=='gram_fast':return catalog.scores(P,history.negative_ids,2,candidate_ids=candidates,chunk_size=args.chunk)
                if method=='gram_strict':return signed_scores(catalog.rows(candidates),P,catalog.rows(history.negative_ids),2,chunk_size=args.chunk)
                if method=='adapter_reference':return score_user(model,candidates,cache,chunk_size=args.chunk)
                return score_user_cached(model,candidates,catalog,chunk_size=args.chunk)
            workloads.append(dict(workload=workload,ordinal=ordinal,candidate_count=len(candidates),
                positive_count=len(history.positive_ids),negative_count=len(history.negative_ids),
                candidate_ids_sha256=digest(candidates),positive_ids_sha256=digest(history.positive_ids),
                negative_ids_sha256=digest(history.negative_ids),
                filter='all catalog items' if workload=='full' else 'exclude capped fitted supports only'))
            try:
                reference=signed_scores(cache.rows(candidates),P,N,2,chunk_size=args.chunk)
                reference_ids=candidates[np.lexsort((candidates,-reference))[:args.k]]
                tolerance=16*X.shape[1]*np.finfo(float).eps*3
                expected_scores={}
                for method in methods:
                    got,_=invoke(method)
                    values=score_array(method)
                    expected_scores[method]=values
                    checks.append(dict(workload=workload,ordinal=ordinal,method=method,
                        topk_mismatch=not np.array_equal(reference_ids,got),
                        max_score_error=float(np.max(np.abs(reference-values),initial=0.)),
                        score_tolerance=tolerance,numeric_accepted=bool(np.all(np.abs(reference-values)<=tolerance)),
                        reference_fallback=method=='gram_strict',
                        boundary_gap=float(np.sort(reference)[-args.k]-np.sort(reference)[-args.k-1])
                            if len(reference)>args.k else None))
                    if (not checks[-1]['numeric_accepted'] or method in
                            ('reference','gram_strict','adapter_reference') and checks[-1]['topk_mismatch']):
                        errors.append(dict(workload=workload,ordinal=ordinal,method=method,
                            stage='correctness',exception='Method failed its declared numerical/reference ranking contract'))
                    tracemalloc.start()
                    try:
                        invoke(method)
                        _,peak=tracemalloc.get_traced_memory()
                    finally:
                        tracemalloc.stop()
                    query_peaks[method]=max(query_peaks[method],peak)
            except Exception as error:
                errors.append(dict(workload=workload,ordinal=ordinal,stage='correctness',
                                   exception=f'{type(error).__name__}: {error}'))
            for repeat in range(-args.warmups,args.repeats):
                rotation=(ordinal+repeat+args.warmups)%len(methods)
                order=methods[rotation:]+methods[:rotation]
                for position,method in enumerate(order):
                    started=time.perf_counter_ns()
                    try:
                        invoke(method)
                        elapsed=(time.perf_counter_ns()-started)/1e9
                        result=dict(seconds=elapsed,status='ok')
                    except Exception as error:
                        result=dict(seconds=None,status='failed',exception=f'{type(error).__name__}: {error}')
                    if repeat>=0:
                        observations.append(dict(workload=workload,ordinal=ordinal,repeat=repeat,
                                                 method=method,order_position=position,**result))
            for repeat in range(args.repeats):
                rotation=(ordinal+repeat)%len(methods)
                for method in methods[rotation:]+methods[:rotation]:
                    try:
                        value=component_probe(method,model,cache,catalog,candidates,args,expected_scores[method])
                    except Exception as error:
                        value=dict(status='failed',exception=f'{type(error).__name__}: {error}')
                        errors.append(dict(workload=workload,ordinal=ordinal,method=method,
                                           stage='component_probe',exception=value['exception']))
                    components.append(dict(workload=workload,ordinal=ordinal,repeat=repeat,method=method,**value))
    source_after=source_identity()
    if source_before != source_after:
        errors.append(dict(stage='source_changed',exception='Source changed during benchmark; timings are not a stable-source result'))
    blas=io.StringIO()
    with redirect_stdout(blas):np.show_config()
    result=dict(schema_version=1,recorded_at=datetime.now(timezone.utc).isoformat(),
        source=source_before,source_unchanged=source_before==source_after,
        source_after=source_after if source_after!=source_before else None,input=provenance,seed=args.seed,
        configuration=dict(K=2,penalty=2,k=args.k,items=len(ids),dimensions=X.shape[1],dtype=str(X.dtype),
            users=len(histories),warmups=args.warmups,repeats=args.repeats,methods=list(methods),
            workload_names=['full','capped_support_excluded'],
            fit_config_sha256=canonical_sha256(config_record(config)),config_file_sha256=file_reference(args.config)['sha256'],
            reference_chunk=args.chunk,dense_chunk='entire requested candidate list',
            query_chunk=args.chunk,build_block=args.build_block),
        identity=dict(ordered_catalog_ids_sha256=digest(ids),geometry_sha256=digest(X),
                      catalog_sha256=catalog.identity_sha256),
        runtime=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
            machine=platform.machine(),processor=platform.processor(),cpu_count=os.cpu_count(),
            cpu_evidence=cpu_evidence(),numpy_blas_configuration=blas.getvalue(),
            threadpools=threadpool_info(),thread_limit=1,
            thread_environment={key:os.environ.get(key) for key in
                ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
            thread_qualification='threadpoolctl observes supported pools; Accelerate thread count is not independently verified'),
        boundaries=dict(preparation_seconds=preparation_seconds,build_seconds=build_seconds,
            fitting=fits,inference_seconds=0,
            query='API validation, row gathering, scoring and deterministic full sort/top-k; excludes correctness checks and fitting',
            adapter='adapter_cached includes per-call model validation and negative row equality',
            validation='included in timed APIs; no universal floating-point separation certificate',
            components=f'{args.repeats} rotated probes after query repetitions: validation/gather, prevalidated scoring, ranking; chunk retention differs from public APIs; never subtract these times from query time'),
        memory=dict(**catalog.memory,build_tracemalloc_peak_bytes=build_peak,
            query_tracemalloc_peak_bytes=query_peaks,rss_before_build=rss_before,rss_after_build=rss_after,
            note='tracemalloc includes NumPy allocations; RSS samples are endpoints, not a process peak'),
        workloads=workloads,observations=observations,components=components,correctness=checks,exceptions=errors)
    result['summary']=summarize(observations,build_seconds=build_seconds,
                                expected_repeats=args.repeats,expected_ordinals=range(len(histories)),
                                expected_methods=methods,expected_workloads=['full','capped_support_excluded'])
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--items',type=int,default=3883)
    parser.add_argument('--users',type=int,default=32)
    parser.add_argument('--warmups',type=int,default=1)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--seed',type=int,default=4271)
    parser.add_argument('--k',type=int,default=10)
    parser.add_argument('--chunk',type=int,default=512)
    parser.add_argument('--build-block',type=int,default=256)
    parser.add_argument('--local-bundle',type=Path)
    parser.add_argument('--config',type=Path,default=Path('config/study.json'))
    parser.add_argument('--output',type=Path,default=Path('.cache/retrieval/synthetic.json'))
    args=parser.parse_args(argv)
    if args.items<256 or not 1<=args.users<=32 or args.warmups<1 or args.repeats<5 or args.k<1 or args.k>args.items-192:
        parser.error('Require items>=256, users=1..32, warmups>=1, repeats>=5 and a valid k after support exclusion')
    with threadpool_limits(limits=1):
        result=run(args)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),exceptions=len(result['exceptions']),summary=result['summary']),sort_keys=True))
    return 1 if result['exceptions'] or any(r['status']!='ok' for r in result['observations']) else 0


if __name__=='__main__':
    raise SystemExit(main())
