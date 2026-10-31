"""Independent score oracles and exact reference rankings for optional retrieval."""
import numpy as np
import pytest

from signed_study.models import signed_scores, fit_user, score_user, UserModel
from signed_study.data import History


def fixture(n=80, d=32):
    from signed_study.retrieval import GramCatalog
    X = np.random.default_rng(4271).normal(size=(n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    ids = np.arange(100, 100+n)
    return GramCatalog(ids, X, build_block_size=7), X, ids


@pytest.mark.parametrize('penalty', [0., 1., 8., 1e100])
@pytest.mark.parametrize('positive,negative', [(1, 0), (1, 1), (3, 5)])
def test_scores_match_independent_scalar_oracle(penalty, positive, negative):
    catalog, X, ids = fixture()
    P = X[:positive]
    chosen = ids[10:10+negative]
    candidates = ids[[12, 2, 12, 9, 30]]
    actual = catalog.scores(P, chosen, penalty, candidate_ids=iter(candidates), chunk_size=3)
    expected = [max(sum(float(a)*float(b) for a,b in zip(x,p)) for p in P)
                - (penalty * max(sum(float(a)*float(b) for a,b in zip(x,n))
                   for n in X[10:10+negative]) if negative else 0.) for x in X[[12,2,12,9,30]]]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=64*np.finfo(float).eps*(1+penalty))


@pytest.mark.parametrize('chunk', [1, 7, 512])
def test_cancellation_strict_ranking_is_full_reference(chunk):
    catalog, X, ids = fixture()
    result = catalog.topk(X[[3]], ids[[3]], 1, k=10, chunk_size=chunk)
    reference = signed_scores(X, X[[3]], X[[3]], 1, chunk_size=chunk)
    order = np.lexsort((ids, -reference))[:10]
    np.testing.assert_array_equal(result.item_ids, ids[order])
    np.testing.assert_array_equal(result.scores, reference[order])
    assert result.diagnostics['mode'] == 'strict'
    assert result.diagnostics['reference_fallback'] is True
    fast = catalog.topk(X[[3]], ids[[3]], 1, k=10, mode='fast', chunk_size=chunk)
    scores = catalog.scores(X[[3]], ids[[3]], 1, chunk_size=chunk)
    np.testing.assert_array_equal(fast.item_ids, ids[np.lexsort((ids, -scores))[:10]])


def test_filter_order_empty_duplicate_and_ties():
    from signed_study.retrieval import GramCatalog
    X = np.array([[1.,0.], [1.,0.], [-1.,0.], [0.,1.]])
    c = GramCatalog([9, 2, 3, 4], X)
    np.testing.assert_array_equal(c.rows([4,9,4]), X[[3,0,3]])
    assert c.rows([]).shape == (0,2)
    assert c.scores(X[:1], [], 0, candidate_ids=[]).shape == (0,)
    assert c.topk(X[:1], [3], 1, k=2).item_ids.tolist() == [2,9]
    assert c.topk(X[:1], [], 0, candidate_ids=[4,9,2], k=3).item_ids.tolist() == [2,9,4]
    with pytest.raises(ValueError, match='distinct'):
        c.topk(X[:1], [], 0, candidate_ids=[9,9], k=1)
    for k in (0, 5, True, 1.5):
        with pytest.raises(ValueError): c.topk(X[:1], [], 0, k=k)


def test_exact_integer_ids_and_mutation_isolation():
    from signed_study.retrieval import GramCatalog
    X = np.eye(3)
    ids = np.array([-(2**63), 2**53+1, 2**63-1], dtype=np.int64)
    c = GramCatalog(ids, X)
    X[:] = 0; ids[:] = 0
    for public in (c.item_ids, c.vectors):
        with pytest.raises(ValueError): public.setflags(write=True)
    rows = c.rows([2**53+1]); rows[:] = 0
    np.testing.assert_array_equal(c.rows([2**53+1]), [[0,1,0]])
    for malformed in ([True], [1.], [[1]], [2**63], [-(2**63)-1]):
        with pytest.raises(ValueError): c.rows(malformed)
    with pytest.raises(KeyError): c.rows([2**53])
    with pytest.raises(ValueError): GramCatalog(np.array([2**63], dtype=np.uint64), [[1.,0.]])


@pytest.mark.parametrize('block', [1, 3, 100])
def test_memory_refusal_precedes_quadratic_allocation(block, monkeypatch):
    from signed_study import retrieval
    original = np.empty
    allocated = []
    def empty(shape, *args, **kw):
        allocated.append(shape)
        return original(shape, *args, **kw)
    monkeypatch.setattr(retrieval.np, 'empty', empty)
    with pytest.raises(MemoryError): retrieval.GramCatalog([1,2,3], np.eye(3), max_gram_bytes=71)
    assert (3,3) not in allocated
    c = retrieval.GramCatalog([1,2,3], np.eye(3), max_gram_bytes=72, build_block_size=block)
    assert c.memory['gram_bytes'] == 72
    assert c.memory['catalog_bytes'] == 72


def test_reject_bad_geometry_and_overflow():
    from signed_study.retrieval import GramCatalog
    for X in (np.empty((0,2)), [[2.,0.]], [[float('nan'),0.]]):
        with pytest.raises(ValueError): GramCatalog(range(len(X)), X)
    with pytest.raises(ValueError): GramCatalog([1], [[1.+0j,0j]])
    c = GramCatalog([1,2], [[1.,0.], [-1.,0.]])
    for p in (-1, float('inf'), True):
        with pytest.raises(ValueError): c.scores([[1.,0.]], [1], p)
    for P in ([], [[0.,0.]], [[1.,0.,0.]]):
        with pytest.raises(ValueError): c.scores(P, [], 0)
    c = GramCatalog([1], [[1.+3e-13,0.]])  # Within the declared unit tolerance.
    with pytest.raises(ValueError, match='nonfinite'):
        c.topk([[1.,0.]], [1], np.finfo(float).max, k=1)


@pytest.mark.parametrize('family,parameters', [('SignedMax', {'K':2,'lambda':2}),
    ('AllExemplarMargin', {'lambda':1}), ('PositiveMax', {'K':2}), ('Centroid', {})])
def test_real_fit_adapter_and_strict_model_ranking(family, parameters, local_config):
    from signed_study.retrieval import score_user_cached
    c, X, ids = fixture(n=24, d=384)
    h = History(1, 1, 'development', tuple(ids[:10]), tuple(ids[10:15]))
    model = fit_user(family, h, c, parameters, config=local_config)
    chosen = ids[[23, 4, 7, 18, 2]]
    reference = score_user(model, chosen, c, chunk_size=3)
    np.testing.assert_allclose(score_user_cached(model, chosen, c, chunk_size=3), reference,
                               rtol=0, atol=384*16*np.finfo(float).eps*3)
    strict = c.topk_user(model, chosen, k=3, chunk_size=3)
    order = np.lexsort((chosen, -reference))[:3]
    np.testing.assert_array_equal(strict.item_ids, chosen[order])
    np.testing.assert_array_equal(strict.scores, reference[order])
    if len(model.state['N']):
        model.state['N'][0] *= -1
        with pytest.raises(ValueError, match='support'): score_user_cached(model, chosen, c)


def test_adapter_rejects_failed_unsupported_and_wrong_support():
    from signed_study.retrieval import score_user_cached
    c, X, ids = fixture()
    for model in (UserModel('RBFSVM', {}, (), {}, {'status':'ok'}),
        UserModel('SignedMax', {}, (), {}, {'status':'failed'}),
        UserModel('SignedMax', {'lambda':1}, (ids[2],), {'P':X[:1], 'N':X[1:2]}, {'status':'ok'})):
        with pytest.raises(ValueError): score_user_cached(model, ids, c)


@pytest.mark.parametrize('parameters,diagnostics,state', [
    ({}, {'status':'ok'}, 'valid'),
    ({'K':2,'lambda':1,'invented':0}, {'status':'ok'}, 'valid'),
    (None, {'status':'ok'}, 'valid'),
    ({'K':2,'lambda':1}, None, 'valid'),
    ({'K':2,'lambda':1}, {'status':'ok'}, None)])
def test_adapter_rejects_malformed_mutable_records(parameters, diagnostics, state):
    from signed_study.retrieval import score_user_cached
    c, X, ids = fixture()
    state = {'P':X[:1], 'N':X[1:2]} if state=='valid' else state
    model=UserModel('SignedMax',parameters,(int(ids[0]),int(ids[1])),state,diagnostics)
    with pytest.raises(ValueError): score_user_cached(model, ids, c)


@pytest.mark.parametrize('build_block', [1, 7, 128])
@pytest.mark.parametrize('chunk', [1, 7, 128])
def test_multiple_cancelling_supports_preserve_all_reference_ties(build_block,chunk):
    from signed_study.retrieval import GramCatalog
    X=np.random.default_rng(731).normal(size=(24,32)); X/=np.linalg.norm(X,axis=1,keepdims=True)
    X[-1]=X[0]  # Different IDs retain duplicate geometry.
    ids=np.arange(24,dtype=np.int64)+100
    catalog=GramCatalog(ids,X,build_block_size=build_block)
    support=[3,7,1]; P=X[support]
    candidates=ids[[23,18,0,6,2,17,1,8,3,9]]
    reference=signed_scores(catalog.rows(candidates),P,P,1,chunk_size=chunk)
    np.testing.assert_array_equal(reference,np.zeros(len(candidates)))
    strict=catalog.topk(P,ids[support],1,candidate_ids=candidates,k=7,chunk_size=chunk)
    np.testing.assert_array_equal(strict.item_ids,np.sort(candidates)[:7])
    np.testing.assert_array_equal(strict.scores,np.zeros(7))
    cached=catalog.scores(P,ids[support],1,candidate_ids=candidates,chunk_size=chunk)
    np.testing.assert_allclose(cached,0,rtol=0,atol=32*16*np.finfo(float).eps*2)


def test_raw_negative_cosines_are_retained_with_independent_exact_values():
    from signed_study.retrieval import GramCatalog
    X=np.array([[.8,.6],[.8,-.6],[-1.,0.]])
    catalog=GramCatalog([2,1,3],X)
    np.testing.assert_array_equal(catalog.scores([[1.,0.]],[3],1),[1.6,1.6,-2.])
    np.testing.assert_array_equal(catalog.scores([[1.,0.]],[3],0),[.8,.8,-1.])
    np.testing.assert_array_equal(catalog.scores([[1.,0.]],[],7),[.8,.8,-1.])
    assert catalog.topk([[1.,0.]],[3],1,k=2).item_ids.tolist()==[1,2]


def test_near_and_internal_boundary_ties_use_exact_score_order():
    from signed_study.retrieval import GramCatalog
    cosines=np.array([.5,np.nextafter(.5,-np.inf),np.nextafter(.5,np.inf),.5,-.5])
    X=np.column_stack((cosines,np.sqrt(1-cosines*cosines)))
    ids=np.array([40,20,30,10,50])
    catalog=GramCatalog(ids,X)
    requested=ids[[4,1,3,0,2]]
    result=catalog.topk([[1.,0.]],[],0,candidate_ids=requested,k=3,chunk_size=2)
    assert result.item_ids.tolist()==[30,10,40]
    assert result.scores[0]>result.scores[1]==result.scores[2]
    assert catalog.rows([20])[0,0]<result.scores[-1]


def test_unsigned_representable_ids_keep_distinct_large_integer_identity():
    from signed_study.retrieval import GramCatalog
    ids=np.array([2**53,2**53+1,2**63-1],dtype=np.uint64)
    catalog=GramCatalog(ids,np.eye(3))
    np.testing.assert_array_equal(catalog.rows(ids[[1,0,1]]),np.eye(3)[[1,0,1]])
    with pytest.raises(KeyError): catalog.rows(np.array([2**53+2],dtype=np.uint64))
    with pytest.raises(ValueError): catalog.rows(np.array([2**63],dtype=np.uint64))


@pytest.mark.parametrize('invalid', [0,-1,True,1.5])
def test_invalid_chunk_and_build_sizes_are_rejected(invalid):
    from signed_study.retrieval import GramCatalog
    with pytest.raises(ValueError): GramCatalog([1,2],np.eye(2),build_block_size=invalid)
    catalog=GramCatalog([1,2],np.eye(2))
    with pytest.raises(ValueError): catalog.scores([[1.,0.]],[2],1,chunk_size=invalid)
    with pytest.raises(ValueError): catalog.topk([[1.,0.]],[2],1,k=1,chunk_size=invalid)
    model=UserModel('SignedMax',{'K':1,'lambda':1},(1,2),{'P':np.eye(2)[:1],'N':np.eye(2)[1:]},{'status':'ok'})
    with pytest.raises(ValueError): catalog.topk_user(model,[1,2],k=1,chunk_size=invalid)


def test_adapter_revalidates_parameter_positive_and_support_mutations(local_config):
    from dataclasses import replace
    from signed_study.retrieval import score_user_cached
    c,X,ids=fixture(n=24,d=384)
    history=History(1,1,'development',tuple(ids[:10]),tuple(ids[10:15]))
    model=fit_user('SignedMax',history,c,{'K':2,'lambda':2},config=local_config)
    score_user_cached(model,ids,c)
    model.parameters['lambda']=0
    model.state['P'][:]=X[20]
    reference=score_user(model,ids,c,chunk_size=3)
    np.testing.assert_allclose(score_user_cached(model,ids,c,chunk_size=3),reference,
                               rtol=0,atol=384*16*np.finfo(float).eps)
    result=c.topk_user(model,ids,k=7,chunk_size=3)
    np.testing.assert_array_equal(result.item_ids,ids[np.lexsort((ids,-reference))[:7]])
    model.parameters['lambda']=-1
    with pytest.raises(ValueError): score_user_cached(model,ids,c)
    model.parameters['lambda']=1
    wrong=replace(model,support_ids=(*model.support_ids[:-1],int(ids[23])))
    with pytest.raises(ValueError,match='support'): score_user_cached(wrong,ids,c)
    model.state['P'][0]*=2
    with pytest.raises(ValueError): score_user_cached(model,ids,c)


@pytest.mark.parametrize('method,parameters',[('SignedMax',{'K':2,'lambda':2}),
    ('AllExemplarMargin',{'lambda':2}),('PositiveMax',{'K':2}),('Centroid',{})])
def test_real_adapter_empty_negatives_has_only_positive_geometry(method,parameters,local_config):
    from signed_study.retrieval import score_user_cached
    c,_,ids=fixture(n=24,d=384)
    model=fit_user(method,History(1,1,'development',tuple(ids[:10]),()),c,parameters,config=local_config)
    assert model.state['N'].shape==(0,384)
    expected=score_user(model,ids,c,chunk_size=7)
    np.testing.assert_allclose(score_user_cached(model,ids,c,chunk_size=7),expected,
                               rtol=0,atol=384*16*np.finfo(float).eps)
    result=c.topk_user(model,ids,k=5,chunk_size=7)
    np.testing.assert_array_equal(result.item_ids,ids[np.lexsort((ids,-expected))[:5]])


@pytest.mark.parametrize('penalty',[1+0j,1+1j,'1',pytest.param(2**2048,id='int-beyond-float64')])
def test_non_real_or_unrepresentable_penalty_rejected(penalty):
    c,_,_=fixture(n=4,d=3)
    with pytest.raises(ValueError): c.scores(c.vectors[:1],[],penalty)
