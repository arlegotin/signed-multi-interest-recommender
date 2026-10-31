"""Raw observations alone must reconstruct paired summaries and amortization."""
import pytest


def test_summary_uses_paired_user_medians_and_mean_savings():
    from signed_study.retrieval_benchmark import summarize
    observations = [dict(workload='full', ordinal=u, repeat=r, method=m, seconds=t,
                         status='ok') for u,d,g in ((0,2.,1.),(1,100.,10.),(2,3.,3.))
                    for r in range(5) for m,t in (('dense',d),('gram_fast',g))]
    result = summarize(observations, build_seconds=9.)['full']
    assert result['paired']['dense/gram_fast']['median_ratio'] == 2.
    assert result['methods']['dense']['median_seconds'] == 3.
    assert result['paired']['dense/gram_fast']['mean_savings_seconds'] == pytest.approx(91/3)
    assert result['paired']['dense/gram_fast']['build_amortization_queries'] == pytest.approx(27/91)


def test_failures_stay_in_denominator_and_nonpositive_savings_have_no_amortization():
    from signed_study.retrieval_benchmark import summarize
    rows = [dict(workload='excluded', ordinal=0, repeat=0, method=m, seconds=t, status='ok')
            for m,t in (('dense',1.),('gram_fast',2.))]
    rows += [dict(workload='excluded', ordinal=1, repeat=0, method='gram_fast', seconds=None,
                  status='failed', exception='deliberate')]
    value = summarize(rows, build_seconds=2.)['excluded']
    assert value['observed_ordinals'] == 2
    assert value['methods']['gram_fast']['failed_observations'] == 1
    assert value['paired']['dense/gram_fast']['paired_users'] == 1
    assert value['paired']['dense/gram_fast']['missing_users'] == 1
    assert value['paired']['dense/gram_fast']['build_amortization_queries'] is None


def test_duplicate_observations_are_rejected():
    from signed_study.retrieval_benchmark import summarize
    row = dict(workload='full', ordinal=0, repeat=0, method='dense', seconds=1., status='ok')
    with pytest.raises(ValueError): summarize([row,row], build_seconds=0)


def test_declared_missing_repeats_or_whole_users_do_not_become_complete():
    from signed_study.retrieval_benchmark import summarize
    rows=[dict(workload='full',ordinal=0,repeat=r,method=m,seconds=1.,status='ok')
          for m in ('dense','gram_fast') for r in range(5) if m!='gram_fast' or r<4]
    result=summarize(rows,build_seconds=0,expected_repeats=5,expected_ordinals=range(2))['full']
    assert result['methods']['gram_fast']['complete_users']==0
    assert result['methods']['dense']['complete_users']==1
    assert result['scheduled_ordinals']==2
    assert result['observed_ordinals']==1
    assert result['paired']['dense/gram_fast']['missing_users']==2


def test_benchmark_retains_observations_but_rejects_changed_source(monkeypatch):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('benchmark_retrieval',root/'scripts/benchmark_retrieval.py')
    benchmark=importlib.util.module_from_spec(spec);spec.loader.exec_module(benchmark)
    identities=iter([{'commit':'before'},{'commit':'after'}])
    monkeypatch.setattr(benchmark,'source_identity',lambda:next(identities))
    args=SimpleNamespace(local_bundle=None,seed=1,items=256,users=1,config=root/'config/study.json',
                         build_block=64,k=10,chunk=32,warmups=1,repeats=5)
    result=benchmark.run(args)
    assert result['source']=={'commit':'before'}
    assert result['source_unchanged'] is False
    assert any(e['stage']=='source_changed' for e in result['exceptions'])
    assert len(result['observations'])==60
    assert len(result['components'])==60
    assert all(row['status']=='ok' and all(row[key]>=0 for key in
        ('validation_gather_seconds','score_kernel_seconds','ranking_seconds'))
        for row in result['components'])


def test_whole_missing_workload_and_methods_retain_the_declared_denominator():
    from signed_study.retrieval_benchmark import summarize
    rows=[dict(workload='full',ordinal=0,repeat=0,method='dense',seconds=1.,status='ok')]
    result=summarize(rows,build_seconds=0,expected_repeats=1,expected_ordinals=range(2),
        expected_methods=['dense','gram_fast'],expected_workloads=['full','excluded'])
    missing=result['excluded']
    assert missing['scheduled_ordinals']==2 and missing['observed_ordinals']==0
    assert missing['methods']['gram_fast']['complete_users']==0
    assert missing['paired']['dense/gram_fast']['missing_users']==2
    assert result['full']['methods']['gram_fast']['observations']==0
