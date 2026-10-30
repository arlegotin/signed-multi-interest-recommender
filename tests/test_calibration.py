"""Outcome-independent sampling and measured, conservative admission."""
import hashlib
from types import SimpleNamespace

import pytest

from signed_study.data import Episode, EpisodeStore, History


def cohort(development=418, confirmation=1776, dataset='ml-1m'):
    def episodes(n, role, offset):
        return tuple(Episode(History(offset+i, 100, role, tuple(range(10+i % 40)),
                                     tuple(range(100, 105+i % 10))), tuple(range(200, 210+i % 30)))
                     for i in range(n))
    return EpisodeStore(episodes(development, 'development', 1),
                        episodes(confirmation, 'confirmation', 10000), {'dataset': dataset}, {})


def test_calibration_selection_uses_rank_quartiles_and_long_candidates():
    from signed_study import run
    assert hasattr(run, 'select_calibration_users'), 'ordered calibration sampler is missing'
    episodes = list(cohort().development)
    selected = run.select_calibration_users(episodes)
    digest = lambda uid: hashlib.sha256(f'signed-study:{uid}'.encode()).hexdigest()
    ranked = sorted(episodes, key=lambda e: (len(e.history.positive_ids)+len(e.history.negative_ids), digest(e.history.user_id)))
    expected = []
    for q in range(4):
        part = ranked[q*len(ranked)//4:(q+1)*len(ranked)//4]
        small = sorted(part, key=lambda e: digest(e.history.user_id))[:4]
        long = sorted([e for e in part if e not in small],
                      key=lambda e: (-len(e.candidate_ids), digest(e.history.user_id)))[:4]
        expected.extend(e.history.user_id for e in small+long)
    assert selected == sorted(expected, key=digest)
    assert run.select_calibration_users(list(reversed(episodes))) == selected
    with pytest.raises(ValueError, match='32'):
        run.select_calibration_users(episodes[:31])
    with pytest.raises(ValueError, match='development'):
        run.select_calibration_users(list(cohort().confirmation))


def measured():
    # All seven families measured at endpoints, two observed support sizes.
    return {'user_measurements': [dict(family=family, support_count=support,
             positive_count=support-5, negative_count=5, candidate_count=20,
             K=4, chunk_size=512, fit_seconds=.0001, score_seconds=.0001, write_seconds=.0001,
             score_measurements={str(chunk):dict(score_seconds=.0001,write_seconds=.0001,
                                overhead_seconds=0.,numerical_accepted=True) for chunk in (1,7,512)})
             for family in ('MeanNegative', 'SignedMax', 'AllExemplarMargin',
                            'PrototypeRatio', 'ExemplarRatio', 'LinearLogistic', 'RBFSVM')
             for support in (15, 64)], 'fixed_seconds': 1.,
            'optional_seconds': 1., 'memory_bound': False, 'chunk_invariance': True}


def test_forecast_exact_scale_order_and_fallback(local_config):
    from signed_study import run
    assert hasattr(run, 'forecast_scale'), 'measured forecast is missing'
    state = {'experimental_seconds': 0., 'attempted_evaluations': 0,
             'stage_seconds': {key: 0. for key in local_config.resources['stage_seconds']}}
    ledger = SimpleNamespace(config=local_config, snapshot=lambda: state)
    result = run.forecast_scale(measured(), cohort(), ledger)
    assert result['scale'] == [418, 1776]
    assert result['safety_multiplier'] == 1.5
    assert len(result['planned_user_ids']['confirmation']) == 1776
    # Each intended configuration still consumes its evaluation reservation.
    state['attempted_evaluations'] = 99999
    result = run.forecast_scale(measured(), cohort(), ledger)
    assert result['status'] == 'infeasible'
    assert result['tested_scales'] == [[418,1776], [256,1776], [418,1500],
                                     [256,1500], [418,1000], [256,1000]]
    assert result['adjustments'][:3] == ['remove_optional_work',
                  'embedding_batch_unchanged_not_memory_bound', 'reduce_scoring_chunk']
    state['attempted_evaluations'] = 53000
    assert run.forecast_scale(measured(), cohort(), ledger)['scale'] == [256,1500]
    result = run.forecast_scale(measured(), cohort(62,225,'ml-100k'), ledger)
    assert result['tested_scales'] == [[62,225]]
    state['attempted_evaluations'] = 0
    assert run.forecast_scale(measured(), cohort(62,225,'ml-100k'), ledger)['scale'] == [62,225]


def test_forecast_preserves_report_reserve_and_rejects_unmeasured_families(local_config):
    from signed_study import run
    assert hasattr(run, 'forecast_scale')
    ledger = SimpleNamespace(config=local_config, snapshot=lambda: {
        'experimental_seconds': 12599., 'attempted_evaluations': 0,
        'stage_seconds': {key: 0. for key in local_config.resources['stage_seconds']}})
    assert run.forecast_scale(measured(), cohort(), ledger)['status'] == 'infeasible'
    m = measured()
    m['user_measurements'] = m['user_measurements'][:-2]
    with pytest.raises(ValueError, match='family'):
        run.forecast_scale(m, cohort(), ledger)


def test_catalog_timing_forecast_uses_full_catalog_not_evaluation_candidates(local_config):
    from signed_study.run import forecast_scale
    ledger = SimpleNamespace(config=local_config, snapshot=lambda: {
        'experimental_seconds':0.,'attempted_evaluations':0,
        'stage_seconds':{s:0. for s in local_config.resources['stage_seconds']}})
    small=measured()
    small['catalog_items']=256
    large={**small,'catalog_items':3883}
    assert forecast_scale(large,cohort(),ledger)['seconds']['G5'] > 5*forecast_scale(small,cohort(),ledger)['seconds']['G5']


def test_forecast_covers_ten_selected_families_and_three_matched_controls(local_config):
    from signed_study.run import forecast_scale
    ledger=SimpleNamespace(config=local_config,snapshot=lambda:{
        'experimental_seconds':0.,'attempted_evaluations':0,
        'stage_seconds':{s:0. for s in local_config.resources['stage_seconds']}})
    measurements=measured()
    measurements['optional_seconds']=0.
    result=forecast_scale(measurements,cohort(),ledger)
    assert result['reserved_evaluations']==65368
    planning=result['planning']
    assert planning['confirmation_entries']==13
    assert planning['timing_evaluations']==2496
    assert planning['numerical_diagnostic_evaluations']==704
    assert {row['name'] for row in planning['comparison_schedule']}=={
        'Centroid','PositiveMax','RocchioDirection','MeanNegative','SignedMax',
        'AllExemplarMargin','PrototypeRatio','ExemplarRatio','LinearLogistic','RBFSVM',
        'matched_MeanNegative','matched_PositiveMax','matched_K1_SignedMax'}


def test_fallback_refuses_budget_that_covers_twelve_but_not_thirteen_entries(local_config):
    from signed_study.run import forecast_scale
    ledger=SimpleNamespace(config=local_config,snapshot=lambda:{
        'experimental_seconds':0.,'attempted_evaluations':88000,
        'stage_seconds':{s:0. for s in local_config.resources['stage_seconds']}})
    measurements=measured()
    measurements['optional_seconds']=0.
    result=forecast_scale(measurements,cohort(62,225,'ml-100k'),ledger)
    assert result['status']=='infeasible'


@pytest.mark.parametrize('alternate',['slow','missing','invalid'])
def test_reduced_chunk_requires_its_own_measured_costs(local_config,alternate):
    from signed_study.run import forecast_scale
    ledger=SimpleNamespace(config=local_config,snapshot=lambda:{
        'experimental_seconds':0.,'attempted_evaluations':53000,
        'stage_seconds':{s:0. for s in local_config.resources['stage_seconds']}})
    measurements=measured()
    fast=forecast_scale(measurements,cohort(),ledger)
    assert fast['scale']==[256,1500]
    for row in measurements['user_measurements']:
        if alternate=='slow': row['score_measurements']['7']['score_seconds']=100.
        elif alternate=='missing': del row['score_measurements']['7']
        else: row['score_measurements']['7']['numerical_accepted']=False
    result=forecast_scale(measurements,cohort(),ledger)
    assert result['status']=='infeasible'
    assert result['policy']['score_chunk_size']==7
    if alternate in ('missing','invalid'):
        assert 'remeasurement' in result['stop_reason']
    else:
        assert result['tested_scales']==[[418,1776],[256,1776],[418,1500],[256,1500],[418,1000],[256,1000]]
