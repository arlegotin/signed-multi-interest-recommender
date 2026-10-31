"""Prepared forecasts must retain the old arithmetic, including boundary bits."""
import copy
import math

import pytest

from signed_study.calibration import _fit_cost, _score_units
from signed_study.data import Episode, History
from signed_study.models import model_configuration


def old_projection(episodes, entries, measurements, config, exact=False):
    total = 0.
    for entry in entries:
        rows = [r for r in measurements if r['family'] == entry.family and
                (not exact or entry.family not in ('LinearLogistic', 'RBFSVM') or
                 r.get('parameters') == entry.parameters)]
        priced = []
        for row in rows:
            chunk = config.numerics['score_chunk_size']
            timing = row if row.get('chunk_size') == chunk else row['score_measurements'][str(chunk)]
            priced.append({**row, **{k: timing.get(k, 0.) for k in
                                    ('score_seconds', 'write_seconds', 'overhead_seconds')}})
        for e in episodes:
            h = e.history
            units = _score_units(entry.family, len(h.positive_ids), len(h.negative_ids),
                                 len(e.candidate_ids), entry.parameters.get('K', 4))
            rate = max(r['score_seconds'] / _score_units(entry.family, r['positive_count'],
                       r['negative_count'], r['candidate_count'], r.get('K', 4)) for r in priced)
            total += (_fit_cost(priced, len(h.positive_ids) + len(h.negative_ids)) + units * rate
                      + max(r['write_seconds'] + r['overhead_seconds'] for r in priced))
    return total


def inputs():
    families = ('SignedMax', 'MeanNegative', 'AllExemplarMargin', 'PrototypeRatio',
                'ExemplarRatio', 'LinearLogistic', 'RBFSVM')
    rows = [dict(family=f, parameters={'C': c}, support_count=n, positive_count=n-3,
                 negative_count=3, candidate_count=11, K=2, chunk_size=512,
                 fit_seconds=t, score_seconds=.00013 * c, write_seconds=.00007,
                 overhead_seconds=.00003) for f in families for c in (1, 2)
            for n, t in ((8, .003), (16, .001), (32, .006), (32, .004))]
    episodes = [Episode(History(i, 1, 'development', tuple(range(n-3)), (90, 91, 92)),
                        tuple(range(7+i))) for i, n in enumerate((5, 8, 13, 32, 61))]
    entries = [model_configuration(f, {'C': c} if f in ('LinearLogistic', 'RBFSVM')
                                   else {'K': 2, 'lambda': c}) for f in families for c in (1, 2)]
    return rows, episodes, entries


@pytest.mark.parametrize('exact', [False, True])
def test_prepared_projection_is_bit_identical_to_original(local_config, exact):
    from signed_study.costs import PreparedCosts
    from signed_study.development import projected_cost
    rows, episodes, entries = inputs()
    expected = old_projection(episodes, entries, rows, local_config, exact)
    prepared = PreparedCosts(rows, chunk_size=512)
    assert projected_cost(episodes, entries, rows, local_config,
                          exact_parameters=exact, prepared=prepared) == expected
    # Owning a local snapshot prevents a changed input silently modifying a campaign forecast.
    rows[0]['fit_seconds'] = 1000.
    assert prepared.total(episodes, entries, exact_parameters=exact) == expected


def test_family_envelopes_are_built_once_per_filter(local_config, monkeypatch):
    from signed_study import costs
    rows, episodes, entries = inputs()
    scans = []
    real = costs.FamilyCost.prepare
    def prepare(rows, family):
        scans.append(len(rows))
        return real(rows, family)
    monkeypatch.setattr(costs.FamilyCost, 'prepare', prepare)
    prepared = costs.PreparedCosts(rows, chunk_size=512)
    prepared.total(episodes * 10, entries)
    prepared.total(episodes * 20, entries)
    assert len(scans) == 7
    assert sum(scans) == len(rows)


@pytest.mark.parametrize('field,value', [('support_count', 0), ('candidate_count', 0),
    ('positive_count', -1), ('negative_count', 0), ('K', 0),
    ('score_seconds', math.inf), ('fit_seconds', -1)])
def test_prepared_costs_reject_invalid_measurements(field, value):
    from signed_study.costs import PreparedCosts
    rows, episodes, entries = inputs()
    rows[0][field] = value
    with pytest.raises(ValueError):
        PreparedCosts(rows, chunk_size=512).total(episodes, entries)


def test_prepared_cost_requires_exact_solver_and_accepted_chunk():
    from signed_study.costs import PreparedCosts
    rows, episodes, _ = inputs()
    with pytest.raises(ValueError, match='measured'):
        PreparedCosts(rows, chunk_size=7).total(episodes, [model_configuration('SignedMax', {'K': 2})])
    with pytest.raises(ValueError, match='family'):
        PreparedCosts(rows, chunk_size=512).total(episodes,
            [model_configuration('LinearLogistic', {'C': 99})], exact_parameters=True)


@pytest.mark.parametrize('damage', ['missing_score','missing_write','empty_accepted_chunk','wrong_support'])
def test_missing_timing_evidence_never_becomes_zero_cost(damage):
    from signed_study.costs import PreparedCosts
    rows, episodes, entries=inputs()
    chunk=512
    if damage=='missing_score':del rows[0]['score_seconds']
    elif damage=='missing_write':del rows[0]['write_seconds']
    elif damage=='wrong_support':rows[0]['support_count']+=1
    else:
        rows[0]['score_measurements']={'7':{'numerical_accepted':True}}
        chunk=7
    with pytest.raises(ValueError):
        PreparedCosts(rows,chunk_size=chunk).total(episodes,entries)


def test_whole_forecast_matches_original_outputs_at_float_budget_boundaries(local_config):
    """Golden digests came from the untouched inspected revision, not new helpers."""
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from test_calibration import measured, cohort
    from signed_study.calibration import forecast_scale
    from signed_study.protocol import canonical_sha256
    fixture=json.loads((Path(__file__).parent/'fixtures/forecast-boundaries.json').read_text())
    assert canonical_sha256(measured())==fixture['measurement_sha256']
    for case in fixture['cases']:
        ledger=SimpleNamespace(config=local_config,snapshot=lambda:case['state'])
        actual=forecast_scale(measured(),cohort(),ledger)
        assert canonical_sha256(actual)==case['forecast_sha256'],case['name']
