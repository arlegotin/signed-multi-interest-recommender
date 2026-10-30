"""Outcome-independent calibration sampling and empirical resource forecasts."""
import hashlib
import math

from .data import Episode, EpisodeStore
from .models import initial_configurations


def user_digest(user_id):
    return hashlib.sha256(f'signed-study:{user_id}'.encode()).hexdigest()


def select_calibration_users(episodes: list[Episode]) -> list[int]:
    if len(episodes) < 32:
        raise ValueError('Calibration requires at least 32 development users')
    if any(e.history.role != 'development' for e in episodes):
        raise ValueError('Calibration accepts development episodes only')
    if len({e.history.user_id for e in episodes}) != len(episodes):
        raise ValueError('Duplicate development user')
    ranked = sorted(episodes, key=lambda e: (
        len(e.history.positive_ids) + len(e.history.negative_ids), user_digest(e.history.user_id)))
    chosen = []
    for q in range(4):
        part = ranked[q*len(ranked)//4:(q+1)*len(ranked)//4]
        small = sorted(part, key=lambda e: user_digest(e.history.user_id))[:4]
        remaining = [e for e in part if e not in small]
        chosen.extend(small + sorted(remaining, key=lambda e: (
            -len(e.candidate_ids), user_digest(e.history.user_id)))[:4])
    return sorted((e.history.user_id for e in chosen), key=user_digest)


def _fit_cost(rows, support):
    # Upper envelope of observed size bins; extrapolate only the measured slope,
    # never a claimed SVM quadratic/cubic complexity law.
    bins = {}
    for r in rows:
        n = r['support_count']
        bins[n] = max(bins.get(n, 0.), r['fit_seconds'])
    sizes = sorted(bins)
    upper = [bins[n] for n in sizes if n >= support]
    if upper:
        return max(upper)
    slopes = [max(0., (bins[b]-bins[a])/(b-a)) for a, b in zip(sizes, sizes[1:])]
    slope = max(slopes, default=bins[sizes[-1]] / sizes[-1])
    return max(bins.values()) + slope*(support-sizes[-1])


def _score_units(family, positive, negative, candidates, K):
    if family == 'LinearLogistic':
        references = 1
    elif family in ('AllExemplarMargin', 'ExemplarRatio', 'RBFSVM'):
        references = positive + negative
    else:
        references = min(K, positive) + (1 if family == 'MeanNegative' else negative)
    return 384 * candidates * references


def forecast_scale(measurements: dict, cohort: EpisodeStore, ledger) -> dict:
    """Forecast remaining G3/G4/G5 from actual capped support/candidate sizes.

    Reserve 92 development entries, 12 selected/matched confirmation entries,
    eight all-grid smoke users and 32x12x(1+5) timing evaluations. These are
    conservative planning reservations, not claims of executed fits.
    """
    config, state = ledger.config, ledger.snapshot()
    registry = initial_configurations(config)
    families = {entry.family for entry in registry}
    rows = measurements['user_measurements']
    by_family = {family: [r for r in rows if r['family'] == family] for family in families}
    if any(not r for r in by_family.values()):
        raise ValueError('Every comparator family needs measured calibration work')
    for r in rows:
        if any(not math.isfinite(r[key]) or r[key] < 0 for key in
               ('fit_seconds', 'score_seconds', 'write_seconds')):
            raise ValueError('Invalid calibration duration')
        if min(r['support_count'], r['candidate_count'], r['positive_count'], r['negative_count']) < 1:
            raise ValueError('Invalid calibration sizes')

    def cost(e, entry, candidate_count=None):
        observed = by_family[entry.family]
        h = e.history
        units = _score_units(entry.family, len(h.positive_ids), len(h.negative_ids),
                             len(e.candidate_ids) if candidate_count is None else candidate_count,
                             entry.parameters.get('K', 4))
        rate = max(r['score_seconds'] / _score_units(entry.family, r['positive_count'],
                   r['negative_count'], r['candidate_count'], r.get('K', 4)) for r in observed)
        return (_fit_cost(observed, len(h.positive_ids)+len(h.negative_ids)) + units*rate
                + max(r['write_seconds'] + r.get('overhead_seconds',0.) for r in observed))

    dev = sorted(cohort.development, key=lambda e: user_digest(e.history.user_id))
    confirm = sorted(cohort.confirmation, key=lambda e: user_digest(e.history.user_id))
    dev_cost = [sum(cost(e, c) for c in registry) + 14*max(cost(e, c) for c in registry) for e in dev]
    conf_cost = [12*max(cost(e, c) for c in registry) for e in confirm]
    scales = ([list(config.calibration['fallback_scale_pair'])] if cohort.audit['dataset'] == 'ml-100k'
              else [list(p) for p in config.calibration['primary_scale_pairs']])
    if cohort.audit['dataset'] not in ('ml-1m', 'ml-100k'):
        raise ValueError('Production forecast requires official dataset provenance')
    caps = config.resources
    margin = caps['safety_multiplier']
    result = dict(status='infeasible', scale=None, safety_multiplier=margin, tested_scales=[],
                  adjustments=[], policy=dict(embedding_batch_size=config.encoder['batch_size'],
                  score_chunk_size=config.numerics['score_chunk_size'], blas_threads=config.timing['blas_threads'],
                  inference_workers=1), planning=dict(development_entries=92, confirmation_entries=12,
                  smoke_users=8, smoke_entries=78, timing_evaluations=2304,
                  numerical_diagnostic_evaluations=320),
                  support_model='upper observed size-bin envelope plus measured nonnegative slope',
                  stop_reason='Smallest permitted scale cannot fit remaining allocations and reserve')
    fixed = measurements.get('fixed_seconds', 0.)
    catalog_items=measurements.get('catalog_items',3883 if cohort.audit['dataset']=='ml-1m' else 1682)
    if type(catalog_items) is not int or catalog_items<=0:
        raise ValueError('Measured catalog size must be positive')
    timing = 2304 * max((max(cost(e,c,catalog_items) for c in registry) for e in dev[:32]),default=0.)
    numerical = 320 * max((max(cost(e,c) for c in registry) for e in confirm[:64]),default=0.)
    smoke_seconds=sum(dev_cost[:8])

    def projection(pair, optional):
        d, c = pair
        seconds = {'G3': sum(dev_cost[:d]) + smoke_seconds + fixed,
                   'G4': sum(conf_cost[:c]) + fixed, 'G5': timing + numerical + fixed + optional}
        evaluations = d*92 + c*12 + 624 + 2304 + 320 + (4200 if optional else 0)
        fits = (d <= len(dev) and c <= len(confirm)
                and state['attempted_evaluations']+evaluations <= caps['evaluation_limit']
                and all(state['stage_seconds'][s]+margin*t <= caps['stage_seconds'][s] for s,t in seconds.items())
                and state['experimental_seconds']+margin*sum(seconds.values()) <=
                    caps['experimental_seconds']-caps['report_reserve_seconds'])
        return fits, seconds, evaluations

    optional = measurements.get('optional_seconds', 0.)
    first_fits, _, _ = projection(scales[0], optional)
    if measurements.get('memory_bound') or not first_fits:
        optional = 0.
        result['adjustments'].append('remove_optional_work')
        if measurements.get('memory_bound'):
            result['policy']['embedding_batch_size'] = max(1, config.encoder['batch_size']//2)
            result['adjustments'].append('reduce_embedding_batch')
            # A smaller batch is a proposed policy, never invented evidence of
            # lower RSS. An actual remeasurement is needed before proceeding.
            if not measurements.get('reduced_batch_memory_verified', False):
                result['stop_reason'] = 'Reduced embedding batch needs measured memory verification'
        else:
            result['adjustments'].append('embedding_batch_unchanged_not_memory_bound')
        result['policy']['score_chunk_size'] = min(7, config.numerics['score_chunk_size'])
        result['adjustments'].append('reduce_scoring_chunk')
        if not measurements.get('chunk_invariance'):
            result['stop_reason'] = 'Reduced scoring chunk lacks invariance evidence'
            return result
    for pair in scales:
        result['tested_scales'].append(pair)
        fits, seconds, evaluations = projection(pair, optional)
        if measurements.get('memory_bound') and not measurements.get('reduced_batch_memory_verified'):
            fits = False
        if fits:
            d,c = pair
            result.update(status='feasible', scale=pair, seconds=seconds,
                          smoke_seconds=smoke_seconds,catalog_items=catalog_items,
                          reserved_evaluations=evaluations, optional_work=bool(optional), stop_reason=None,
                          planned_user_ids={'development': [e.history.user_id for e in dev[:d]],
                                            'confirmation': [e.history.user_id for e in confirm[:c]]},
                          interval_width_estimate={'half_width': 1.96/math.sqrt(c),
                             'basis': 'conservative SD bound 1 for paired AUC difference; development estimate follows G3'})
            break
    return result
