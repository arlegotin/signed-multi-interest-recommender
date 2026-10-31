"""Local, input-owned empirical cost envelopes; no cross-campaign cache."""
from bisect import bisect_left
from copy import deepcopy
from dataclasses import dataclass
import json
import math


@dataclass(frozen=True)
class FamilyCost:
    """An observed family/filter/chunk envelope with the original arithmetic."""

    sizes: tuple
    upper: tuple
    maximum: float
    slope: float
    score_rate: float
    publication: float

    @staticmethod
    def prepare(rows, family):
        from .calibration import _score_units
        if not rows:
            raise ValueError('No measured cost for required family')
        bins = {}
        rates = []
        for row in rows:
            if not {'fit_seconds','score_seconds','write_seconds'} <= row.keys():
                raise ValueError('Missing measured cost duration evidence')
            for key in ('support_count', 'positive_count', 'negative_count', 'candidate_count', 'K'):
                value = row.get(key, 4 if key == 'K' else None)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError('Invalid measured cost sizes')
            if row['support_count'] != row['positive_count'] + row['negative_count']:
                raise ValueError('Measured support size differs from class counts')
            for key in ('fit_seconds', 'score_seconds', 'write_seconds', 'overhead_seconds'):
                if not math.isfinite(row.get(key, 0.)) or row.get(key, 0.) < 0:
                    raise ValueError('Invalid measured development cost')
            n = row['support_count']
            bins[n] = max(bins.get(n, 0.), row['fit_seconds'])
            units = _score_units(family, row['positive_count'], row['negative_count'],
                                 row['candidate_count'], row.get('K', 4))
            rates.append(row['score_seconds'] / units)
        sizes = tuple(sorted(bins))
        # Suffix maxima give exactly the same max of all observed larger bins.
        upper = [0.] * len(sizes)
        for i in range(len(sizes)-1, -1, -1):
            upper[i] = max(bins[sizes[i]], upper[i+1] if i+1 < len(sizes) else 0.)
        slopes = [max(0., (bins[b]-bins[a])/(b-a)) for a, b in zip(sizes, sizes[1:])]
        return FamilyCost(sizes, tuple(upper), max(bins.values()),
            max(slopes, default=bins[sizes[-1]] / sizes[-1]), max(rates),
            max(r['write_seconds'] + r.get('overhead_seconds', 0.) for r in rows))

    def fit(self, support):
        index = bisect_left(self.sizes, support)
        if index < len(self.sizes):
            return self.upper[index]
        return self.maximum + self.slope * (support-self.sizes[-1])

    def cost(self, episode, family, K=4, candidate_count=None):
        from .calibration import _score_units
        history = episode.history
        units = _score_units(family, len(history.positive_ids), len(history.negative_ids),
            len(episode.candidate_ids) if candidate_count is None else candidate_count, K)
        return (self.fit(len(history.positive_ids) + len(history.negative_ids))
                + units * self.score_rate + self.publication)


class PreparedCosts:
    """Snapshot timing input once; prepare each exact family/filter only once.

    Construct a new instance after endpoint measurements or chunk policy changes.
    Query totals retain entry/episode summation order to preserve decisions.
    """

    def __init__(self, measurements, *, chunk_size):
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError('Invalid cost chunk size')
        self._rows = deepcopy(list(measurements))
        self._chunk = chunk_size
        self._families = {}
        self._prepared = {}
        for row in self._rows:
            self._families.setdefault(row['family'], []).append(row)

    def envelope(self, entry, *, exact_parameters=False):
        filtered = exact_parameters and entry.family in ('LinearLogistic', 'RBFSVM')
        key = (entry.family, json.dumps(entry.parameters, sort_keys=True) if filtered else None)
        if key not in self._prepared:
            rows = self._families.get(entry.family, [])
            if filtered:
                rows = [r for r in rows if r.get('parameters') == entry.parameters]
            priced = []
            for row in rows:
                timing = (row if row.get('chunk_size') == self._chunk else
                          row.get('score_measurements', {}).get(str(self._chunk)))
                if (timing is None or not {'score_seconds','write_seconds'} <= timing.keys()
                        or timing is not row and timing.get('numerical_accepted') is not True):
                    raise ValueError('Missing measured canonical scoring chunk cost')
                priced.append({**row, **{k: timing.get(k, 0.) for k in
                    ('score_seconds', 'write_seconds', 'overhead_seconds')}})
            self._prepared[key] = FamilyCost.prepare(priced, entry.family)
        return self._prepared[key]

    def total(self, episodes, entries, *, exact_parameters=False):
        total = 0.
        for entry in entries:
            envelope = self.envelope(entry, exact_parameters=exact_parameters)
            for episode in episodes:
                total += envelope.cost(episode, entry.family, entry.parameters.get('K', 4))
        return total
