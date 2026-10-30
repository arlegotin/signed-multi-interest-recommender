"""Tiny, history-only baseline fixtures; no dataset/model acquisition."""

from collections import Counter
from dataclasses import replace
import json
import pickle
from pathlib import Path
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from signed_study.data import Episode, History
from signed_study import models
from signed_study.protocol import load_config


class Vectors:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    def rows(self, ids):
        return self.values[list(ids)]


@pytest.fixture
def config():
    return load_config(Path('config/study.json'))


@pytest.fixture
def support():
    # Positive centroid is (1,0), negative mean is (-.8,0).
    return History(1, 10, 'development', (0, 1), (2, 3)), Vectors([
        [.8, .6], [.8, -.6], [-.8, .6], [-.8, -.6], [1, 0], [0, 1],
    ])


def test_registry_and_baseline_information_parity(config, support):
    history, vectors = support
    entries = models.initial_configurations(config)
    families = ['MeanNegative', 'SignedMax', 'AllExemplarMargin', 'PrototypeRatio',
                'ExemplarRatio', 'LinearLogistic', 'RBFSVM']
    assert [Counter(e.family for e in entries)[f] for f in families] == [21, 21, 7, 3, 1, 5, 20]
    assert len(entries) == len({e.id for e in entries}) == 78
    assert entries == models.initial_configurations(config)
    fitted_models = [models.fit_user(e.family, history, vectors, e.parameters) for e in entries]
    expected_support_ids = history.positive_ids + history.negative_ids
    assert all(m.support_ids == expected_support_ids for m in fitted_models)
    assert all(m.diagnostics['status'] == 'ok' for m in fitted_models)
    for model in fitted_models:
        scores = models.score_user(model, (4, 5), vectors)
        assert scores.shape == (2,) and scores.dtype == np.float64
        if model.method == 'MeanNegative':
            X = vectors.rows((4, 5))
            A_K = np.max(X @ model.state['P'].T, axis=1)
            penalty = model.parameters['lambda']
            mean_negative_scores = scores
            N = vectors.rows(history.negative_ids)
            np.testing.assert_allclose(mean_negative_scores, A_K - penalty * X @ N.mean(axis=0))
    # Extracted views consume existing grid entries, with identical scores.
    for alias, params, family in [('Centroid', {}, 'SignedMax'),
                                 ('PositiveMax', {'K': 2}, 'SignedMax'),
                                 ('RocchioDirection', {'lambda': 2}, 'MeanNegative')]:
        alias_model = models.fit_user(alias, history, vectors, params)
        source = models.fit_user(family, history, vectors,
                                 {'K': params.get('K', 1), 'lambda': params.get('lambda', 0)})
        np.testing.assert_array_equal(models.score_user(alias_model, (4, 5), vectors),
                                      models.score_user(source, (4, 5), vectors))


def test_logistic_real_fit_orders_separable_classes_and_uses_balancing(support):
    history, vectors = support
    history = replace(history, positive_ids=(0, 1, 4))
    model = models.fit_user('LinearLogistic', history, vectors, {'C': 1})
    assert model.diagnostics['status'] == 'ok'
    assert model.diagnostics['class_weights'] == {0: 1.25, 1: 5 / 6}
    scores = models.score_user(model, (0, 1, 4, 2, 3), vectors)
    assert min(scores[:3]) > max(scores[3:])
    assert model.state['coef'].flags.c_contiguous
    assert model.state['coef'].dtype == np.float64


def test_rbf_real_fit_recovers_nonlinear_order_and_history_only_scale():
    vectors = Vectors([[1, 0], [-1, 0], [0, 1], [0, -1], [.6, .8]])
    history = History(1, 10, 'development', (0, 1), (2, 3))
    model = models.fit_user('RBFSVM', history, vectors, {'C': 100, 'g': 16})
    assert model.diagnostics['status'] == 'ok'
    assert model.state['m2'] == 2
    assert model.state['gamma'] == 8
    scores = models.score_user(model, (0, 1, 2, 3), vectors, chunk_size=1)
    assert min(scores[:2]) > max(scores[2:])
    np.testing.assert_allclose(scores, models.score_user(model, (0, 1, 2, 3), vectors, chunk_size=7), atol=1e-12)
    # Zero off-diagonal distances are excluded, rather than diluting the median.
    duplicate_history = replace(history, positive_ids=(0, 0, 0, 1))
    with pytest.raises(ValueError, match='distinct'):
        models.fit_user('RBFSVM', duplicate_history, vectors, {'C': 1, 'g': 1})
    duplicates = Vectors([[1, 0], [1, 0], [1, 0], [-1, 0]])
    model = models.fit_user('RBFSVM', history, duplicates, {'C': 1, 'g': 1})
    assert model.state['m2'] == 4
    identical = Vectors([[1, 0]] * 4)
    model = models.fit_user('RBFSVM', history, identical, {'C': 1, 'g': 1})
    assert model.state['m2'] == 1


def test_candidate_label_mutation_cannot_change_fit_or_scores(config, support):
    history, vectors = support
    episode = Episode(history, (4, 5))
    candidate_labels = {4: 1, 5: 0}
    for entry in models.initial_configurations(config):
        before = models.fit_user(entry.family, episode.history, vectors, entry.parameters)
        state = pickle.dumps(before.state)
        scores = models.score_user(before, episode.candidate_ids, vectors).tobytes()
        candidate_labels[4], candidate_labels[5] = candidate_labels[5], candidate_labels[4]
        after = models.fit_user(entry.family, episode.history, vectors, entry.parameters)
        assert pickle.dumps(after.state) == state
        assert models.score_user(after, episode.candidate_ids, vectors).tobytes() == scores


@pytest.mark.parametrize('family,failure', [
    (family, failure) for family in ('LinearLogistic', 'RBFSVM')
    for failure in ('warning', 'iterations', 'nonfinite', 'exception', 'status')
    if family == 'RBFSVM' or failure != 'status'
])
def test_solver_failure_diagnostics_are_retained(monkeypatch, support, family, failure):
    history, vectors = support
    cls = models.LogisticRegression if family == 'LinearLogistic' else models.SVC
    real_fit = cls.fit

    def damaged_fit(self, X, y, *args, **kwargs):
        assert X.dtype == np.float64 and X.flags.c_contiguous
        if failure == 'exception':
            raise RuntimeError('injected solver exception')
        result = real_fit(self, X, y, *args, **kwargs)
        if failure == 'warning':
            warnings.warn('injected convergence warning', ConvergenceWarning)
        elif failure == 'iterations':
            self.n_iter_[:] = self.max_iter
        elif failure == 'nonfinite':
            self.intercept_[:] = np.nan
        elif failure == 'status':
            self.fit_status_ = 1
        return result

    monkeypatch.setattr(cls, 'fit', damaged_fit)
    params = {'C': 1} if family == 'LinearLogistic' else {'C': 1, 'g': 1}
    model = models.fit_user(family, history, vectors, params)
    assert model.diagnostics['status'] == 'failed'
    assert failure in model.diagnostics['failures']
    if failure == 'warning':
        assert model.diagnostics['warnings'][0]['category'] == 'ConvergenceWarning'
    with pytest.raises(ValueError, match='failed'):
        models.score_user(model, (4, 5), vectors)


def test_nonfinite_scoring_marks_model_failed(support):
    history, vectors = support
    model = models.fit_user('LinearLogistic', history, vectors, {'C': 1})
    model.state['coef'][:] = np.inf
    with pytest.raises(ValueError, match='nonfinite'):
        models.score_user(model, (4, 5), vectors)
    assert model.diagnostics['status'] == 'failed'
    assert 'nonfinite' in model.diagnostics['failures']


def test_resolved_global_iteration_repairs_are_explicit(config, support, tmp_path, monkeypatch):
    history, vectors = support
    settings = json.loads(Path('config/study.json').read_text())
    settings['prototype']['max_updates'] = 100
    settings['linear_logistic']['max_iter'] = 4000
    settings['rbf_svm']['max_iter'] = 200000
    settings['pre_freeze_repairs'] = [
        {'kind': kind, 'stage': 'G1', 'reason': 'synthetic repair plumbing test'}
        for kind in ('prototype_iterations', 'solver_iterations')
    ]
    config_path = tmp_path / 'repair-config.json'
    config_path.write_text(json.dumps(settings))
    repaired = load_config(config_path)
    real_prototypes = models.spherical_prototypes

    def checked_prototypes(*args, **kwargs):
        assert kwargs['max_updates'] == 100
        return real_prototypes(*args, **kwargs)

    monkeypatch.setattr(models, 'spherical_prototypes', checked_prototypes)
    for family, parameters, limit in [('SignedMax', {'K': 2, 'lambda': 1}, 100),
                                      ('LinearLogistic', {'C': 1}, 4000),
                                      ('RBFSVM', {'C': 1, 'g': 1}, 200000)]:
        model = models.fit_user(family, history, vectors, parameters, config=repaired)
        assert model.diagnostics['iteration_limit'] == limit
        assert model.diagnostics['status'] == 'ok'
        assert model.diagnostics['policy']['source'] == 'resolved_config'
        assert len(model.diagnostics['policy']['pre_freeze_repairs']) == 2
        if family != 'SignedMax':
            assert model.diagnostics['solver_settings']['max_iter'] == limit


@pytest.mark.parametrize('family', ['LinearLogistic', 'RBFSVM'])
def test_saved_decisions_match_independent_sklearn_fit(support, family):
    history, vectors = support
    history = replace(history, positive_ids=(0, 1, 4))
    X = np.ascontiguousarray(vectors.rows((0, 1, 4, 2, 3)))
    y = np.array([1, 1, 1, 0, 0])
    candidates = (3, 5, 4, 0, 1)
    if family == 'LinearLogistic':
        params = {'C': .1}
        reference = LogisticRegression(C=.1, l1_ratio=0, solver='liblinear', fit_intercept=True,
                                       class_weight='balanced', tol=1e-8, max_iter=2000, random_state=0)
    else:
        params = {'C': .1, 'g': 4}
        # Independently form all pair differences; exclude zero off-diagonal distances.
        squares = [sum((a - b) ** 2) for i, a in enumerate(X) for b in X[i + 1:]]
        gamma = 4 / np.median([d for d in squares if d > 0])
        reference = SVC(C=.1, kernel='rbf', class_weight='balanced', tol=1e-6,
                        max_iter=100000, cache_size=64, gamma=gamma)
    reference.fit(X, y)
    model = models.fit_user(family, history, vectors, params)
    np.testing.assert_allclose(models.score_user(model, candidates, vectors, chunk_size=1),
                               reference.decision_function(vectors.rows(candidates)), rtol=1e-12, atol=1e-12)
    assert not any(hasattr(v, 'fit') for v in model.state.values())


def test_empty_negatives_and_zero_rocchio_direction(support):
    history, vectors = support
    empty = replace(history, negative_ids=())
    for family in ('SignedMax', 'MeanNegative', 'AllExemplarMargin'):
        parameters = {'lambda': 8}
        if family != 'AllExemplarMargin':
            parameters['K'] = 1
        model = models.fit_user(family, empty, vectors, parameters)
        np.testing.assert_allclose(models.score_user(model, (4, 5), vectors),
                                   [1, 0] if family != 'AllExemplarMargin' else [.8, .6])
    for family, parameters in [('LinearLogistic', {'C': 1}), ('RBFSVM', {'C': 1, 'g': 1}),
                               ('PrototypeRatio', {'K': 1}), ('ExemplarRatio', {})]:
        with pytest.raises(ValueError, match='negative'):
            models.fit_user(family, empty, vectors, parameters)
    zero = models.fit_user('RocchioDirection', History(1, 1, 'development', (0,), (1,)),
                           Vectors([[1, 0], [1, 0], [0, 1], [-1, 0]]), {'lambda': 1})
    np.testing.assert_array_equal(models.score_user(zero, (0, 1, 2, 3),
                                  Vectors([[1, 0], [1, 0], [0, 1], [-1, 0]])), np.zeros(4))


def test_prototype_defect_returns_failed_fit_evidence(monkeypatch, support):
    history, vectors = support

    def defective_constructor(*args, **kwargs):
        raise ArithmeticError('injected objective decrease defect')

    monkeypatch.setattr(models, 'spherical_prototypes', defective_constructor)
    model = models.fit_user('SignedMax', history, vectors, {'K': 2, 'lambda': 1})
    assert model.diagnostics['status'] == 'failed'
    assert model.diagnostics['exception']['category'] == 'ArithmeticError'
    assert model.state == {}


def test_finite_prototype_cap_hit_is_reported_without_failing(monkeypatch, support):
    history, vectors = support
    constructor = models.spherical_prototypes

    def cap_hit(*args, **kwargs):
        result = constructor(*args, **kwargs)
        return replace(result, flags={**result.flags, 'cap_hit': True})

    monkeypatch.setattr(models, 'spherical_prototypes', cap_hit)
    model = models.fit_user('SignedMax', history, vectors, {'K': 2, 'lambda': 1})
    assert model.diagnostics['status'] == 'ok'
    assert model.diagnostics['prototype']['cap_hit'] is True
    assert np.isfinite(models.score_user(model, (4, 5), vectors)).all()


def test_geometric_adapter_oracles_and_chunking(support):
    history, vectors = support
    expected = {'SignedMax': [2.6, -1.2], 'MeanNegative': [2.6, 0],
                'AllExemplarMargin': [2.4, -.6], 'PrototypeRatio': [1, .5]}
    for family, oracle in expected.items():
        parameters = {'K': 1, 'lambda': 2} if family != 'AllExemplarMargin' else {'lambda': 2}
        if family == 'PrototypeRatio':
            parameters = {'K': 1}
        model = models.fit_user(family, history, vectors, parameters)
        scores = models.score_user(model, (4, 5), vectors, chunk_size=1)
        if family == 'PrototypeRatio':
            # For x=(0,1): dP=sqrt(2), dN=sqrt(.8), ratio=dN/(dP+dN).
            oracle[1] = np.sqrt(.8) / (np.sqrt(2) + np.sqrt(.8))
        np.testing.assert_allclose(scores, oracle, atol=1e-14)
        np.testing.assert_allclose(scores, models.score_user(model, (4, 5), vectors))
        assert models.score_user(model, (), vectors).shape == (0,)
    with pytest.raises(ValueError, match='positive'):
        models.fit_user('SignedMax', replace(history, positive_ids=()), vectors, {'K': 1, 'lambda': 1})
