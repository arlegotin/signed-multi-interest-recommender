"""Deterministic geometry, equally informed baseline fits, and declared grids.

All computation is CPU float64. These functions consume support vectors only;
neither prototype construction nor scoring accepts evaluation labels.
"""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .data import History
from .protocol import StudyConfig, canonical_sha256


@dataclass(frozen=True)
class PrototypeResult:
    vectors: np.ndarray
    objective: float
    iterations: int
    flags: dict


def _unit_rows(value, name):
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] == 0:
        raise ValueError(f'{name} must be a matrix of unit rows')
    if not np.isfinite(rows).all():
        raise ValueError(f'{name} must contain finite unit rows')
    if np.any(np.abs(np.linalg.norm(rows, axis=1) - 1) > 1e-12):
        raise ValueError(f'{name} row norms must differ from one by at most 1e-12')
    return rows


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def spherical_prototypes(X: np.ndarray, item_ids: np.ndarray, K: int, *, max_updates=50) -> PrototypeResult:
    """Bounded spherical Lloyd updates, initialized from exact distinct rows.

    IDs fix observation/tie order even if the input is shuffled. Flags preserve
    any empty/zero-sum event per prototype across all updates. The objective
    trace includes initialization, then one entry per completed update.
    """
    X = _unit_rows(X, 'X')
    if not len(X):
        raise ValueError('ineligible history: positive support is empty')
    _positive_integer(K, 'K')
    _positive_integer(max_updates, 'max_updates')
    item_ids = np.asarray(item_ids)
    if (item_ids.ndim != 1 or len(item_ids) != len(X)
            or not np.issubdtype(item_ids.dtype, np.integer)
            or len(np.unique(item_ids)) != len(item_ids)):
        raise ValueError('item_ids must be distinct integers, one per observation')
    X = X[np.argsort(item_ids, kind='stable')]
    # np.unique's lexical ordering must not override the item-ID tie rule.
    _, first_indices = np.unique(X, axis=0, return_index=True)
    distinct = X[np.sort(first_indices)]
    effective = min(int(K), len(distinct))
    mean = X.mean(axis=0)  # Duplicate observations retain their full weight.
    mean_fallback = bool(np.linalg.norm(mean) <= 1e-12)
    first = 0 if mean_fallback else int(np.argmax(distinct @ mean))
    selected = [first]
    while len(selected) < effective:
        closeness = np.max(distinct @ distinct[selected].T, axis=1)
        closeness[selected] = np.inf
        selected.append(int(np.argmin(closeness)))
    centers = distinct[selected].copy()
    similarities = X @ centers.T
    assignments = np.argmax(similarities, axis=1)
    objective = float(np.max(similarities, axis=1).sum())
    trace = [objective]
    empty_clusters = [False] * effective
    zero_sum_clusters = [False] * effective
    converged = False
    for iteration in range(1, max_updates + 1):
        for k in range(effective):
            members = X[assignments == k]
            if not len(members):
                empty_clusters[k] = True
                continue
            total = members.sum(axis=0)
            norm = np.linalg.norm(total)
            if norm <= 1e-12:
                zero_sum_clusters[k] = True
                continue
            centers[k] = total / norm
        similarities = X @ centers.T
        new_assignments = np.argmax(similarities, axis=1)
        updated_objective = float(np.max(similarities, axis=1).sum())
        improvement = updated_objective - objective
        if improvement < -1e-9 * len(X):
            raise ArithmeticError('spherical objective decreased beyond 1e-9 per observation')
        trace.append(updated_objective)
        objective = updated_objective
        converged = bool(np.array_equal(assignments, new_assignments) or improvement <= 1e-10 * len(X))
        if converged:
            break
        assignments = new_assignments
    # This checks the returned vectors as well as all normalized updates.
    _unit_rows(centers, 'prototypes')
    return PrototypeResult(centers, objective, iteration, {
        'requested_K': int(K), 'effective_K': effective,
        'empty_clusters': empty_clusters, 'zero_sum_clusters': zero_sum_clusters,
        'mean_fallback': mean_fallback, 'cap_hit': not converged,
        'objective_trace': trace,
    })


def _score_inputs(X, P, N):
    X, P, N = (_unit_rows(rows, name) for rows, name in [(X, 'X'), (P, 'P'), (N, 'N')])
    if X.shape[1] != P.shape[1] or X.shape[1] != N.shape[1]:
        raise ValueError('X, P, and N must have the same vector dimension')
    if not len(P):
        raise ValueError('ineligible history: positive support is empty')
    return X, P, N


def signed_scores(X, P, N, penalty: float, *, chunk_size=512) -> np.ndarray:
    """Positive maximum minus penalty times raw negative maximum; no clipping."""
    X, P, N = _score_inputs(X, P, N)
    if not np.isscalar(penalty) or not np.isfinite(penalty) or penalty < 0:
        raise ValueError('penalty must be finite and nonnegative')
    _positive_integer(chunk_size, 'chunk_size')
    scores = np.empty(len(X), dtype=np.float64)
    for start in range(0, len(X), chunk_size):
        chunk = X[start:start + chunk_size]
        positive = np.max(chunk @ P.T, axis=1)
        if len(N) and penalty != 0:
            positive -= penalty * np.max(chunk @ N.T, axis=1)
        scores[start:start + chunk_size] = positive
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite signed scores')
    return scores


def ratio_scores(X, P, N) -> np.ndarray:
    """Nearest-class Euclidean ratio d_N/(d_P+d_N), including exact zeros.

    Direct distances avoid cancellation near identical rows. Bounded cdist
    blocks use no candidate-by-support-by-dimension difference tensor and need
    no cosine clipping or denominator epsilon.
    """
    X, P, N = _score_inputs(X, P, N)
    if not len(N):
        raise ValueError('ineligible history: negative support is empty for distance ratio')
    scores = np.empty(len(X), dtype=np.float64)
    for start in range(0, len(X), 512):
        chunk = X[start:start + 512]
        dp = np.min(cdist(chunk, P, metric='euclidean'), axis=1)
        dn = np.min(cdist(chunk, N, metric='euclidean'), axis=1)
        denominator = dp + dn
        values = np.full(len(chunk), .5, dtype=np.float64)
        np.divide(dn, denominator, out=values, where=denominator != 0)
        scores[start:start + 512] = values
    return scores


@dataclass(frozen=True)
class ModelConfig:
    family: str
    parameters: dict
    id: str


def model_configuration(family, parameters):
    identity = canonical_sha256({'family': family, 'parameters': parameters})
    return ModelConfig(family, dict(parameters), f'{family}:{identity}')


def endpoint_requests(family_choices, config: StudyConfig) -> list[ModelConfig]:
    """The single prescribed expansion, including the complete new RBF corner."""
    def parameters(family):
        value = family_choices.get(family)
        if value is None:
            return {}
        return (value if isinstance(value, ModelConfig) else value['configuration']).parameters

    def outer(value, grid, expanded):
        return expanded[0] if value == grid[0] else expanded[-1] if value == grid[-1] else None

    rbf = parameters('RBFSVM')
    initial_rbf = rbf.get('C') in config.C_grid and rbf.get('g') in config.g_grid
    C = outer(rbf.get('C'), config.C_grid, config.selection['expanded_C']) if initial_rbf else None
    g = outer(rbf.get('g'), config.g_grid, config.selection['expanded_g']) if initial_rbf else None
    pairs = set()
    if C is not None:
        pairs.update((C, v) for v in config.g_grid)
    if g is not None:
        pairs.update((v, g) for v in config.C_grid)
    if C is not None and g is not None:
        pairs.add((C, g))
    entries = [model_configuration('RBFSVM', {'C': c, 'g': v}) for c,v in sorted(pairs)]
    C = outer(parameters('LinearLogistic').get('C'), config.C_grid, config.selection['expanded_C'])
    if C is not None:
        entries.append(model_configuration('LinearLogistic', {'C': C}))
    for family in ('MeanNegative', 'AllExemplarMargin', 'SignedMax'):
        p = parameters(family)
        if p.get('lambda') == config.lambda_grid[-1]:
            entries.append(model_configuration(family, {**p, 'lambda': config.selection['expanded_lambda']}))
    return entries


@dataclass(frozen=True)
class UserModel:
    method: str
    parameters: dict
    support_ids: tuple[int, ...]
    state: dict
    diagnostics: dict


def initial_configurations(config: StudyConfig) -> list[ModelConfig]:
    """All declared entries, including score-equivalent entries, in fixed order.

    IDs bind family and canonical parameter values, independent of list order.
    Centroid/PositiveMax/RocchioDirection are views of these entries only.
    Accounting remains the caller's responsibility even when it reuses scores.
    """
    entries = []

    def add(family, parameters):
        entries.append(model_configuration(family, parameters))

    for family in ('MeanNegative', 'SignedMax'):
        for K in config.K_grid:
            for penalty in config.lambda_grid:
                add(family, {'K': K, 'lambda': penalty})
    for penalty in config.lambda_grid:
        add('AllExemplarMargin', {'lambda': penalty})
    for K in config.K_grid:
        add('PrototypeRatio', {'K': K})
    add('ExemplarRatio', {})
    for C in config.C_grid:
        add('LinearLogistic', {'C': C})
    for C in config.C_grid:
        for g in config.g_grid:
            add('RBFSVM', {'C': C, 'g': g})
    return entries


def _adapter_parameters(method, parameters):
    expected = {
        'MeanNegative': {'K', 'lambda'}, 'SignedMax': {'K', 'lambda'},
        'AllExemplarMargin': {'lambda'}, 'PrototypeRatio': {'K'},
        'ExemplarRatio': set(), 'LinearLogistic': {'C'}, 'RBFSVM': {'C', 'g'},
        'Centroid': set(), 'PositiveMax': {'K'}, 'RocchioDirection': {'lambda'},
    }
    if method not in expected:
        raise ValueError(f'unknown method: {method}')
    parameters = dict(parameters)
    if set(parameters) != expected[method]:
        raise ValueError(f'{method} requires parameters {sorted(expected[method])}')
    if 'K' in parameters:
        _positive_integer(parameters['K'], 'K')
    for key in ('lambda', 'C', 'g'):
        if key in parameters:
            value = parameters[key]
            if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
                    or not np.isfinite(value) or value < 0 or (key != 'lambda' and value == 0)):
                raise ValueError(f'{key} must be finite and {"nonnegative" if key == "lambda" else "positive"}')
    return parameters


def _item_ids(ids, *, distinct=False):
    ids = tuple(ids)
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in ids):
        raise ValueError('item IDs must be integers')
    if distinct and len(set(ids)) != len(ids):
        raise ValueError('support item IDs must be distinct across both classes')
    return ids


def _failed(diagnostics, failure):
    diagnostics['status'] = 'failed'
    if failure not in diagnostics['failures']:
        diagnostics['failures'].append(failure)


def fit_user(method, history: History, embeddings, parameters, *, config: StudyConfig | None = None) -> UserModel:
    """Fit from capped History only, retaining no sklearn estimator objects.

    Production callers pass the same validated resolved config for every fit.
    Its pre-freeze global iteration repairs apply without per-user retries.
    Omission is intended for pure fixtures and uses nominal protocol limits.
    Failed fits return empty state plus persistable diagnostics; callers must
    save those diagnostics and apply the study's bounded failure policy.
    """
    parameters = _adapter_parameters(method, parameters)
    positive_ids, negative_ids = tuple(history.positive_ids), tuple(history.negative_ids)
    support_ids = _item_ids(positive_ids + negative_ids, distinct=True)
    if not positive_ids:
        raise ValueError('ineligible history: positive support is empty')
    if not negative_ids and method in ('LinearLogistic', 'RBFSVM', 'PrototypeRatio', 'ExemplarRatio'):
        raise ValueError(f'ineligible history: negative support is empty for {method}')
    support = np.ascontiguousarray(_unit_rows(embeddings.rows(support_ids), 'support'), dtype=np.float64)
    if len(support) != len(support_ids):
        raise ValueError('embedding rows must match support item IDs')
    positive, negative = support[:len(positive_ids)], support[len(positive_ids):]
    limits = {
        'prototype_max_updates': 50 if config is None else config.prototype['max_updates'],
        'logistic_max_iter': 2000 if config is None else config.linear_logistic['max_iter'],
        'svm_max_iter': 100000 if config is None else config.rbf_svm['max_iter'],
    }
    diagnostics = {
        'status': 'ok', 'failures': [], 'warnings': [],
        'policy': {**limits, 'source': 'nominal_fixture' if config is None else 'resolved_config',
                   'pre_freeze_repairs': [] if config is None else [dict(r) for r in config.pre_freeze_repairs]},
    }
    model = UserModel(method, parameters, support_ids, {}, diagnostics)
    if method not in ('LinearLogistic', 'RBFSVM'):
        if method in ('AllExemplarMargin', 'ExemplarRatio'):
            P = positive.copy()
        else:
            diagnostics['iteration_limit'] = limits['prototype_max_updates']
            try:
                prototypes = spherical_prototypes(positive, np.asarray(positive_ids), parameters.get('K', 1),
                                                  max_updates=limits['prototype_max_updates'])
            except Exception as error:
                _failed(diagnostics, 'exception')
                diagnostics['exception'] = {'category': type(error).__name__, 'message': str(error)}
                return model
            P = prototypes.vectors
            diagnostics['prototype'] = {'objective': prototypes.objective, 'iterations': prototypes.iterations,
                                        **prototypes.flags}
        model.state.update(P=P, N=negative.copy())
        return model

    labels = np.concatenate((np.ones(len(positive), dtype=np.int64), np.zeros(len(negative), dtype=np.int64)))
    weights = {0: len(support) / (2 * len(negative)), 1: len(support) / (2 * len(positive))}
    diagnostics['class_weights'] = weights
    if method == 'LinearLogistic':
        limit = limits['logistic_max_iter']
        # sklearn 1.9: l1_ratio=0 expresses L2 without deprecated penalty=.
        settings = dict(C=parameters['C'], l1_ratio=0, solver='liblinear', fit_intercept=True,
                        class_weight=weights, tol=1e-8, max_iter=limit, random_state=0)
        estimator = LogisticRegression(**settings)
    else:
        limit = limits['svm_max_iter']
        distances = pdist(support, metric='sqeuclidean')
        positive_distances = distances[distances > 0]
        m2 = float(np.median(positive_distances)) if len(positive_distances) else 1.0
        gamma = float(parameters['g'] / m2)
        settings = dict(C=parameters['C'], gamma=gamma, kernel='rbf', class_weight=weights,
                        tol=1e-6, max_iter=limit, cache_size=64)
        # sklearn 1.9's omitted probability sentinel disables calibration.
        estimator = SVC(**settings)
        model.state.update(m2=m2, gamma=gamma)
    diagnostics.update(iteration_limit=limit, solver_settings=settings)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            estimator.fit(support, labels)
            iterations = np.asarray(estimator.n_iter_, dtype=np.int64)
            diagnostics['n_iter'] = iterations.tolist()
            if np.any(iterations >= limit):
                _failed(diagnostics, 'iterations')
            if method == 'RBFSVM':
                diagnostics['fit_status'] = int(estimator.fit_status_)
                if estimator.fit_status_ != 0:
                    _failed(diagnostics, 'status')
                model.state.update(support_vectors=np.ascontiguousarray(estimator.support_vectors_, dtype=np.float64),
                                   dual_coef=np.ascontiguousarray(estimator.dual_coef_[0], dtype=np.float64))
            else:
                model.state['coef'] = np.ascontiguousarray(estimator.coef_[0], dtype=np.float64)
            model.state['intercept'] = float(estimator.intercept_[0])
            training_scores = estimator.decision_function(support)
            if (not np.isfinite(training_scores).all()
                    or any(not np.isfinite(value).all() for value in model.state.values())):
                _failed(diagnostics, 'nonfinite')
        except Exception as error:
            _failed(diagnostics, 'exception')
            diagnostics['exception'] = {'category': type(error).__name__, 'message': str(error)}
    diagnostics['warnings'] = [{'category': w.category.__name__, 'message': str(w.message)} for w in caught]
    if caught:
        _failed(diagnostics, 'warning')
    del estimator
    if diagnostics['status'] == 'failed':
        model.state.clear()
    return model


def score_user(model: UserModel, candidate_ids, embeddings, *, chunk_size=512) -> np.ndarray:
    """Uncalibrated float64 decisions, with bounded candidate/support blocks.

    Scoring failures update the model's persistable diagnostics before raising.
    No estimator is kept alive; binary SVC decisions use its saved RBF expansion.
    """
    _positive_integer(chunk_size, 'chunk_size')
    candidate_ids = _item_ids(candidate_ids)
    if model.diagnostics['status'] != 'ok':
        raise ValueError('cannot score failed fit; persist model diagnostics')
    scores = np.empty(len(candidate_ids), dtype=np.float64)
    try:
        for start in range(0, len(candidate_ids), chunk_size):
            ids = candidate_ids[start:start + chunk_size]
            X = np.ascontiguousarray(_unit_rows(embeddings.rows(ids), 'candidates'), dtype=np.float64)
            if len(X) != len(ids):
                raise ValueError('embedding rows must match candidate item IDs')
            state, method = model.state, model.method
            with np.errstate(over='ignore', invalid='ignore'):
                if method == 'LinearLogistic':
                    values = X @ state['coef'] + state['intercept']
                elif method == 'RBFSVM':
                    kernel = np.exp(-state['gamma'] * cdist(X, state['support_vectors'], metric='sqeuclidean'))
                    values = kernel @ state['dual_coef'] + state['intercept']
                elif method in ('PrototypeRatio', 'ExemplarRatio'):
                    values = ratio_scores(X, state['P'], state['N'])
                elif method in ('MeanNegative', 'RocchioDirection'):
                    values = np.max(X @ state['P'].T, axis=1)
                    if len(state['N']):
                        values -= model.parameters['lambda'] * (X @ state['N'].mean(axis=0))
                else:
                    values = signed_scores(X, state['P'], state['N'], model.parameters.get('lambda', 0),
                                           chunk_size=chunk_size)
            if not np.isfinite(values).all():
                _failed(model.diagnostics, 'nonfinite')
                raise ValueError('nonfinite decision scores')
            scores[start:start + len(ids)] = values
    except Exception as error:
        _failed(model.diagnostics, 'nonfinite' if 'nonfinite' in str(error) else 'scoring')
        model.diagnostics['score_exception'] = {'category': type(error).__name__, 'message': str(error)}
        raise
    return scores
