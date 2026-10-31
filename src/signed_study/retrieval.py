"""Optional bounded catalog Gram retrieval, independent of frozen study scoring.

Cached scores are float64 approximations to the same real formula. ``topk``
defaults to a full reference call: no empirical tolerance certifies BLAS ties.
Explicit ``mode='fast'`` ranks the cache's own scores and can disagree at ties.
No dataset, encoder, outcomes, Torch runtime or network access is used here.
"""
from dataclasses import dataclass
import hashlib
from types import MappingProxyType

import numpy as np


def _integer(value, name, *, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _ids(values):
    try:
        values = list(values)
    except TypeError as error:
        raise ValueError('IDs must be an iterable of exact int64 integers') from error
    for value in values:
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not -(2**63) <= int(value) < 2**63):
            raise ValueError('IDs must be exact int64 integers')
    return np.asarray(values, dtype=np.int64)


def _rows(value, dimensions=None, *, nonempty=False):
    try:
        value = np.asarray(value)
        if np.iscomplexobj(value):
            raise ValueError('Geometry must contain real float64 unit rows')
        value = np.asarray(value, dtype=np.float64)
    except (TypeError,ValueError) as error:
        raise ValueError('Geometry must contain real float64 unit rows') from error
    if (value.ndim != 2 or value.shape[1] < 1 or
            dimensions is not None and value.shape[1] != dimensions or nonempty and not len(value)
            or not np.isfinite(value).all()
            or np.any(np.abs(np.linalg.norm(value, axis=1)-1) > 1e-12)):
        raise ValueError('Expected aligned finite unit rows (norm tolerance 1e-12)')
    return value


def _immutable(value):
    value = np.asarray(value)
    return np.frombuffer(value.tobytes(order='C'), dtype=value.dtype).reshape(value.shape)


def _penalty(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int,float,np.integer,np.floating)):
        raise ValueError('penalty must be finite and nonnegative')
    try:
        value=float(value)
    except (OverflowError,TypeError,ValueError) as error:
        raise ValueError('penalty must be a finite nonnegative float64 value') from error
    if not np.isfinite(value) or value < 0:
        raise ValueError('penalty must be finite and nonnegative')
    return value


@dataclass(frozen=True)
class TopKResult:
    """Ranked IDs/scores and the explicit numerical mode used to select them."""
    item_ids: np.ndarray
    scores: np.ndarray
    diagnostics: dict


class GramCatalog:
    """Own unit geometry/IDs and one private float64 N×N matrix.

    The 256 MiB default cap covers the matrix alone. Geometry has immutable byte
    backing, and row results are independent copies. The matrix is deliberately
    private and never returned as a public view; Python private attributes are
    not an adversarial security boundary. Building writes directly into matrix
    row blocks without creating a second full Gram matrix.
    """

    def __init__(self, item_ids, vectors, *, max_gram_bytes=256*1024**2,
                 build_block_size=256, manifest_sha256=None):
        ids = _ids(item_ids)
        X = _rows(vectors, nonempty=True)
        if len(ids) != len(X) or len(set(map(int, ids))) != len(ids):
            raise ValueError('Catalog IDs must be distinct and aligned with rows')
        cap = _integer(max_gram_bytes, 'max_gram_bytes', minimum=0)
        block = _integer(build_block_size, 'build_block_size')
        required = len(ids)*len(ids)*8  # Python integers cannot wrap before refusal.
        if required > cap:
            raise MemoryError(f'Gram matrix requires {required} bytes; matrix cap is {cap}; use dense scoring')
        if manifest_sha256 is not None and (not isinstance(manifest_sha256, str)
                or len(manifest_sha256) != 64 or any(c not in '0123456789abcdef' for c in manifest_sha256)):
            raise ValueError('manifest_sha256 must be a lowercase SHA256 digest')
        self._ids = _immutable(ids)
        self._vectors = _immutable(X)
        self._index = MappingProxyType({int(item):i for i,item in enumerate(ids)})
        self._gram = np.empty((len(ids),len(ids)), dtype=np.float64)
        for start in range(0,len(ids),block):
            np.matmul(self._vectors[start:start+block], self._vectors.T,
                      out=self._gram[start:start+block])
            if not np.isfinite(self._gram[start:start+block]).all():
                raise ValueError('nonfinite Gram matrix')
        self._gram.flags.writeable = False
        digest = hashlib.sha256()
        for part in (self._ids.tobytes(), self._vectors.tobytes(), (manifest_sha256 or '').encode()):
            digest.update(len(part).to_bytes(8,'big')); digest.update(part)
        self.identity_sha256 = digest.hexdigest()
        self.manifest_sha256 = manifest_sha256
        self.build_block_size = block

    @property
    def item_ids(self):
        return self._ids

    @property
    def vectors(self):
        return self._vectors

    @property
    def memory(self):
        import sys
        return dict(gram_bytes=self._gram.nbytes, catalog_bytes=self._vectors.nbytes,
            id_bytes=self._ids.nbytes,
            index_estimated_bytes=sys.getsizeof(dict(self._index)) + sum(
                sys.getsizeof(k)+sys.getsizeof(v) for k,v in self._index.items()),
            build_block_rows=min(len(self._ids),self.build_block_size),
            workspace_policy='query chunks plus one negative row; BLAS workspace excluded')

    def _indices(self, ids):
        return np.fromiter((self._index[int(item)] for item in ids), dtype=np.intp, count=len(ids))

    def rows(self, ids):
        """Gather independent float64 rows, preserving order and duplicate IDs."""
        return self._vectors[self._indices(_ids(ids))]

    def _query(self, P, negative_ids, penalty, candidate_ids, chunk_size):
        chunk = _integer(chunk_size, 'chunk_size')
        P = _rows(P, self._vectors.shape[1], nonempty=True)
        negatives = _ids(negative_ids)
        ni = self._indices(negatives)
        ids = self._ids if candidate_ids is None else _ids(candidate_ids)
        ci = self._indices(ids)
        return P, negatives, ni, _penalty(penalty), ids, ci, chunk

    def scores(self, P, negative_ids, penalty, *, candidate_ids=None, chunk_size=512):
        """Return cached-formula scores in caller order, with raw negative cosines.

        Workspaces scale as chunk×(dimension + number of positive prototypes),
        plus scalar vectors. Negative supports are read one matrix row at a time.
        Full catalogs use contiguous row slices. No candidate×support×dimension
        tensor or second full matrix is allocated.
        """
        P, _, ni, penalty, ids, ci, chunk = self._query(P,negative_ids,penalty,candidate_ids,chunk_size)
        return self._scores(P,ni,penalty,ids,ci,chunk)

    def _scores(self, P, ni, penalty, ids, ci, chunk):
        scores = np.empty(len(ids), dtype=np.float64)
        full = np.array_equal(ids,self._ids)
        # Near-full workloads favor contiguous matrix rows plus one scalar gather.
        # Small subsets keep workspace and row reads proportional to the request.
        near_full = len(ids) >= .75*len(self._ids)
        catalog_negative = None
        if len(ni) and penalty and near_full:
            catalog_negative = np.full(len(self._ids),-np.inf)
            for row in ni:
                np.maximum(catalog_negative,self._gram[row],out=catalog_negative)
        with np.errstate(over='ignore', invalid='ignore'):
            for start in range(0,len(ids),chunk):
                stop = min(start+chunk,len(ids))
                X = self._vectors[start:stop] if full else self._vectors[ci[start:stop]]
                positive = np.full(stop-start,-np.inf)
                for p_start in range(0,len(P),chunk):
                    np.maximum(positive,np.max(X @ P[p_start:p_start+chunk].T,axis=1),out=positive)
                if len(ni) and penalty:
                    if catalog_negative is not None:
                        negative = catalog_negative[start:stop] if full else catalog_negative[ci[start:stop]]
                    else:
                        negative = np.full(stop-start,-np.inf)
                        for row in ni:
                            np.maximum(negative,self._gram[row,ci[start:stop]],out=negative)
                    positive -= penalty*negative
                scores[start:stop] = positive
        if not np.isfinite(scores).all():
            raise ValueError('nonfinite cached scores')
        return scores

    @staticmethod
    def _rank(ids, scores, k, mode):
        k = _integer(k,'k')
        if k > len(ids):
            raise ValueError('k exceeds candidate count')
        if len(set(map(int,ids))) != len(ids):
            raise ValueError('Top-k candidates must be distinct')
        if not np.isfinite(scores).all():
            raise ValueError('nonfinite scores before ranking')
        order = np.lexsort((ids,-scores))[:k]
        return TopKResult(_immutable(ids[order]),_immutable(scores[order]),
            dict(mode=mode, reference_fallback=mode=='strict',
                 reference_policy='full requested order and chunk; no separation certificate'))

    def topk(self, P, negative_ids, penalty, *, k, candidate_ids=None, chunk_size=512, mode='strict'):
        """Strict ranks the full signed_scores call; fast ranks its own scores."""
        from .models import signed_scores
        P, negatives, ni, penalty, ids, ci, chunk = self._query(P,negative_ids,penalty,candidate_ids,chunk_size)
        if mode == 'strict':
            with np.errstate(over='ignore',invalid='ignore'):
                scores = signed_scores(self._vectors[ci],P,self._vectors[ni],penalty,chunk_size=chunk)
        elif mode == 'fast':
            scores = self._scores(P,ni,penalty,ids,ci,chunk)
        else:
            raise ValueError('mode must be strict or fast')
        return self._rank(ids,scores,k,mode)

    def topk_user(self, model, candidate_ids, *, k, chunk_size=512, mode='strict'):
        """Strict reference is the existing score_user on this geometry/model."""
        from .models import score_user
        ids = _ids(candidate_ids)
        _model_query(model,self)  # Check supports even when strict uses dense scoring.
        if mode == 'strict':
            scores = score_user(model,ids,self,chunk_size=chunk_size)
        elif mode == 'fast':
            scores = score_user_cached(model,ids,self,chunk_size=chunk_size)
        else:
            raise ValueError('mode must be strict or fast')
        return self._rank(ids,scores,k,mode)


def _model_query(model, catalog):
    from .models import UserModel, _adapter_parameters
    if (not isinstance(model,UserModel) or not isinstance(model.diagnostics,dict)
            or not isinstance(model.parameters,dict) or not isinstance(model.state,dict)
            or model.diagnostics.get('status') != 'ok'
            or model.method not in ('SignedMax','AllExemplarMargin','PositiveMax','Centroid')):
        raise ValueError('Expected a successful supported UserModel')
    try:
        parameters = _adapter_parameters(model.method,model.parameters)
        P = _rows(model.state['P'],catalog.vectors.shape[1],nonempty=True)
        N = _rows(model.state['N'],catalog.vectors.shape[1])
        supports = _ids(model.support_ids)
        if len(supports) <= len(N) or len(set(map(int,supports))) != len(supports):
            raise ValueError('Missing or duplicated model support IDs')
        catalog._indices(supports)
        negative_ids = supports[-len(N):] if len(N) else supports[:0]
        if not np.array_equal(catalog.rows(negative_ids),N):
            raise ValueError('Catalog negative support identity differs from model vectors')
        penalty = _penalty(parameters.get('lambda',0))
    except (KeyError,TypeError) as error:
        raise ValueError('Malformed model support/state') from error
    return P,negative_ids,penalty


def score_user_cached(model, candidate_ids, catalog: GramCatalog, *, chunk_size=512):
    """Optional cached scores; revalidate mutable model supports on every call.

    Equality of negative rows proves support consistency, not universal encoder
    identity. Empty-negative models have no negative geometry to verify. Bind
    external provenance to catalog.identity_sha256/manifest_sha256 when available.
    This API does not promise bit-identical canonical scores or ranked IDs.
    """
    P,negative_ids,penalty = _model_query(model,catalog)
    return catalog.scores(P,negative_ids,penalty,candidate_ids=candidate_ids,chunk_size=chunk_size)
