# test_affinitymat.py
# The inverted-index neighbour search gives the same result as 2.5.2's
# all-pairs _sparse_mm_dot, bit for bit.

import numpy as np
import pytest
import scipy.sparse

from modiscolite import affinitymat


def _gkmer_like(n, n_features=3000, nnz=60, seed=0):
    # Rows like normalised gapped k-mer vectors: column indices drawn from a
    # skewed vocabulary (a few k-mers shared by many rows, as within a motif),
    # signed values, L2-normalised; a few duplicate rows (exact ties) and an
    # empty one (a seqlet with no k-mers).
    rng = np.random.default_rng(seed)
    weights = 1.0 / np.arange(1, n_features + 1) ** 1.1
    weights /= weights.sum()
    rows, cols, vals = [], [], []
    for i in range(n):
        if i == n // 2:
            continue
        c = np.unique(rng.choice(n_features, nnz, p=weights))
        rows.extend([i] * len(c))
        cols.extend(c * 7919)  # spread out, like the 5**15-wide k-mer index
        vals.extend(rng.normal(0.3, 1.0, len(c)))
    X = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, n_features * 7919))
    X = X.tolil()
    for i in range(0, n, 97):
        X[i + 1] = X[i]
    X = X.tocsr()
    X.sort_indices()
    norms = np.sqrt(np.asarray(X.multiply(X).sum(1))).ravel()
    norms[norms == 0] = 1.0
    return scipy.sparse.csr_matrix(scipy.sparse.diags(1 / norms) @ X)


def _bits(a):
    return np.ascontiguousarray(a).view(np.int64)


@pytest.mark.parametrize('k', [1, 10, 151, 400])
def test_same_as_all_pairs(k):
    X = _gkmer_like(400, seed=1)
    Y = _gkmer_like(400, seed=2)
    X.sort_indices()
    Y.sort_indices()
    sims_ref, nbrs_ref = affinitymat._sparse_mm_dot(X.data, X.indices.astype('int64'),
        X.indptr.astype('int64'), Y.data, Y.indices.astype('int64'), Y.indptr.astype('int64'), k)
    sims, nbrs = affinitymat._sparse_mm_dot_indexed(X, Y, k)
    np.testing.assert_array_equal(nbrs, nbrs_ref)
    np.testing.assert_array_equal(_bits(sims), _bits(sims_ref))
    assert sims.dtype == sims_ref.dtype and nbrs.dtype == nbrs_ref.dtype


def test_rows_with_few_overlaps_fill_with_zeros_then_negatives():
    # A sparse vocabulary: most rows share nothing, so the top k runs into
    # the zero dots and, at k = n, the negative ones.
    X = _gkmer_like(150, n_features=50000, nnz=5, seed=3)
    Y = _gkmer_like(150, n_features=50000, nnz=5, seed=4)
    X.sort_indices()
    Y.sort_indices()
    for k in (20, 150):
        ref = affinitymat._sparse_mm_dot(X.data, X.indices.astype('int64'), X.indptr.astype('int64'),
            Y.data, Y.indices.astype('int64'), Y.indptr.astype('int64'), k)
        got = affinitymat._sparse_mm_dot_indexed(X, Y, k)
        np.testing.assert_array_equal(got[1], ref[1])
        np.testing.assert_array_equal(_bits(got[0]), _bits(ref[0]))
    assert (ref[0] < 0).any() and (ref[0] == 0).any()


def test_unsorted_rows_fall_back_to_all_pairs():
    X = _gkmer_like(120, seed=5)
    Y = _gkmer_like(120, seed=6)
    X = X[:, ::-1][:, ::-1]  # same matrix, indices not known to be sorted
    X.has_sorted_indices = False
    ref = affinitymat._sparse_mm_dot(X.data, X.indices.astype('int64'), X.indptr.astype('int64'),
        Y.data, Y.indices.astype('int64'), Y.indptr.astype('int64'), 30)
    got = affinitymat._sparse_mm_dot_indexed(X, Y, 30)
    np.testing.assert_array_equal(got[1], ref[1])
    np.testing.assert_array_equal(_bits(got[0]), _bits(ref[0]))
