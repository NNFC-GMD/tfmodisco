# test_cluster.py
# The Leiden restarts give the same clustering whether they run in this
# process or in worker processes (n_jobs), and the same as 2.5.2's loop.

import igraph as ig
import leidenalg
import numpy as np
import pytest
import scipy.sparse

from modiscolite import cluster


def _leiden_2_5_2(affinity_mat, n_seeds=2, n_leiden_iterations=-1):
    # LeidenCluster as released in 2.5.2, verbatim.
    n_vertices = affinity_mat.shape[0]
    n_cols = affinity_mat.indptr
    sources = np.concatenate([np.ones(n_cols[i+1] - n_cols[i], dtype='int32') * i for i in range(n_vertices)])

    g = ig.Graph(directed=None)
    g.add_vertices(n_vertices)
    g.add_edges(zip(sources, affinity_mat.indices))

    best_clustering = None
    best_quality = None

    for seed in range(1, n_seeds+1):
        partition = leidenalg.find_partition(
            graph=g,
            partition_type=leidenalg.ModularityVertexPartition,
            weights=affinity_mat.data,
            n_iterations=n_leiden_iterations,
            initial_membership=None,
            seed=seed*100)

        quality = np.array(partition.quality())
        membership = np.array(partition.membership)

        if best_quality is None or quality > best_quality:
            best_quality = quality
            best_clustering = membership

    return best_clustering


def _affinity(n=1500, k=40, n_communities=6, mixing=0.3, seed=0):
    # A symmetric sparse affinity like the density-adapted one TFMoDISco
    # clusters: both (i, j) and (j, i) stored, float64 weights. Loose
    # communities, so the restarts do not all land on the same partition.
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n), k)
    size = n // n_communities
    same = rows % n_communities + n_communities * rng.integers(0, size, rows.size)
    cols = np.where(rng.random(rows.size) < 1 - mixing, same, rng.integers(0, n, rows.size))
    a = scipy.sparse.csr_matrix((rng.random(rows.size), (rows, cols)), shape=(n, n), dtype='float64')
    a += a.T
    a /= a.data.sum()
    return a


@pytest.fixture
def always_parallel(monkeypatch):
    monkeypatch.setattr(cluster, 'PARALLEL_MIN_EDGES', 0)


@pytest.mark.parametrize('n_seeds', [1, 2, 7])
@pytest.mark.parametrize('n_jobs', [1, 2, 3, 8])
def test_same_as_2_5_2(always_parallel, n_seeds, n_jobs):
    a = _affinity()
    expected = _leiden_2_5_2(a, n_seeds=n_seeds)
    got = cluster.LeidenCluster(a, n_seeds=n_seeds, n_jobs=n_jobs)
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)


def test_restarts_differ():
    # The comparison above means something only if the seeds disagree.
    a = _affinity()
    g = cluster._graph(a.shape[0], a.indptr, a.indices)
    qualities = {float(cluster._restart(g, a.data, seed, -1)[0]) for seed in range(1, 8)}
    assert len(qualities) > 1


def test_small_matrix_stays_in_process(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('started workers for a small matrix')

    monkeypatch.setattr(cluster, '_parallel_restarts', fail)
    a = _affinity(n=200, k=10)
    np.testing.assert_array_equal(cluster.LeidenCluster(a, n_seeds=3, n_jobs=4),
        _leiden_2_5_2(a, n_seeds=3))


def test_tie_goes_to_the_lowest_seed():
    first, second = np.array([0, 0, 1]), np.array([1, 1, 0])
    assert cluster._best([(np.array(1.0), first), (np.array(1.0), second)]) is first
    assert cluster._best([(np.array(0.5), first), (np.array(1.0), second)]) is second


def test_worker_failure_raises(always_parallel, monkeypatch):
    monkeypatch.setattr(cluster, '_WORKER', "import sys; sys.stderr.write('boom'); sys.exit(3)")
    with pytest.raises(RuntimeError, match='exit 3: boom'):
        cluster.LeidenCluster(_affinity(n=200, k=10), n_seeds=2, n_jobs=2)
