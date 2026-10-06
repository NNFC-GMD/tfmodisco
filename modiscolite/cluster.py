# cluster.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar

import os
import shutil
import subprocess
import sys
import tempfile

import leidenalg
import numpy as np
import igraph as ig

# Below this many stored entries in the affinity matrix the restarts stay in
# this process even when n_jobs > 1: a worker takes about a second to start and
# has to rebuild the graph, which is as long as a restart on a graph this size
# takes. The result is the same either way.
PARALLEL_MIN_EDGES = 1000000

# A worker loads this file by path instead of importing the package, so it does
# not pay for importing all of modiscolite (several seconds) and does not touch
# sys.path. argv: this file, then _worker's arguments.
_WORKER = (
    "import importlib.util, sys\n"
    "spec = importlib.util.spec_from_file_location('_modisco_leiden_worker', sys.argv[1])\n"
    "module = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(module)\n"
    "module._worker(sys.argv[2:])\n")


def _graph(n_vertices, indptr, indices):
    n_cols = indptr
    sources = np.concatenate([np.ones(n_cols[i+1] - n_cols[i], dtype='int32') * i for i in range(n_vertices)])

    g = ig.Graph(directed=None)
    g.add_vertices(n_vertices)
    g.add_edges(zip(sources, indices))
    return g


def _restart(g, weights, seed, n_leiden_iterations):
    partition = leidenalg.find_partition(
        graph=g,
        partition_type=leidenalg.ModularityVertexPartition,
        weights=weights,
        n_iterations=n_leiden_iterations,
        initial_membership=None,
        seed=seed*100)

    return np.array(partition.quality()), np.array(partition.membership)


def _best(results):
    """The membership of the best restart; on a tie the lowest seed wins."""
    best_clustering = None
    best_quality = None

    for quality, membership in results:
        if best_quality is None or quality > best_quality:
            best_quality = quality
            best_clustering = membership

    return best_clustering


def _worker(argv):
    """Run some of the restarts in a separate process: argv is the directory
    holding the matrix, the number of vertices, n_leiden_iterations and the
    seeds. Each seed's quality and membership are saved next to the matrix."""
    tmp_dir, n_vertices, n_leiden_iterations = argv[0], int(argv[1]), int(argv[2])
    indptr = np.load(os.path.join(tmp_dir, 'indptr.npy'))
    indices = np.load(os.path.join(tmp_dir, 'indices.npy'))
    data = np.load(os.path.join(tmp_dir, 'data.npy'))

    g = _graph(n_vertices, indptr, indices)
    for seed in map(int, argv[3:]):
        quality, membership = _restart(g, data, seed, n_leiden_iterations)
        np.save(os.path.join(tmp_dir, 'quality_{}.npy'.format(seed)), quality)
        np.save(os.path.join(tmp_dir, 'membership_{}.npy'.format(seed)), membership)


def _parallel_restarts(affinity_mat, n_seeds, n_leiden_iterations, n_jobs):
    """The restarts in up to n_jobs fresh processes, not forked ones: by the
    time clustering runs, numba's thread pool is up, and forking a process with
    live threads can deadlock the child."""
    seeds = list(range(1, n_seeds+1))
    n_workers = min(n_jobs, n_seeds)

    tmp_dir = tempfile.mkdtemp(prefix='modisco_leiden_')
    try:
        np.save(os.path.join(tmp_dir, 'indptr.npy'), affinity_mat.indptr)
        np.save(os.path.join(tmp_dir, 'indices.npy'), affinity_mat.indices)
        np.save(os.path.join(tmp_dir, 'data.npy'), affinity_mat.data)

        workers = []
        for i in range(n_workers):
            cmd = [sys.executable, '-c', _WORKER, os.path.abspath(__file__), tmp_dir,
                str(affinity_mat.shape[0]), str(n_leiden_iterations)]
            cmd += [str(seed) for seed in seeds[i::n_workers]]

            log_path = os.path.join(tmp_dir, 'worker_{}.log'.format(i))
            with open(log_path, 'w') as log:
                workers.append((log_path, subprocess.Popen(cmd, cwd=tmp_dir,
                    stdout=subprocess.DEVNULL, stderr=log)))

        failed = []
        for log_path, process in workers:
            if process.wait() != 0:
                with open(log_path) as log:
                    failed.append('exit {}: {}'.format(process.returncode, log.read()[-2000:]))
        if failed:
            raise RuntimeError('Leiden worker failed:\n' + '\n'.join(failed))

        return _best((np.load(os.path.join(tmp_dir, 'quality_{}.npy'.format(seed))),
            np.load(os.path.join(tmp_dir, 'membership_{}.npy'.format(seed))))
            for seed in seeds)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def LeidenCluster(affinity_mat, n_seeds=2, n_leiden_iterations=-1, n_jobs=1):
    """Leiden clustering of a sparse affinity matrix, keeping the best of
    n_seeds restarts.

    n_jobs > 1 runs the restarts in up to that many worker processes when the
    matrix has at least PARALLEL_MIN_EDGES stored entries. Each restart is
    seeded and independent of the others, and the best is chosen in seed order
    either way, so the clustering does not depend on n_jobs. The workers read
    the matrix from a temporary directory (TMPDIR; 12 to 16 bytes per stored entry)
    and each builds its own copy of the graph.
    """
    if n_jobs > 1 and n_seeds > 1 and affinity_mat.nnz >= PARALLEL_MIN_EDGES:
        return _parallel_restarts(affinity_mat, n_seeds, n_leiden_iterations, n_jobs)

    g = _graph(affinity_mat.shape[0], affinity_mat.indptr, affinity_mat.indices)
    return _best(_restart(g, affinity_mat.data, seed, n_leiden_iterations)
        for seed in range(1, n_seeds+1))
