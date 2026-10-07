# test_aggregator.py
# The pattern-merging AUROC from arrays equals 2.5.2's from Python lists.

import numpy as np
from sklearn.metrics import roc_auc_score

from modiscolite import affinitymat, aggregator


def _jaccard_scores(n1, n2, seed):
    rng = np.random.default_rng(seed)
    a = np.round(rng.normal(0, 1, (n1, 40)), 1).astype('float32')
    b = np.round(rng.normal(0.2, 1, (n2, 40)), 1).astype('float32')
    m = min(n1, n2) // 4
    b[:m] = a[:m]  # identical rows: ties
    between = affinitymat.jaccard(a[:, :, None], b[:, :, None])[:, :, 0].flatten()
    within = affinitymat.jaccard(a[:, :, None], a[:, :, None])[:, :, 0].flatten()
    return between, within


def test_pair_auroc_same_as_lists():
    for n1, n2, seed in ((300, 200, 0), (50, 400, 1), (1000, 1000, 2)):
        between, within = _jaccard_scores(n1, n2, seed)
        ref = roc_auc_score(y_true=[0 for x in between] + [1 for x in within],
            y_score=list(between) + list(within))
        got = aggregator._pair_auroc(between, within)
        assert float(got).hex() == float(ref).hex()
