# test_extract_seqlets.py
# extract_seqlets' score distribution: np.sort gives exactly what
# np.array(sorted(...)) gave in 2.5.2.

import numpy as np


def test_np_sort_equals_builtin_sorted():
    rng = np.random.default_rng(0)
    tracks = rng.laplace(0, 0.01, (300, 481)).astype('float32')
    tracks[::7, ::5] = 0.0  # ties, and signed zeros
    tracks[::11, ::3] = -0.0
    tracks[5] = tracks[4]
    vals = np.abs(np.concatenate(tracks, axis=0))
    old = np.array(sorted(vals))
    new = np.sort(vals)
    assert old.dtype == new.dtype == np.float32
    np.testing.assert_array_equal(old.view(np.int32), new.view(np.int32))
