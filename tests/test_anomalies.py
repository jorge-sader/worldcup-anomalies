"""Unit tests for the anomaly-scoring math (pure — no network)."""

import numpy as np
import pandas as pd

from worldcup_anomalies.anomalies import _bh_qvalues, _robust_z


def test_bh_qvalues_bounds_and_monotonic():
    p = np.array([0.001, 0.01, 0.03, 0.5, 0.9])
    q = _bh_qvalues(p)
    assert np.all((q >= 0) & (q <= 1))
    # BH q-values are monotone non-decreasing in the p-value ordering.
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)
    # Each q >= its p.
    assert np.all(q + 1e-12 >= p)


def test_bh_handles_nan():
    p = np.array([0.01, np.nan, 0.04])
    q = _bh_qvalues(p)
    assert np.isnan(q[1])
    assert not np.isnan(q[0]) and not np.isnan(q[2])


def test_bh_known_value():
    # Two p-values 0.01, 0.02 with n=2: q_i = p_i * n / rank.
    q = _bh_qvalues(np.array([0.01, 0.02]))
    assert np.isclose(q[0], 0.02)   # 0.01 * 2 / 1
    assert np.isclose(q[1], 0.02)   # 0.02 * 2 / 2


def test_robust_z_flags_outlier():
    s = pd.Series([1, 1, 1, 1, 1, 10.0])
    z = _robust_z(s)
    assert z.iloc[-1] == z.max()      # the 10 is the biggest deviation
    assert z.iloc[0] < z.iloc[-1]


def test_robust_z_constant_series_is_zero():
    z = _robust_z(pd.Series([5.0, 5.0, 5.0]))
    assert (z == 0).all()
