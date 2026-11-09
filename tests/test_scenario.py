"""Welten und Daten: Handwerte der Formeln, Bestimmtheit durch den Seed, Unabhängigkeit der Ströme (nvd_scenario.py)."""

import numpy as np
import pytest

import nvd_scenario as S
from nvd_rng import SplitMix64

Z95 = 1.6448536269514722   # 0,95-Quantil der Standardnormalverteilung


def test_mean_and_spread_hand_values():
    X = np.array([[-2.0], [0.0], [1.0]])
    lin = S.World("linear", 1, (1.0,))
    assert S.mean_of(lin, X) == pytest.approx([40.0, 60.0, 70.0])                          # 60 + 10 x
    assert S.spread_of(lin, X) == pytest.approx([8.0, 8.0, 8.0])
    assert S.spread_of(S.World("streuung-linear", 1, (1.0,)), X) == pytest.approx([1.0, 6.0, 10.0])   # max(1, 6 + 4 x): bei x = -2 greift der Mindestwert 1
    assert S.spread_of(S.World("betrag", 1, (1.0,)), X) == pytest.approx([16.0, 4.0, 10.0])           # 4 + 6 |x|
    assert S.mean_of(S.World("quadratisch", 1, (1.0,)), np.array([[2.0]])) == pytest.approx([60 + 20 + 8 * 3])   # 60 + 10 x + 8 (x^2 - 1)
    assert S.mean_of(S.World("betrag", 1, (1.0,)), X) == pytest.approx([40.0, 60.0, 70.0])           # nur die Streuung ist anders


def test_true_quantile_hand_values():
    X = np.array([[0.0], [-7.0]])
    lin = S.World("linear", 1, (1.0,))
    # x = 0: mu 60, sigma 8: Median 60, 0,95-Quantil 60 + 8 * 1,6449
    assert S.true_quantile(lin, X[:1], 0.5) == pytest.approx([60.0])
    assert S.true_quantile(lin, X[:1], 0.95) == pytest.approx([60.0 + 8.0 * Z95])
    # x = -7: mu = -10; Median der gestutzten Nachfrage 0, 0,95-Quantil -10 + 8 * 1,6449 = 3,159
    assert S.true_quantile(lin, X[1:], 0.5) == pytest.approx([0.0])
    assert S.true_quantile(lin, X[1:], 0.95) == pytest.approx([-10.0 + 8.0 * Z95])


def test_critical_costs():
    assert S.critical_costs(0.8) == pytest.approx((4.0, 1.0))
    assert S.critical_costs(0.5) == pytest.approx((1.0, 1.0))
    assert S.critical_costs(0.95) == pytest.approx((19.0, 1.0))
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            S.critical_costs(bad)


def test_world_is_a_unit_vector_and_depends_only_on_the_seed():
    for p in (1, 3, 10):
        w = S.make_world("linear", p, 5)
        assert np.linalg.norm(w.beta) == pytest.approx(1.0) and len(w.beta) == p
    assert S.make_world("linear", 3, 5) == S.make_world("linear", 3, 5)
    assert S.make_world("linear", 3, 5) != S.make_world("linear", 3, 6)
    assert S.make_world("linear", 3, 5).beta == S.make_world("quadratisch", 3, 5).beta   # die Richtung hängt nicht von der Art ab
    assert abs(S.make_world("linear", 1, 5).beta[0]) == pytest.approx(1.0)


def test_world_validation():
    with pytest.raises(ValueError):
        S.make_world("unbekannt", 3, 0)
    with pytest.raises(ValueError):
        S.make_world("linear", 0, 0)


def test_dataset_is_deterministic_and_streams_are_independent():
    a, b = S.make_dataset("streuung-linear", 3, 50, 3), S.make_dataset("streuung-linear", 3, 50, 3)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y) and np.array_equal(a.Xt, b.Xt)
    bigger = S.make_dataset("streuung-linear", 3, 80, 3)
    assert bigger.world == a.world and np.array_equal(bigger.Xt, a.Xt)               # Welt und Testsatz hängen nicht von n ab
    assert not np.array_equal(S.make_dataset("streuung-linear", 3, 50, 4).X, a.X)
    assert a.X.shape == (50, 3) and a.y.shape == (50,) and a.Xt.shape == (S.N_TEST, 3)
    assert np.all(a.y >= 0.0)


def test_demand_is_cut_at_zero_in_the_far_tail():
    world = S.World("linear", 1, (1.0,))
    X = np.full((2000, 1), -9.0)                                                      # mu = -30, sigma = 8: fast jede Ziehung ist negativ
    d = S.draw_demand(world, X, SplitMix64(1))
    assert d.min() == 0.0 and np.mean(d == 0.0) > 0.99
