"""Randfälle der Kernroutinen (DEMO-PLAYBOOK §12): n < p + 1, gleiche x-Werte, gleiche Nachfragen, tau nahe 1 bei kleinem n, n = 1 und ungültige Eingaben."""

import numpy as np
import pytest

import nvd_algorithm as A
import nvd_evaluation as ev


def test_fewer_observations_than_coefficients_interpolates_without_penalty():
    """n = 3 < p + 1 = 6: das LP passt alle Punkte exakt (Pinball-Verlust 0); eine Strafe verhindert das."""
    rng = np.random.default_rng(30)
    X = rng.standard_normal((3, 5))
    y = np.array([50.0, 60.0, 71.0])
    coef = A.quantreg_fit(X, y, 0.8)
    assert A.pinball_loss(A.predict(coef, X), y, 0.8) == pytest.approx(0.0, abs=1e-8)
    assert A.pinball_loss(A.predict(A.quantreg_fit(X, y, 0.8, 0.5), X), y, 0.8) > 0.1
    assert np.allclose(A.predict(A.ols_fit(X, y), X), y)       # KQ: Lösung kleinster Norm, passt ebenfalls exakt


def test_n_equals_p_plus_one_fits_all_points_exactly():
    rng = np.random.default_rng(31)
    X = rng.standard_normal((4, 3))
    y = rng.uniform(30, 70, 4)
    assert A.pinball_loss(A.predict(A.quantreg_fit(X, y, 0.95), X), y, 0.95) == pytest.approx(0.0, abs=1e-8)


def test_identical_feature_rows_give_the_empirical_quantile():
    """Alle x gleich: KQ plus Quantil bestellt für alle Merkmalswerte dasselbe Quantil der Nachfragen; direkt hat denselben Pinball-Verlust wie die Rangregel."""
    y = np.array([12.0, 7.0, 9.0, 15.0, 3.0, 8.0, 10.0, 11.0, 6.0, 14.0])
    X = np.full((10, 2), 0.5)
    for tau in (0.5, 0.8, 0.95):
        q_emp = A.predict(A.fit_rule("emp", X, y, tau), X)
        q_dir = A.predict(A.fit_rule("direkt", X, y, tau), X)
        assert q_emp == pytest.approx(np.full(10, A.empirical_quantile(y, tau)))
        assert A.pinball_loss(q_dir, y, tau) == pytest.approx(A.pinball_loss(q_emp, y, tau), abs=1e-9)


def test_all_demands_equal_every_rule_orders_that_value():
    X = np.random.default_rng(32).standard_normal((12, 2))
    y = np.full(12, 23.0)
    for rule in ("mittel", "emp", "direkt", "direkt_l1"):
        assert A.predict(A.fit_rule(rule, X, y, 0.9, 0.05), X) == pytest.approx(np.full(12, 23.0), abs=1e-7), rule


def test_tau_close_to_one_with_fewer_than_twenty_observations_orders_the_maximum():
    """n = 10 < 1 / (1 - 0,95): das Quantil der Residuen ist das größte Residuum (Rang 10); ohne Merkmale bestellt direkt das Maximum der Nachfragen."""
    y = np.array([12.0, 7.0, 9.0, 15.0, 3.0, 8.0, 10.0, 11.0, 6.0, 14.0])
    assert A.empirical_quantile(y, 0.95) == 15.0
    X0 = np.zeros((10, 1))
    assert A.predict(A.quantreg_fit(X0, y, 0.95), X0) == pytest.approx(np.full(10, 15.0), abs=1e-7)
    X = np.random.default_rng(33).standard_normal((10, 3))
    for rule in A.RULES:
        assert np.all(np.isfinite(A.fit_rule(rule, X, y, 0.95, 0.02)))


def test_single_observation():
    X, y = np.array([[0.7]]), np.array([20.0])
    assert A.predict(A.quantreg_fit(X, y, 0.8), X) == pytest.approx([20.0], abs=1e-7)
    assert A.predict(A.fit_rule("emp", X, y, 0.8), X) == pytest.approx([20.0])
    assert A.empirical_quantile(y, 0.5) == 20.0


def test_invalid_inputs_raise():
    X, y = np.zeros((4, 1)), np.arange(4.0)
    for tau in (0.0, 1.0, -0.2, 1.3):
        with pytest.raises(ValueError):
            A.quantreg_fit(X, y, tau)
    with pytest.raises(ValueError):
        A.quantreg_fit(X, y, 0.8, -0.1)
    with pytest.raises(ValueError):
        A.empirical_quantile(np.array([]), 0.8)


def test_tau_close_to_the_bounds_still_solves():
    X, y = np.random.default_rng(34).standard_normal((10, 1)), np.arange(10.0)
    for tau in (0.01, 0.99):
        assert np.all(np.isfinite(A.quantreg_fit(X, y, tau)))


def test_bands_with_a_single_seed_and_the_win_count():
    b = ev.band("linear", 1, 30, 0.8, seeds=(0,))
    assert all(len(v) == 1 for v in b.values())
    assert ev.wins(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 1.0])) == 1       # strikt kleiner: nur die erste Stelle
