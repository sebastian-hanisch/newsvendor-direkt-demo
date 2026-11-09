"""Kern gegen Handrechnung (Handbeispiel aus tests/conftest.py) und Eigenschaften der Regeln (DEMO-PLAYBOOK §1b und §12).

Der unabhängige Rechenweg (Aufzählung, Monte Carlo) steht in tests/test_oracle_newsvendor_direkt.py."""

import numpy as np
import pytest

import nvd_algorithm as A


def test_hand_calculation_least_squares(hand):
    X, y, tau = hand
    coef = A.ols_fit(X, y)
    assert coef == pytest.approx([5.0, 7.0])
    assert np.allclose(y - A.predict(coef, X), [-1.0, 0.0, 1.0, -4.0, 0.0, 4.0])
    # Mittel bestellen: Mengen 5 und 12; Kosten von Hand: x = 0 -> 1 + 0 + 4, x = 1 -> 4 + 0 + 16, Summe 25, Mittel 25 / 6
    assert A.newsvendor_cost(A.predict(coef, X), y, 4.0, 1.0) == pytest.approx(25.0 / 6.0)


def test_hand_calculation_least_squares_plus_empirical_quantile(hand):
    X, y, tau = hand
    coef = A.fit_rule("emp", X, y, tau)
    assert coef == pytest.approx([6.0, 7.0])                      # Aufschlag 1 = das Residuum der Rangzahl ceil(0,8 * 6) = 5
    assert np.allclose(A.predict(coef, X), [6, 6, 6, 13, 13, 13])
    # von Hand: x = 0 -> 2 + 1 + 0 = 3, x = 1 -> 5 + 1 + 4 * 3 = 18, Summe 21, Mittel 3,5
    assert A.newsvendor_cost(A.predict(coef, X), y, 4.0, 1.0) == pytest.approx(3.5)


def test_hand_calculation_direct_quantile_regression(hand):
    X, y, tau = hand
    coef = A.quantreg_fit(X, y, tau)
    assert coef == pytest.approx([6.0, 10.0], abs=1e-7)
    q = A.predict(coef, X)
    assert np.allclose(q, [6, 6, 6, 16, 16, 16])
    # von Hand: x = 0 -> 2 + 1 + 0 = 3, x = 1 -> 8 + 4 + 0 = 12, Summe 15, Mittel 2,5; Pinball-Summe 15 / (b + h) = 3,0
    assert A.newsvendor_cost(q, y, 4.0, 1.0) == pytest.approx(2.5)
    assert A.pinball_loss(q, y, tau) * 6 == pytest.approx(3.0)


def test_cost_is_b_plus_h_times_pinball_loss():
    """b * (D - q)+ + h * (q - D)+ = (b + h) * pinball_tau(D - q) mit tau = b / (b + h)."""
    rng = np.random.default_rng(3)
    for b, h in ((4.0, 1.0), (1.0, 1.0), (19.0, 1.0), (2.0, 5.0)):
        y, q = rng.uniform(0, 20, 50), rng.uniform(0, 20, 50)
        assert A.newsvendor_cost(q, y, b, h) == pytest.approx((b + h) * A.pinball_loss(q, y, b / (b + h)))


def test_empirical_quantile_uses_the_rank_rule():
    v = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    assert A.empirical_quantile(v, 0.8) == 4.0     # ceil(0,8 * 5) = 4, das vierte Element
    assert A.empirical_quantile(v, 0.81) == 5.0    # ceil(4,05) = 5
    assert A.empirical_quantile(v, 0.2) == 1.0     # ceil(1,0) = 1 (Rundungsrest darf nicht auf 2 springen)
    assert A.empirical_quantile(v, 0.01) == 1.0


def test_pinball_loss_hand_values():
    assert A.pinball_loss(np.array([5.0]), np.array([9.0]), 0.8) == pytest.approx(0.8 * 4.0)    # Fehlmenge 4, Gewicht tau
    assert A.pinball_loss(np.array([9.0]), np.array([5.0]), 0.8) == pytest.approx(0.2 * 4.0)    # Überschuss 4, Gewicht 1 - tau


def test_each_rule_is_a_function_with_a_defined_result(hand):
    X, y, tau = hand
    for rule in A.RULES:
        assert A.fit_rule(rule, X, y, tau, 0.02).shape == (2,)
    with pytest.raises(ValueError):
        A.fit_rule("unbekannt", X, y, tau)


def test_cross_validation_returns_one_loss_per_strength_and_the_smaller_on_a_tie():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 2))
    y = 60 + 10 * X[:, 0] + 4 * rng.standard_normal(40)
    lam, curve = A.cv_choose_lambda(X, y, 0.8)
    assert len(curve) == len(A.LAMS) and lam in A.LAMS and curve[A.LAMS.index(lam)] == min(curve)
    # Gleichstand: identische Verluste für alle Stärken (konstante Daten, alle Koeffizienten null) -> die kleinste Stärke
    lam0, curve0 = A.cv_choose_lambda(np.zeros((20, 1)), np.full(20, 7.0), 0.8)
    assert lam0 == A.LAMS[0] and max(curve0) - min(curve0) < 1e-9


def test_cross_validation_folds_are_deterministic():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 3))
    y = 50 + 5 * X[:, 0] + rng.standard_normal(30)
    assert A.cv_choose_lambda(X, y, 0.9) == A.cv_choose_lambda(X, y, 0.9)


def test_direct_never_costs_more_than_two_stage_on_the_training_data():
    """Beide Regeln liegen in derselben linearen Klasse, und das LP findet darin das Minimum: auf den Trainingsdaten hat direkt nie höhere Kosten als KQ + Quantil oder Mittel bestellen."""
    rng = np.random.default_rng(5)
    for trial in range(40):
        n, p = int(rng.integers(6, 40)), int(rng.integers(1, 5))
        X = rng.standard_normal((n, p))
        y = np.maximum(0, 60 + 10 * X[:, 0] + (3 + 2 * np.abs(X[:, 0])) * rng.standard_normal(n))
        tau = float(rng.choice([0.5, 0.8, 0.95]))
        d = A.pinball_loss(A.predict(A.quantreg_fit(X, y, tau), X), y, tau)
        for rule in ("emp", "mittel"):
            assert d <= A.pinball_loss(A.predict(A.fit_rule(rule, X, y, tau), X), y, tau) + 1e-9, (trial, rule)


def test_training_pinball_does_not_fall_when_the_penalty_grows():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((50, 4))
    y = 40 + 8 * X[:, 0] + 5 * rng.standard_normal(50)
    losses = [A.pinball_loss(A.predict(A.quantreg_fit(X, y, 0.8, lam), X), y, 0.8) for lam in A.LAMS]
    assert all(b >= a - 1e-9 for a, b in zip(losses, losses[1:]))


def test_penalty_shrinks_the_l1_norm():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((50, 4))
    y = 40 + 8 * X[:, 0] + 5 * rng.standard_normal(50)
    norms = [float(np.abs(A.quantreg_fit(X, y, 0.8, lam)[1:]).sum()) for lam in A.LAMS]
    assert all(b <= a + 1e-9 for a, b in zip(norms, norms[1:]))
    assert norms[-1] < norms[0]


def test_shifting_all_demands_shifts_the_intercept_only():
    rng = np.random.default_rng(8)
    X = rng.standard_normal((30, 2))
    y = 50 + 6 * X[:, 0] + 4 * rng.standard_normal(30)
    base = A.quantreg_fit(X, y, 0.8)
    shifted = A.quantreg_fit(X, y + 17.0, 0.8)
    assert shifted[0] == pytest.approx(base[0] + 17.0, abs=1e-6) and shifted[1:] == pytest.approx(base[1:], abs=1e-6)
    assert A.fit_rule("emp", X, y + 17.0, 0.8)[0] == pytest.approx(A.fit_rule("emp", X, y, 0.8)[0] + 17.0)


def test_scaling_the_demands_scales_all_coefficients():
    rng = np.random.default_rng(9)
    X = rng.standard_normal((30, 2))
    y = 50 + 6 * X[:, 0] + 4 * rng.standard_normal(30)
    base = A.quantreg_fit(X, y, 0.8)
    assert A.quantreg_fit(X, 3.0 * y, 0.8) == pytest.approx(3.0 * base, abs=1e-5)


def test_the_order_of_the_observations_does_not_matter():
    rng = np.random.default_rng(10)
    X = rng.standard_normal((30, 2))
    y = 50 + 6 * X[:, 0] + 4 * rng.standard_normal(30)
    perm = rng.permutation(30)
    for rule in ("mittel", "emp", "direkt"):
        assert A.fit_rule(rule, X[perm], y[perm], 0.9) == pytest.approx(A.fit_rule(rule, X, y, 0.9), abs=1e-6)


def test_tau_one_half_is_the_median_regression():
    """Bei tau = 0,5 ist der Pinball-Verlust die Hälfte des mittleren Betragsfehlers; kein anderer Koeffizient (hier: zufällige Gegenkandidaten) liefert einen kleineren Betragsfehler."""
    rng = np.random.default_rng(11)
    X = rng.standard_normal((40, 2))
    y = 30 + 4 * X[:, 0] + rng.standard_t(3, 40)
    coef = A.quantreg_fit(X, y, 0.5)
    best = float(np.mean(np.abs(y - A.predict(coef, X))))
    assert A.pinball_loss(A.predict(coef, X), y, 0.5) == pytest.approx(0.5 * best)
    for _ in range(200):
        other = coef + rng.standard_normal(3) * rng.choice([0.01, 0.1, 1.0])
        assert float(np.mean(np.abs(y - A.predict(other, X)))) >= best - 1e-9
