"""Orakel-Tests: Quantilregression, Newsvendor-Kosten und Welt gegen unabhängige Rechenwege (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §5).

1. LP gegen **Aufzählung** aller Ecken (Geraden durch Paare von Datenpunkten, bei Strafe zusätzlich waagerechte Geraden durch einen Punkt), p = 1, kleines n.
2. LP gegen `sklearn.linear_model.QuantileRegressor` (Zielfunktion: mittlerer Pinball-Verlust plus alpha mal L1-Norm; eigene Matrixaufstellung), falls scikit-learn im Test-venv liegt.
3. Erwartete Kosten (geschlossene Formel der gestutzten Normalverteilung) gegen **Monte Carlo** mit numpy `default_rng`; wahres Quantil gegen das empirische Quantil von Ziehungen.
4. Das Orakel ist optimal: kein anderer Bestellwert hat kleinere erwartete Kosten.
5. SplitMix64 gegen eine zweite, reine Python-Implementierung mit ganzen Zahlen und gegen den veröffentlichten Referenzwert für Seed 0.
Der große Lauf (200 Instanzen) lief im Scratch-Ordner; hier steht die kleine, schnelle Fassung."""

import numpy as np
import pytest

import nvd_algorithm as A
import nvd_scenario as S
from nvd_rng import SplitMix64


def _objective(x, y, tau, lam, c0, c):
    r = y - c0 - c * x
    return float(np.mean(np.maximum(tau * r, (tau - 1.0) * r)) + lam * abs(c))


def _enumerated_minimum(x, y, tau, lam):
    """Kleinster Zielwert über alle Ecken der Anordnung der Geraden c0 + x_i c = y_i (und c = 0): die Zielfunktion ist konvex und stückweise linear."""
    n = len(x)
    candidates = [(float(y[i]), 0.0) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] != x[j]:
                c = (y[i] - y[j]) / (x[i] - x[j])
                candidates.append((float(y[i] - c * x[i]), float(c)))
    return min(_objective(x, y, tau, lam, c0, c) for c0, c in candidates)


def test_lp_matches_vertex_enumeration_on_random_small_instances():
    rng = np.random.default_rng(20)
    for trial in range(60):
        n = int(rng.integers(4, 10))
        x = np.round(rng.standard_normal(n), 2) if trial % 3 else rng.integers(0, 3, n).astype(float)   # jede dritte Instanz mit vielen gleichen x-Werten
        y = np.round(50 + 6 * x + 5 * rng.standard_normal(n), 1)
        tau = float(rng.choice([0.5, 0.8, 0.95]))
        lam = float(rng.choice([0.0, 0.0, 0.05, 0.3]))
        coef = A.quantreg_fit(x.reshape(-1, 1), y, tau, lam)
        assert _objective(x, y, tau, lam, coef[0], coef[1]) == pytest.approx(_enumerated_minimum(x, y, tau, lam), abs=1e-7), (trial, tau, lam)


def test_the_enumeration_oracle_reproduces_the_hand_example():
    x = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    y = np.array([4, 5, 6, 8, 12, 16], dtype=float)
    assert _enumerated_minimum(x, y, 0.8, 0.0) * 6 == pytest.approx(3.0)       # Pinball-Summe 3,0 von Hand (Gegenprobe über alle Geraden durch Datenpunktpaare)


def test_lp_matches_scikit_learn_quantile_regressor():
    linear_model = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(21)
    for trial in range(15):
        n, p = int(rng.integers(15, 40)), int(rng.integers(1, 5))
        X = rng.standard_normal((n, p))
        y = 50 + 6 * X[:, 0] + (3 + 2 * np.abs(X[:, 0])) * rng.standard_normal(n)
        tau, lam = float(rng.choice([0.5, 0.8, 0.95])), float(rng.choice([0.0, 0.02, 0.1]))
        coef = A.quantreg_fit(X, y, tau, lam)
        ref = linear_model.QuantileRegressor(quantile=tau, alpha=lam, solver="highs").fit(X, y)

        def value(c0, c):
            return A.pinball_loss(c0 + X @ c, y, tau) + lam * float(np.abs(c).sum())

        assert value(coef[0], coef[1:]) == pytest.approx(value(float(ref.intercept_), np.asarray(ref.coef_)), abs=1e-7), trial


def _truncated_world():
    """Ein Merkmal, mu = 60 + 10 x, sigma = 8; die Merkmalswerte -6, -7, -5 geben mu = 0, -10, 10: die Stutzung bei 0 greift stark."""
    return S.World("linear", 1, (1.0,)), np.array([[-6.0], [-7.0], [-5.0], [0.0], [1.0]])


def test_expected_cost_matches_monte_carlo():
    world, X = _truncated_world()
    z = np.random.default_rng(22).standard_normal(400_000)
    for tau in (0.5, 0.8, 0.95):
        b, h = S.critical_costs(tau)
        for q0 in (0.0, 3.0, 12.0, 70.0):
            q = np.full(len(X), q0)
            exact = S.expected_cost_at(world, X, q, b, h)
            for i in range(len(X)):
                D = np.maximum(0.0, S.mean_of(world, X)[i] + S.spread_of(world, X)[i] * z)
                mc = float(np.mean(b * np.maximum(D - q0, 0.0) + h * np.maximum(q0 - D, 0.0)))
                assert exact[i] == pytest.approx(mc, rel=0.02, abs=0.03), (tau, q0, i)


def test_negative_order_quantities_are_cut_at_zero():
    world, X = _truncated_world()
    assert S.expected_cost_at(world, X, np.full(len(X), -5.0), 4.0, 1.0) == pytest.approx(S.expected_cost_at(world, X, np.zeros(len(X)), 4.0, 1.0))


def test_true_quantile_matches_the_empirical_quantile_of_draws():
    world, X = _truncated_world()
    z = np.random.default_rng(23).standard_normal(400_000)
    for tau in (0.5, 0.8, 0.95):
        q = S.true_quantile(world, X, tau)
        for i in range(len(X)):
            D = np.maximum(0.0, S.mean_of(world, X)[i] + S.spread_of(world, X)[i] * z)
            assert q[i] == pytest.approx(float(np.quantile(D, tau)), abs=0.1), (tau, i)


def test_the_true_quantile_is_the_cost_optimal_order_quantity():
    for kind in S.WORLD_KINDS:
        d = S.make_dataset(kind, 3, 10, 4)
        for tau in (0.5, 0.8, 0.95):
            b, h = S.critical_costs(tau)
            qs = S.true_quantile(d.world, d.Xt, tau)
            best = S.expected_cost_at(d.world, d.Xt, qs, b, h)
            for delta in (-6.0, -1.0, -0.05, 0.05, 1.0, 6.0):
                assert np.all(S.expected_cost_at(d.world, d.Xt, qs + delta, b, h) >= best - 1e-9), (kind, tau, delta)


def test_extra_cost_is_never_negative_and_zero_for_the_oracle():
    d = S.make_dataset("streuung-linear", 3, 50, 1)
    assert S.extra_cost_percent(d.world, d.Xt, S.true_quantile(d.world, d.Xt, 0.8), 0.8) == pytest.approx(0.0, abs=1e-9)
    rng = np.random.default_rng(24)
    for _ in range(20):
        assert S.extra_cost_percent(d.world, d.Xt, rng.uniform(0, 120, len(d.Xt)), 0.8) >= 0.0


def _splitmix_reference(seed, k):
    """Der k-te Ausgabewert (k ab 1) mit ganzen Zahlen von Python, ohne numpy."""
    mask = (1 << 64) - 1
    z = (seed + k * 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return z ^ (z >> 31)


def test_splitmix64_matches_an_integer_reference_and_the_published_value():
    assert int(SplitMix64(0).u64(1)[0]) == 0xE220A8397B1DCDAF            # veröffentlichter erster Ausgabewert für Seed 0
    for seed in (0, 1, 12345, 2 ** 40 + 7, 2 ** 63 + 5):
        assert [int(v) for v in SplitMix64(seed).u64(5)] == [_splitmix_reference(seed, k) for k in range(1, 6)]


def test_splitmix64_stream_advances_and_is_reproducible():
    a = SplitMix64(9)
    first, second = a.u64(3), a.u64(3)
    assert [int(v) for v in np.concatenate([first, second])] == [int(v) for v in SplitMix64(9).u64(6)]
    u = SplitMix64(10).uniform(10_000)
    assert 0.0 <= u.min() and u.max() < 1.0 and abs(u.mean() - 0.5) < 0.02
    z = SplitMix64(11).normal(20_000)
    assert abs(z.mean()) < 0.03 and abs(z.std() - 1.0) < 0.03
