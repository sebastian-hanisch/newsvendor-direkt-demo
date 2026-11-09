"""Kern: Bestellregeln für den Newsvendor mit Merkmalen (nur numpy/scipy).

Alle Regeln sind linear in den Merkmalen, q(x) = c0 + x . c, und unterscheiden sich nur darin, wie c gelernt wird:

* `mittel`:  Kleinste Quadrate (KQ), die Prognose des Mittelwerts wird bestellt.
* `emp`:     KQ plus empirisches Quantil der Residuen (Rang ceil(tau * n), unabhängig von x): zweistufig, Prognose plus Reststreuung.
* `direkt`:  lineare Quantilregression zum Niveau tau. Ihr Verlust ist der Pinball-Verlust, und der ist bis auf den Faktor (b + h) der Newsvendor-Kosten
             selbst: b * (D - q)+ + h * (q - D)+ = (b + h) * pinball_tau(D - q) mit tau = b / (b + h). Die Regel lernt also direkt auf den Kosten.
* `direkt_l1`: wie `direkt`, mit L1-Strafe lam * |c|_1 (ohne den Achsenabschnitt) auf den mittleren Pinball-Verlust.

Die Quantilregression ist ein lineares Programm, gelöst mit `scipy.optimize.linprog` (HiGHS, dünne Matrizen). Jede Regel ist eine eigene Funktion mit
expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b)."""

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

LAMS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5)   # Raster der Strafstärken für die Kreuzvalidierung und den Regler
N_FOLDS = 5
RULES = ("mittel", "emp", "direkt", "direkt_l1")


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((len(X), 1)), X])


def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Kleinste Quadrate mit Achsenabschnitt; Koeffizienten (c0, c1, ..., cp). Bei Rangmangel die Lösung kleinster Norm."""
    return np.linalg.lstsq(add_intercept(X), y, rcond=None)[0]


def predict(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    return coef[0] + X @ coef[1:]


def pinball_loss(q: np.ndarray, y: np.ndarray, tau: float) -> float:
    """Mittlerer Pinball-Verlust zum Niveau tau: tau * (y - q)+ + (1 - tau) * (q - y)+."""
    r = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    return float(np.mean(np.maximum(tau * r, (tau - 1.0) * r)))


def newsvendor_cost(q: np.ndarray, y: np.ndarray, b: float, h: float) -> float:
    """Mittlere Kosten b * (y - q)+ + h * (q - y)+ der Bestellmengen q gegen die Nachfragen y."""
    q = np.asarray(q, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean(b * np.maximum(y - q, 0.0) + h * np.maximum(q - y, 0.0)))


def empirical_quantile(values: np.ndarray, tau: float) -> float:
    """Das ceil(tau * n)-kleinste Element (Rangregel, Quantil der empirischen Verteilung)."""
    v = np.sort(np.asarray(values, dtype=float))
    if len(v) == 0:
        raise ValueError("leere Stichprobe")
    rank = min(max(int(np.ceil(tau * len(v) - 1e-12)), 1), len(v))
    return float(v[rank - 1])


def quantreg_fit(X: np.ndarray, y: np.ndarray, tau: float, lam: float = 0.0) -> np.ndarray:
    """Lineare Quantilregression zum Niveau tau mit optionaler L1-Strafe lam (nicht auf den Achsenabschnitt).

    LP: min  lam * sum(c+ + c-) + (tau * sum u + (1 - tau) * sum v) / n   mit  y - c0 - X (c+ - c-) = u - v,  u, v, c+, c- >= 0.
    Rückgabe (c0, c1, ..., cp). Bei Mehrdeutigkeit (etwa n <= p ohne Strafe) liefert HiGHS eine von mehreren gleich guten Lösungen."""
    if not 0.0 < tau < 1.0:
        raise ValueError("tau muss zwischen 0 und 1 liegen (ohne die Grenzen)")
    if lam < 0:
        raise ValueError("lam darf nicht negativ sein")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    cost = np.concatenate([[0.0], lam * np.ones(2 * p), tau * np.ones(n) / n, (1.0 - tau) * np.ones(n) / n])
    eye = sparse.identity(n, format="csr")
    A_eq = sparse.hstack([sparse.csr_matrix(np.ones((n, 1))), sparse.csr_matrix(X), sparse.csr_matrix(-X), eye, -eye], format="csr")
    bounds = [(None, None)] + [(0, None)] * (2 * p + 2 * n)
    res = linprog(cost, A_eq=A_eq, b_eq=y, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"Quantilregression: LP nicht gelöst ({res.message})")
    x = res.x
    return np.concatenate([[x[0]], x[1:1 + p] - x[1 + p:1 + 2 * p]])


def cv_choose_lambda(X: np.ndarray, y: np.ndarray, tau: float, lams=LAMS, folds: int = N_FOLDS) -> tuple[float, list[float]]:
    """Strafstärke per k-facher Kreuzvalidierung auf dem Pinball-Verlust (Faltung = Zeilenindex modulo folds, deterministisch).

    Rückgabe (beste Stärke, Kreuzvalidierungsverlust je Stärke); bei Gleichstand (Unterschied unter 1e-9) gewinnt die kleinere Stärke."""
    n = len(y)
    fold = np.arange(n) % folds
    curve = []
    for lam in lams:
        total = 0.0
        for f in range(folds):
            train, valid = fold != f, fold == f
            coef = quantreg_fit(X[train], y[train], tau, lam)
            total += pinball_loss(predict(coef, X[valid]), y[valid], tau) * int(valid.sum())
        curve.append(total / n)
    best = next(i for i, v in enumerate(curve) if v <= min(curve) + 1e-9)
    return float(lams[best]), curve


def fit_rule(rule: str, X: np.ndarray, y: np.ndarray, tau: float, lam: float = 0.0) -> np.ndarray:
    """Koeffizienten (c0, ..., cp) der Regel; `direkt_l1` nutzt die gegebene Strafstärke lam."""
    if rule == "mittel":
        return ols_fit(X, y)
    if rule == "emp":
        coef = ols_fit(X, y)
        shifted = coef.copy()
        shifted[0] += empirical_quantile(y - predict(coef, X), tau)
        return shifted
    if rule == "direkt":
        return quantreg_fit(X, y, tau, 0.0)
    if rule == "direkt_l1":
        return quantreg_fit(X, y, tau, lam)
    raise ValueError(f"unbekannte Regel: {rule}")
