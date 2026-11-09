"""Welten und Daten: Newsvendor mit Merkmalen (nur numpy/scipy).

Nachfrage D = max(0, Y) mit Y ~ Normal(mu(x), sigma(x)^2), Merkmale x ~ Normal(0, I). Vier Welten:

* `linear`:          mu = 60 + 10 * (x . beta), sigma = 8 (konstant); die Wahrheit liegt in der linearen Klasse.
* `streuung-linear`: mu wie oben, sigma = max(1, 6 + 4 * x1): die Streuung wächst linear mit dem ersten Merkmal.
* `betrag`:          mu wie oben, sigma = 4 + 6 * |x1|: die Streuung hängt von x1 ab, aber nicht monoton (eine lineare Regel trifft sie nicht).
* `quadratisch`:     mu = 60 + 10 * (x . beta) + 8 * (x1^2 - 1), sigma = 8: die Wahrheit liegt nicht in der linearen Klasse.

beta ist ein zufälliger Einheitsvektor. Das wahre tau-Quantil der Nachfrage ist max(0, mu + sigma * z_tau) und damit die optimale Bestellmenge (Orakel).
Die Kosten einer Bestellregel werden **exakt** als Erwartung über die Nachfrage gerechnet (geschlossene Formel für die gestutzte Normalverteilung) und über
einen festen Testsatz von Merkmalen gemittelt; es gibt keine Stichprobe der Nachfrage im Test und damit kein Monte-Carlo-Rauschen in den Mehrkosten.
Bestellmengen unter 0 werden bei der Bewertung auf 0 gesetzt (man kann nichts Negatives bestellen).

Zufall nur über SplitMix64 (`nvd_rng`); jede Größe hat ihren eigenen Strom (Rolle im oberen Teil des Seeds), damit sich Welt, Daten und Testsatz nicht verschieben,
wenn eine andere Größe geändert wird."""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri

from nvd_rng import SplitMix64

WORLD_KINDS = ("linear", "streuung-linear", "betrag", "quadratisch")
N_TEST = 2000          # Merkmalspunkte des festen Testsatzes je Seed
_ROLE_WORLD, _ROLE_TRAIN, _ROLE_TEST = 1 << 32, 2 << 32, 3 << 32


@dataclass(frozen=True)
class World:
    kind: str
    p: int
    beta: tuple[float, ...]


@dataclass(frozen=True)
class Dataset:
    world: World
    X: np.ndarray    # (n, p) Trainingsmerkmale
    y: np.ndarray    # (n,) beobachtete Nachfragen
    Xt: np.ndarray   # (N_TEST, p) Testmerkmale


def make_world(kind: str, p: int, seed: int) -> World:
    if kind not in WORLD_KINDS:
        raise ValueError(f"unbekannte Welt: {kind}")
    if p < 1:
        raise ValueError("p muss mindestens 1 sein")
    beta = SplitMix64(seed + _ROLE_WORLD).uniform(p) * 2.0 - 1.0
    norm = float(np.linalg.norm(beta))
    beta = beta / norm if norm > 0 else np.eye(p)[0]
    return World(kind, p, tuple(float(b) for b in beta))


def mean_of(world: World, X: np.ndarray) -> np.ndarray:
    """mu(x) der Normalverteilung vor der Stutzung bei 0."""
    m = 60.0 + 10.0 * (X @ np.asarray(world.beta))
    if world.kind == "quadratisch":
        m = m + 8.0 * (X[:, 0] ** 2 - 1.0)
    return m


def spread_of(world: World, X: np.ndarray) -> np.ndarray:
    """sigma(x) der Normalverteilung vor der Stutzung bei 0."""
    if world.kind == "streuung-linear":
        return np.maximum(1.0, 6.0 + 4.0 * X[:, 0])
    if world.kind == "betrag":
        return 4.0 + 6.0 * np.abs(X[:, 0])
    return np.full(len(X), 8.0)


def draw_features(n: int, p: int, rng: SplitMix64) -> np.ndarray:
    return rng.normal(n * p).reshape(n, p)


def draw_demand(world: World, X: np.ndarray, rng: SplitMix64) -> np.ndarray:
    return np.maximum(0.0, mean_of(world, X) + spread_of(world, X) * rng.normal(len(X)))


def make_dataset(kind: str, p: int, n: int, seed: int) -> Dataset:
    """Deterministisch aus (Welt, p, n, Seed). Die Welt hängt nicht von n ab, der Testsatz nicht von n und nicht von den Trainingsdaten."""
    world = make_world(kind, p, seed)
    rng = SplitMix64(seed + _ROLE_TRAIN)
    X = draw_features(n, p, rng)
    y = draw_demand(world, X, rng)
    Xt = draw_features(N_TEST, p, SplitMix64(seed + _ROLE_TEST))
    return Dataset(world, X, y, Xt)


def critical_costs(tau: float) -> tuple[float, float]:
    """Kosten (b, h) je fehlende und je überschüssige Einheit zum kritischen Verhältnis tau = b / (b + h), mit h = 1 festgelegt."""
    if not 0.0 < tau < 1.0:
        raise ValueError("tau muss zwischen 0 und 1 liegen (ohne die Grenzen)")
    return tau / (1.0 - tau), 1.0


def true_quantile(world: World, X: np.ndarray, tau: float) -> np.ndarray:
    """Optimale Bestellmenge: das tau-Quantil von D = max(0, Y), also max(0, mu + sigma * z_tau)."""
    return np.maximum(0.0, mean_of(world, X) + spread_of(world, X) * float(ndtri(tau)))


def expected_cost_at(world: World, X: np.ndarray, q: np.ndarray, b: float, h: float) -> np.ndarray:
    """Erwartete Kosten je Merkmalspunkt: b * E[(D - q)+] + h * E[(q - D)+] für D = max(0, Y), exakt.

    Für q >= 0 gilt (D - q)+ = (Y - q)+ und (q - D)+ = (q - Y)+ - (-Y)+; die Erwartungen der Normalverteilung sind
    E[(Y - q)+] = s * phi(z) - (q - m) * (1 - Phi(z)) und E[(q - Y)+] = s * phi(z) + (q - m) * Phi(z) mit z = (q - m) / s."""
    q = np.maximum(np.asarray(q, dtype=float), 0.0)
    m, s = mean_of(world, X), spread_of(world, X)

    def phi(z):
        return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)

    z = (q - m) / s
    under = s * phi(z) - (q - m) * (1.0 - ndtr(z))      # E[(Y - q)+]  = E[(D - q)+]
    over_y = s * phi(z) + (q - m) * ndtr(z)             # E[(q - Y)+]
    z0 = -m / s
    neg = s * phi(z0) - m * ndtr(z0)                    # E[(0 - Y)+]  = E[(-Y)+]
    return b * under + h * (over_y - neg)


def expected_cost(world: World, X: np.ndarray, q: np.ndarray, b: float, h: float) -> float:
    """Mittel der erwarteten Kosten über die Merkmalspunkte."""
    return float(np.mean(expected_cost_at(world, X, q, b, h)))


def extra_cost_percent(world: World, X: np.ndarray, q: np.ndarray, tau: float) -> float:
    """Mehrkosten der Regel q gegenüber dem Orakel (wahres Quantil) in Prozent der Orakelkosten."""
    b, h = critical_costs(tau)
    oracle = expected_cost(world, X, true_quantile(world, X, tau), b, h)
    return 100.0 * (expected_cost(world, X, q, b, h) - oracle) / oracle
