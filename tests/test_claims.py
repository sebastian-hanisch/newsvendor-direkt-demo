"""Zahlen der README (Abschnitt „Befunde“) an den festen Seeds 0 bis 29, als Bänder (Median ± max(12 % des Werts, 0,5 Prozentpunkte), Zählungen ± 3 Seeds).

Die Quelle der README-Zahlen ist `tools/messung_ausgabe.txt` (erzeugt von `tools/messung.py` mit denselben Funktionen wie die App). Trainierte und per Kreuzvalidierung
gewählte Werte sind plattformabhängig und stehen deshalb nur als Bänder und als Ungleichungen hier. Nicht im Test: die Zeile n = 5000 und Abschnitt H der Messung
(Rangregel gegen interpoliertes Quantil); sie stehen in der README als „Messung dieser README, nicht im Test“."""

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import nvd_evaluation as ev

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=None)
def _band(kind, p, n, tau):
    return ev.band(kind, p, n, tau)


@lru_cache(maxsize=None)
def _penalty(kind, p, n, tau):
    return ev.penalty_band(kind, p, n, tau, 0.1)


@lru_cache(maxsize=None)
def _grid(kind):
    return ev.grid_band(kind, 0.95)


def _near(value, expected):
    return abs(value - expected) <= max(0.12 * expected, 0.5)


def _check(b, expected, wins_expected=None):
    for rule, e in expected.items():
        m = float(np.median(b[rule]))
        assert _near(m, e), (rule, m, e)
    if wins_expected is not None:
        assert abs(ev.wins(b["direkt"], b["emp"]) - wins_expected) <= 3, (ev.wins(b["direkt"], b["emp"]), wins_expected)


# Zeilen: (Welt, p, n, tau, {Regel: Median in %}, Seeds mit direkt besser)
MITTEL = [
    ("linear", 3, 1000, 0.5, {"mittel": 0.2, "emp": 0.1}, None),
    ("linear", 3, 1000, 0.8, {"mittel": 42.8, "emp": 0.2}, None),
    ("linear", 3, 1000, 0.95, {"mittel": 287.7, "emp": 0.3}, None),
    ("linear", 3, 100, 0.8, {"mittel": 44.2}, None),
    ("linear", 3, 100, 0.95, {"mittel": 290.9}, None),
]
STREUUNG_P3 = [
    ("streuung-linear", 3, 100, 0.95, {"emp": 38.0, "direkt": 13.3}, 27),
    ("streuung-linear", 3, 1000, 0.95, {"emp": 34.5, "direkt": 7.8}, 30),
    ("streuung-linear", 3, 1000, 0.8, {"emp": 13.3, "direkt": 1.9}, 30),
]
STREUUNG_P10 = [
    ("streuung-linear", 10, 1000, 0.95, {"emp": 33.4, "direkt": 9.8}, 30),
    ("streuung-linear", 10, 100, 0.95, {"emp": 41.0, "direkt": 46.6}, 13),
    ("streuung-linear", 10, 30, 0.95, {"emp": 54.7, "direkt": 185.6}, 0),
]
KONSTANT = [
    ("linear", 3, 1000, 0.95, {"emp": 0.3, "direkt": 0.9}, 4),
    ("linear", 10, 100, 0.95, {"emp": 12.7, "direkt": 33.2}, 2),
    ("linear", 10, 30, 0.95, {"emp": 47.3, "direkt": 197.9}, 1),
    ("linear", 10, 30, 0.8, {"emp": 25.7, "direkt": 42.7}, 7),
]
FALSCHE_KLASSE = [
    ("quadratisch", 3, 1000, 0.95, {"mittel": 518.8, "emp": 141.3, "direkt": 143.4}, 5),
    ("betrag", 3, 1000, 0.95, {"mittel": 291.5, "emp": 19.5, "direkt": 20.0}, 7),
]


@pytest.mark.parametrize("row", MITTEL, ids=lambda r: f"{r[0]}-p{r[1]}-n{r[2]}-tau{r[3]}")
def test_claim_mittel_bestellen_kostet_bei_schiefen_kosten_viel(row):
    kind, p, n, tau, expected, w = row
    _check(_band(kind, p, n, tau), expected, w)


def test_claim_mittel_bestellen_haengt_kaum_an_n_und_kostet_bei_tau_05_fast_nichts():
    """README: 44,2 % bei n = 100 gegen 42,8 % bei n = 1000 (τ = 0,8); bei τ = 0,5 nur 0,2 % (Welt linear, p = 3)."""
    m100 = float(np.median(_band("linear", 3, 100, 0.8)["mittel"]))
    m1000 = float(np.median(_band("linear", 3, 1000, 0.8)["mittel"]))
    assert abs(m100 - m1000) < 5.0 and m1000 > 30.0
    assert float(np.median(_band("linear", 3, 1000, 0.5)["mittel"])) < 1.0


@pytest.mark.parametrize("row", STREUUNG_P3, ids=lambda r: f"p3-n{r[2]}-tau{r[3]}")
def test_claim_streuung_linear_p3_direkt_gewinnt(row):
    kind, p, n, tau, expected, w = row
    b = _band(kind, p, n, tau)
    _check(b, expected, w)
    assert np.median(b["direkt"]) < np.median(b["emp"])


@pytest.mark.parametrize("row", STREUUNG_P10, ids=lambda r: f"p10-n{r[2]}-tau{r[3]}")
def test_claim_streuung_linear_p10(row):
    kind, p, n, tau, expected, w = row
    _check(_band(kind, p, n, tau), expected, w)


def test_claim_streuung_linear_p10_vorzeichen_je_datenmenge():
    """README: bei p = 10 gewinnt direkt mit n = 1000 klar (alle 30 Seeds), mit n = 30 verliert es in jedem der 30 Seeds; n = 100 liegt dazwischen."""
    assert ev.wins(_band("streuung-linear", 10, 1000, 0.95)["direkt"], _band("streuung-linear", 10, 1000, 0.95)["emp"]) >= 28
    assert ev.wins(_band("streuung-linear", 10, 30, 0.95)["direkt"], _band("streuung-linear", 10, 30, 0.95)["emp"]) <= 2
    mid = _band("streuung-linear", 10, 100, 0.95)
    assert 5 <= ev.wins(mid["direkt"], mid["emp"]) <= 25


@pytest.mark.parametrize("row", KONSTANT, ids=lambda r: f"{r[0]}-p{r[1]}-n{r[2]}-tau{r[3]}")
def test_claim_konstante_streuung_direkt_gewinnt_nicht(row):
    kind, p, n, tau, expected, w = row
    b = _band(kind, p, n, tau)
    _check(b, expected, w)
    assert np.median(b["direkt"]) > np.median(b["emp"])


@pytest.mark.parametrize("row", FALSCHE_KLASSE, ids=lambda r: r[0])
def test_claim_falsche_klasse(row):
    kind, p, n, tau, expected, w = row
    b = _band(kind, p, n, tau)
    _check(b, expected, w)
    assert abs(np.median(b["direkt"]) - np.median(b["emp"])) <= 0.1 * min(np.median(b["direkt"]), np.median(b["emp"]))   # README: höchstens ein Zehntel des kleineren Werts


STRAFE = [
    ("linear", 30, 0.95, {"emp": 47.3, "direkt": 197.9, "direkt_l1": 63.6, "direkt_cv": 65.7}),
    ("linear", 30, 0.8, {"emp": 25.7, "direkt": 42.7, "direkt_cv": 36.3}),
    ("linear", 100, 0.95, {"emp": 12.7, "direkt": 33.2, "direkt_l1": 61.4, "direkt_cv": 45.3}),
    ("streuung-linear", 30, 0.95, {"emp": 54.7, "direkt": 185.6, "direkt_cv": 106.1}),
    ("streuung-linear", 100, 0.95, {"emp": 41.0, "direkt": 46.6, "direkt_cv": 46.6}),
]


@pytest.mark.parametrize("row", STRAFE, ids=lambda r: f"{r[0]}-n{r[1]}-tau{r[2]}")
def test_claim_strafe_p10(row):
    """Bänder um die README-Zahlen (fester Stärke 0,1; Kreuzvalidierung über 0 / 0,02 / 0,05 / 0,1 / 0,2 / 0,5) und die Ungleichungen, die der Text daraus zieht."""
    kind, n, tau, expected = row
    b = _penalty(kind, 10, n, tau)
    for rule, e in expected.items():
        assert abs(float(np.median(b[rule])) - e) <= max(0.2 * e, 1.0), (rule, float(np.median(b[rule])), e)
    if n == 30:
        assert np.median(b["direkt_cv"]) < np.median(b["direkt"])         # bei kleinem n hilft die Strafe
        assert np.median(b["direkt_cv"]) > np.median(b["emp"])            # erreicht KQ + Quantil nicht
    if n == 100 and kind == "linear":
        assert np.median(b["direkt_cv"]) > np.median(b["direkt"])         # bei n = 100 verschlechtert die Wahl der Strafe (Welt linear)
        assert np.median(b["direkt_l1"]) > np.median(b["direkt"])         # feste Stärke 0,1 ist bei n = 100 zu stark
        assert np.median(b["lam_cv"]) <= 0.05


def test_claim_raster_streuung_linear_und_linear():
    """README: Welt „Streuung wächst linear mit x₁“, τ = 0,95: direkt hat in 10 von 12 Zellen im Median weniger Mehrkosten (nicht bei p = 10 mit n = 30 und n = 100); Welt linear: in keiner der 12 Zellen."""
    g = _grid("streuung-linear")
    assert g["ns"] == (30, 100, 300, 1000) and g["ps"] == (1, 3, 10)
    better = g["diff"] < 0
    assert int(better.sum()) in (9, 10, 11)
    assert better[:, 2:].all()                              # ab n = 300 in jeder Zeile
    assert better[0].all() and better[1].all()             # p = 1 und p = 3 in jeder Spalte
    assert g["diff"][2, 0] > 50.0                          # p = 10, n = 30: deutlich schlechter (README: +115,6 Prozentpunkte)
    assert abs(g["diff"][2, 1]) < 10.0 and 8 <= g["wins"][2, 1] <= 20   # p = 10, n = 100: nahe an der Grenze
    h = _grid("linear")
    assert not (h["diff"] < 0).any()
    assert (h["wins"] <= 12).all()


def test_the_limits_table_in_the_app_is_in_the_readme():
    """Zwei Fassungen desselben Textes: die Zeilen der Tabelle „Wo die Annahmen enden“ stehen in App und README wortgleich."""
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    rows = [line for line in app.splitlines() if line.startswith("| **")]
    assert len(rows) == 5
    for line in rows:
        assert line in readme, line


def test_readme_test_count_matches_the_suite():
    """README nennt die Gesamtzahl der Tests; sie muss mit `pytest --collect-only` übereinstimmen."""
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests", "-p", "no:cacheprovider"], capture_output=True, text=True, cwd=ROOT).stdout
    count = int(re.search(r"(\d+) tests? collected", out).group(1))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{count} Tests" in readme, count
