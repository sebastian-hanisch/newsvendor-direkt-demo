"""Messung hinter den Zahlen der README (30 feste Seeds, Mehrkosten gegen das Orakel in Prozent, Median und [Minimum, Maximum]).

Aufruf aus dem Repo-Ordner:  python tools/messung.py > tools/messung_ausgabe.txt
Rechnet mit denselben Funktionen wie die App (nvd_evaluation, nvd_algorithm, nvd_scenario) und einem Prozess; Laufzeit nie als feste Sekundenzahl angeben.
Die Ausgabe `tools/messung_ausgabe.txt` ist die Quelle der Zahlen in der README; `tests/test_claims.py` prüft eine Auswahl davon als Bänder."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nvd_algorithm as A  # noqa: E402
import nvd_constants as C  # noqa: E402
import nvd_evaluation as ev  # noqa: E402
import nvd_scenario as S  # noqa: E402


def fmt(a) -> str:
    return f"{np.median(a):6.1f} [{np.min(a):5.1f},{np.max(a):6.1f}]"


def row(kind, p, n, tau):
    b = ev.band(kind, p, n, tau)
    print(f"  {kind:16s} p = {p:>2} n = {n:>4} tau = {tau:<4}: Mittel bestellen {fmt(b['mittel'])} | KQ+Quantil {fmt(b['emp'])} | direkt {fmt(b['direkt'])} | direkt besser in {ev.wins(b['direkt'], b['emp']):>2}/{len(b['emp'])}")


def main():
    print(f"Seeds: {C.SWEEP_SEEDS[0]} bis {C.SWEEP_SEEDS[-1]} ({len(C.SWEEP_SEEDS)}), Testsatz je Seed 2000 Merkmalspunkte, Mehrkosten exakt gegen das wahre Quantil")
    print("== A: Mittel bestellen, Welt linear, p = 3")
    for n, tau in ((1000, 0.5), (1000, 0.8), (1000, 0.95), (100, 0.8), (100, 0.95)):
        row("linear", 3, n, tau)
    print("== B (M1): Streuung linear, p = 3")
    for n, tau in ((100, 0.95), (1000, 0.95), (5000, 0.95), (1000, 0.8)):
        row("streuung-linear", 3, n, tau)
    print("== C (M2): Streuung linear, p = 10")
    for n, tau in ((1000, 0.95), (100, 0.95), (30, 0.95)):
        row("streuung-linear", 10, n, tau)
    print("== D (M3): Welt linear (konstante Streuung)")
    for p, n, tau in ((3, 1000, 0.95), (10, 100, 0.95), (10, 30, 0.95), (10, 30, 0.8)):
        row("linear", p, n, tau)
    print("== E (M4): mit L1-Strafe, p = 10 (lam fest 0,1 und per 5-facher Kreuzvalidierung aus {0, 0,02, 0,05, 0,1, 0,2, 0,5})")
    for kind in ("linear", "streuung-linear"):
        for n in (30, 100):
            for tau in (0.8, 0.95):
                b = ev.penalty_band(kind, 10, n, tau, 0.1)
                print(f"  {kind:16s} n = {n:>3} tau = {tau:<4}: KQ+Quantil {fmt(b['emp'])} | direkt {fmt(b['direkt'])} | lam 0,1 {fmt(b['direkt_l1'])} | direkt+CV {fmt(b['direkt_cv'])} | Median gewähltes lam {np.median(b['lam_cv']):.2f}")
    print("== F (M5): falsche Klasse, p = 3")
    for kind in ("quadratisch", "betrag"):
        row(kind, 3, 1000, 0.95)
    print("== G: Raster des Experiments 3 (Median der Differenz direkt minus KQ+Quantil, Zahl der Seeds mit Vorteil für direkt)")
    for kind in ("streuung-linear", "linear"):
        g = ev.grid_band(kind, 0.95)
        print(f"  Welt {kind}, tau = 0,95; Spalten n = {g['ns']}")
        for i, p in enumerate(g["ps"]):
            print(f"    p = {p:>2}: " + "  ".join(f"{g['diff'][i, j]:7.1f} ({g['wins'][i, j]:>2})" for j in range(len(g["ns"]))))
    print("== H: KQ + Quantil mit Rangregel (App) gegen linear interpoliertes Quantil (Vormessung), tau = 0,95")
    for kind, p, n in (("linear", 10, 30), ("streuung-linear", 10, 30), ("linear", 10, 100), ("streuung-linear", 3, 100), ("linear", 3, 1000)):
        rank = ev.band(kind, p, n, 0.95, ("emp",))["emp"]
        print(f"  {kind:16s} p = {p:>2} n = {n:>4}: Rangregel {fmt(rank)} | interpoliert {fmt(interpolated_emp(kind, p, n, 0.95))}")


def interpolated_emp(kind, p, n, tau):
    """KQ plus Quantil mit dem linear interpolierten Quantil der Residuen (numpy-Standard) statt der Rangregel: die Fassung der Vormessung des Plans."""
    out = []
    for seed in C.SWEEP_SEEDS:
        d = S.make_dataset(kind, p, n, seed)
        coef = A.ols_fit(d.X, d.y)
        shifted = coef.copy()
        shifted[0] += float(np.quantile(d.y - A.predict(coef, d.X), tau))
        out.append(S.extra_cost_percent(d.world, d.Xt, A.predict(shifted, d.Xt), tau))
    return np.array(out)


if __name__ == "__main__":
    main()
