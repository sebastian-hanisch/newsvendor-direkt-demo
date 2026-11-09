"""Kennzahlen, Bänder über feste Seeds und Verdikte (reine Funktionen). Nur numpy/scipy.

Alle Zahlen sind Mehrkosten gegen das Orakel (das wahre Quantil) in Prozent der Orakelkosten, im Median über die festen Seeds `SWEEP_SEEDS`; ein einzelner Seed ist nie ein
Befund. Verdikte sind reine Funktionen `verdict_*(...) -> (Art, Text)`, die App ruft `getattr(st, Art)(Text)`."""

from dataclasses import dataclass

import numpy as np

import nvd_algorithm as A
import nvd_constants as C
import nvd_scenario as S


@dataclass(frozen=True)
class Settings:
    welt: str = "streuung-linear"
    tau: float = C.DEFAULT_TAU
    n: int = C.DEFAULT_N
    p: int = C.DEFAULT_P
    lam: float = C.DEFAULT_LAM
    n_strafe: int = C.DEFAULT_N_STRAFE
    tau_hand: float = C.DEFAULT_TAU_HAND
    seed: int = C.DEFAULT_SEED


def de(x: float, digits: int = 1) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float) -> str:
    return de(x, 1) + " %"


def tau_text(tau: float) -> str:
    return de(tau, 2) if round(tau * 100) % 10 else de(tau, 1)


# --- Handbeispiel (Experiment 1) -------------------------------------------------------------------------------------------------------

def hand_rules(tau: float) -> dict:
    """Die drei Regeln auf dem Handbeispiel (6 Beobachtungen, ein Merkmal x in {0, 1}) mit Koeffizienten, Bestellmengen und Trainingskosten (h = 1)."""
    X = np.array(C.HAND_X).reshape(-1, 1)
    y = np.array(C.HAND_Y)
    b, h = S.critical_costs(tau)
    out = {}
    for rule in ("mittel", "emp", "direkt"):
        coef = A.fit_rule(rule, X, y, tau)
        q = A.predict(coef, X)
        out[rule] = {
            "intercept": float(coef[0]), "slope": float(coef[1]),
            "q0": float(coef[0]), "q1": float(coef[0] + coef[1]),
            "train_cost": A.newsvendor_cost(q, y, b, h),
            "pinball_sum": A.pinball_loss(q, y, tau) * len(y),
        }
    return out


def hand_cost_curve(tau: float, qs: np.ndarray) -> dict:
    """Summe der Kosten über die drei Beobachtungen je Gruppe als Funktion der Bestellmenge: Gruppe x = 0 und x = 1."""
    b, h = S.critical_costs(tau)
    y = np.array(C.HAND_Y)
    x = np.array(C.HAND_X)
    out = {}
    for g in (0.0, 1.0):
        yg = y[x == g]
        out[g] = np.array([float(np.sum(b * np.maximum(yg - q, 0.0) + h * np.maximum(q - yg, 0.0))) for q in qs])
    return out


# --- Bänder über feste Seeds -----------------------------------------------------------------------------------------------------------

def summarize(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"median": float(np.median(v)), "low": float(v.min()), "high": float(v.max())}


def seed_extras(kind: str, p: int, n: int, tau: float, seed: int, rules=("mittel", "emp", "direkt")) -> dict:
    """Mehrkosten in Prozent je Regel für einen Seed (Welt, Daten und Testsatz aus dem Seed)."""
    d = S.make_dataset(kind, p, n, seed)
    return {r: S.extra_cost_percent(d.world, d.Xt, A.predict(A.fit_rule(r, d.X, d.y, tau), d.Xt), tau) for r in rules}


def band(kind: str, p: int, n: int, tau: float, rules=("mittel", "emp", "direkt"), seeds=C.SWEEP_SEEDS) -> dict:
    """{Regel: Feld der Mehrkosten über die Seeds}, dazu die Zahl der Seeds, in denen `direkt` besser ist als `emp`."""
    rows = [seed_extras(kind, p, n, tau, s, rules) for s in seeds]
    out = {r: np.array([row[r] for row in rows]) for r in rules}
    return out


def wins(direct: np.ndarray, emp: np.ndarray) -> int:
    """Zahl der Seeds, in denen die direkte Regel strikt weniger Mehrkosten hat als KQ plus Quantil."""
    return int(np.sum(np.asarray(direct) < np.asarray(emp)))


def penalty_band(kind: str, p: int, n: int, tau: float, lam: float, seeds=C.SWEEP_SEEDS) -> dict:
    """Experiment 4: `emp`, `direkt` (ohne Strafe), `direkt_l1` mit fester Stärke lam und `direkt_cv` mit per Kreuzvalidierung gewählter Stärke; dazu die gewählten Stärken."""
    cols = {"emp": [], "direkt": [], "direkt_l1": [], "direkt_cv": [], "lam_cv": []}
    for s in seeds:
        d = S.make_dataset(kind, p, n, s)
        lam_cv, _ = A.cv_choose_lambda(d.X, d.y, tau)
        for rule, coef in (("emp", A.fit_rule("emp", d.X, d.y, tau)), ("direkt", A.fit_rule("direkt", d.X, d.y, tau)),
                           ("direkt_l1", A.fit_rule("direkt_l1", d.X, d.y, tau, lam)), ("direkt_cv", A.quantreg_fit(d.X, d.y, tau, lam_cv))):
            cols[rule].append(S.extra_cost_percent(d.world, d.Xt, A.predict(coef, d.Xt), tau))
        cols["lam_cv"].append(lam_cv)
    return {k: np.array(v) for k, v in cols.items()}


def grid_band(kind: str, tau: float, ns=C.GRID_N, ps=C.GRID_P, seeds=C.SWEEP_SEEDS) -> dict:
    """Experiment 3: je Zelle (p, n) der Median der Differenz „Mehrkosten direkt minus Mehrkosten KQ + Quantil“ (je Seed gepaart) und die Zahl der Seeds, in denen direkt besser ist."""
    diff, win = np.zeros((len(ps), len(ns))), np.zeros((len(ps), len(ns)), dtype=int)
    for i, p in enumerate(ps):
        for j, n in enumerate(ns):
            b = band(kind, p, n, tau, ("emp", "direkt"), seeds)
            diff[i, j] = float(np.median(b["direkt"] - b["emp"]))
            win[i, j] = wins(b["direkt"], b["emp"])
    return {"ns": tuple(ns), "ps": tuple(ps), "diff": diff, "wins": win, "m": len(seeds)}


def single_run(settings: Settings) -> dict:
    """Ein Lauf mit dem Zufalls-Seed der Instanz: Bestellmengen auf dem Testsatz gegen das wahre Quantil, für die Abbildung."""
    d = S.make_dataset(settings.welt, settings.p, settings.n, settings.seed)
    q_true = S.true_quantile(d.world, d.Xt, settings.tau)
    out = {"x1": d.Xt[:, 0], "q_true": q_true, "extra": {}, "q": {}}
    for rule in ("emp", "direkt"):
        q = A.predict(A.fit_rule(rule, d.X, d.y, settings.tau), d.Xt)
        out["q"][rule] = q
        out["extra"][rule] = S.extra_cost_percent(d.world, d.Xt, q, settings.tau)
    return out


# --- Verdikte -------------------------------------------------------------------------------------------------------------------------------

def compare_rules(direct: np.ndarray, emp: np.ndarray) -> str:
    """„direkt“ bei mindestens zwei Dritteln der Seeds besser, „emp“ bei höchstens einem Drittel, sonst „gleich“ (uneinheitlich)."""
    m = len(direct)
    w = wins(direct, emp)
    if 3 * w >= 2 * m:
        return "direkt"
    if 3 * w <= m:
        return "emp"
    return "gleich"


def verdict_hand(rules: dict, tau: float) -> tuple[str, str]:
    """Verdikt Experiment 1 (Trainingswerte mit n = 6, kein Testwert)."""
    m, e, d = rules["mittel"]["train_cost"], rules["emp"]["train_cost"], rules["direkt"]["train_cost"]
    if d < e - 1e-9:
        return "success", (f"Bei τ = {tau_text(tau)} kosten die Bestellmengen auf den 6 Beobachtungen im Mittel {de(m, 2)} (Mittel bestellen), {de(e, 2)} (KQ plus Quantil) und "
                           f"**{de(d, 2)}** (direkt). Das sind Trainingswerte mit n = 6, kein Testwert: Der Vorsprung kann an der Anpassung auf nur sechs Beobachtungen liegen.")
    return "info", (f"Bei τ = {tau_text(tau)} kosten die Bestellmengen auf den 6 Beobachtungen im Mittel {de(m, 2)} (Mittel bestellen), {de(e, 2)} (KQ plus Quantil) und "
                    f"{de(d, 2)} (direkt): kein Vorteil für direkt auf diesen Trainingsdaten. Trainingswerte mit n = 6, kein Testwert.")


def verdict_streuung(extra: dict, kind: str, p: int, n: int, tau: float) -> tuple[str, str]:
    """Verdikt Experiment 2: Mehrkosten von KQ plus Quantil gegen direkt im Median über die Seeds, mit Zählung."""
    m = len(extra["direkt"])
    emp, direct = float(np.median(extra["emp"])), float(np.median(extra["direkt"]))
    w = wins(extra["direkt"], extra["emp"])
    head = (f"Welt „{C.WELT_NAMEN[kind]}“, p = {p}, n = {n}, τ = {tau_text(tau)}, {m} Seeds: Mehrkosten gegen das Orakel im Median {pct(emp)} (KQ plus Quantil) gegen {pct(direct)} (direkt); "
            f"direkt ist in {w} von {m} Seeds besser.")
    which = compare_rules(extra["direkt"], extra["emp"])
    if which == "direkt":
        return "success", head + " In dieser Welt lohnt sich das direkte Lernen."
    if which == "emp":
        return "warning", head + " Hier lohnt sich das direkte Lernen nicht: KQ plus Quantil ist bei höchstens einem Drittel der Seeds schlechter."
    return "info", head + " Uneinheitlich: weder in mindestens zwei Dritteln noch in höchstens einem Drittel der Seeds besser."


def verdict_grid(g: dict, kind: str, tau: float) -> tuple[str, str]:
    """Verdikt Experiment 3: in wie vielen Zellen direkt im Median besser ist, und je Merkmalszahl die kleinste Datenmenge dafür."""
    better = g["diff"] < 0
    total = better.size
    parts = []
    for i, p in enumerate(g["ps"]):
        first = next((g["ns"][j] for j in range(len(g["ns"])) if better[i, j]), None)
        parts.append(f"p = {p}: " + (f"ab n = {first}" if first is not None else "in keiner Datenmenge"))
    text = (f"Welt „{C.WELT_NAMEN[kind]}“, τ = {tau_text(tau)}: direkt hat im Median über {g['m']} Seeds in {int(better.sum())} von {total} Zellen weniger Mehrkosten als KQ plus Quantil "
            f"(erste Datenmenge mit Vorteil je Merkmalszahl: " + "; ".join(parts) + ").")
    return ("success" if better.sum() * 2 >= total and better.any() else "info" if better.any() else "warning"), text


def verdict_penalty(b: dict, kind: str, p: int, n: int, tau: float, lam: float) -> tuple[str, str]:
    """Verdikt Experiment 4: die Strafe mit fester Stärke und mit Kreuzvalidierung gegen ohne Strafe und gegen KQ plus Quantil."""
    med = {k: float(np.median(b[k])) for k in ("emp", "direkt", "direkt_l1", "direkt_cv")}
    m = len(b["emp"])
    text = (f"Welt „{C.WELT_NAMEN[kind]}“, p = {p}, n = {n}, τ = {tau_text(tau)}, {m} Seeds, Mehrkosten im Median: ohne Strafe {pct(med['direkt'])}, mit Stärke {de(lam, 2)} {pct(med['direkt_l1'])}, "
            f"mit per Kreuzvalidierung gewählter Stärke {pct(med['direkt_cv'])}; KQ plus Quantil {pct(med['emp'])}.")
    if med["direkt_cv"] > med["direkt"]:
        return "warning", text + " Die Kreuzvalidierung verschlechtert hier gegenüber ohne Strafe (Deutung, nicht gemessen: bei so wenig Daten ist die Wahl der Strafe selbst verrauscht)."
    if med["direkt_cv"] <= med["emp"]:
        return "success", text + " Mit der gewählten Strafe holt die direkte Regel KQ plus Quantil ein oder schlägt sie."
    return "info", text + " Die Strafe hilft, erreicht KQ plus Quantil aber nicht."


def verdict_wrong_class(rows: dict, p: int, n: int, tau: float) -> tuple[str, str]:
    """Verdikt Experiment 5 (Welten „quadratisch“ und „betrag“): liegen die Mehrkosten beider Regeln in jeder Welt nah beieinander (Unterschied höchstens ein Zehntel des kleineren Werts),
    repariert der Verlust die falsche Klasse nicht; sonst wird der Unterschied benannt."""
    parts, close = [], True
    for kind, b in rows.items():
        emp, direct = float(np.median(b["emp"])), float(np.median(b["direkt"]))
        close = close and abs(emp - direct) <= 0.1 * min(emp, direct)
        parts.append(f"„{C.WELT_NAMEN[kind]}“: {pct(emp)} (KQ plus Quantil) gegen {pct(direct)} (direkt), direkt besser in {wins(b['direkt'], b['emp'])} von {len(b['emp'])} Seeds")
    head = f"p = {p}, n = {n}, τ = {tau_text(tau)}, Mehrkosten gegen das Orakel im Median: " + "; ".join(parts) + "."
    if close:
        return "info", head + " Beide Regeln sind linear und liegen in beiden Welten dicht beieinander: der Verlust allein repariert die falsche Klasse nicht."
    return "info", head + " In mindestens einer Welt unterscheiden sich die beiden Regeln um mehr als ein Zehntel des kleineren Werts."
