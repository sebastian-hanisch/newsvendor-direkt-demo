"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle und jede Zahl im Text steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Felder von Mehrkosten, keine Seeds."""

import numpy as np
import pytest

import nvd_evaluation as ev


def _band(direct, emp, mittel=None):
    n = len(direct)
    return {"mittel": np.full(n, 290.0) if mittel is None else np.asarray(mittel), "emp": np.asarray(emp, dtype=float), "direkt": np.asarray(direct, dtype=float)}


def test_compare_rules_thresholds_are_two_thirds_and_one_third():
    emp = np.full(30, 10.0)
    assert ev.compare_rules(np.r_[np.full(20, 5.0), np.full(10, 20.0)], emp) == "direkt"      # 20 von 30: genau zwei Drittel
    assert ev.compare_rules(np.r_[np.full(19, 5.0), np.full(11, 20.0)], emp) == "gleich"      # 19 von 30
    assert ev.compare_rules(np.r_[np.full(11, 5.0), np.full(19, 20.0)], emp) == "gleich"      # 11 von 30
    assert ev.compare_rules(np.r_[np.full(10, 5.0), np.full(20, 20.0)], emp) == "emp"         # 10 von 30: genau ein Drittel
    assert ev.compare_rules(np.full(30, 10.0), emp) == "emp"                                  # Gleichstand zählt nicht als besser


def test_verdict_streuung_success_names_medians_and_count():
    direct = np.r_[np.full(27, 10.0), np.full(3, 50.0)]
    emp = np.full(30, 30.0)
    kind, text = ev.verdict_streuung(_band(direct, emp), "streuung-linear", 3, 100, 0.95)
    assert kind == "success"
    assert "Welt „Streuung wächst linear mit x₁“, p = 3, n = 100, τ = 0,95, 30 Seeds" in text
    assert "im Median 30,0 % (KQ plus Quantil) gegen 10,0 % (direkt)" in text and "direkt ist in 27 von 30 Seeds besser" in text
    assert "lohnt sich das direkte Lernen" in text


def test_verdict_streuung_warning_when_two_stage_is_better():
    direct = np.r_[np.full(4, 1.0), np.full(26, 9.0)]
    emp = np.full(30, 3.0)
    kind, text = ev.verdict_streuung(_band(direct, emp), "linear", 3, 1000, 0.95)
    assert kind == "warning" and "9,0 % (direkt)" in text and "3,0 % (KQ plus Quantil)" in text and "in 4 von 30 Seeds besser" in text and "nicht" in text


def test_verdict_streuung_info_when_mixed():
    direct = np.r_[np.full(13, 1.0), np.full(17, 9.0)]
    kind, text = ev.verdict_streuung(_band(direct, np.full(30, 3.0)), "linear", 10, 100, 0.8)
    assert kind == "info" and "in 13 von 30 Seeds besser" in text and "Uneinheitlich" in text and "τ = 0,8" in text


def test_verdict_hand_names_all_three_costs_and_the_caveat():
    rules = ev.hand_rules(0.8)
    kind, text = ev.verdict_hand(rules, 0.8)
    assert kind == "success"
    assert "4,17 (Mittel bestellen), 3,50 (KQ plus Quantil)" in text and "**2,50** (direkt)" in text and "Trainingswerte mit n = 6, kein Testwert" in text


def test_verdict_hand_without_advantage_is_info():
    rules = {"mittel": {"train_cost": 3.0}, "emp": {"train_cost": 2.0}, "direkt": {"train_cost": 2.0}}
    kind, text = ev.verdict_hand(rules, 0.5)
    assert kind == "info" and "kein Vorteil für direkt" in text and "τ = 0,5 " in text


def test_hand_example_at_tau_one_half_all_three_rules_agree():
    """tau = 0,5 (b = h = 1): die Gruppenmediane sind 5 und 12; KQ liefert dieselben Mengen; das Residuum der Rangzahl ceil(0,5 * 6) = 3 ist 0 (aufsteigend -4, -1, 0, 0, 1, 4).
    Trainingskosten von Hand: x = 0 -> 1 + 0 + 1 = 2, x = 1 -> 4 + 0 + 4 = 8, Summe 10, Mittel 10 / 6."""
    r = ev.hand_rules(0.5)
    for rule in ("mittel", "emp", "direkt"):
        assert r[rule]["q0"] == pytest.approx(5.0) and r[rule]["q1"] == pytest.approx(12.0), rule
        assert r[rule]["train_cost"] == pytest.approx(10.0 / 6.0), rule
    kind, text = ev.verdict_hand(r, 0.5)
    assert kind == "info" and "kein Vorteil für direkt" in text and "τ = 0,5 " in text


def test_tau_text_and_number_format():
    assert ev.tau_text(0.8) == "0,8" and ev.tau_text(0.95) == "0,95" and ev.tau_text(0.5) == "0,5" and ev.tau_text(0.9) == "0,9"
    assert ev.de(3.14159, 2) == "3,14" and ev.pct(38.04) == "38,0 %"


def test_verdict_grid_counts_cells_and_names_first_sample_size():
    g = {"ns": (30, 100, 300), "ps": (1, 10), "m": 30,
         "diff": np.array([[-5.0, -6.0, -7.0], [50.0, 2.0, -3.0]]), "wins": np.zeros((2, 3), dtype=int)}
    kind, text = ev.verdict_grid(g, "streuung-linear", 0.95)
    assert kind == "success"
    assert "in 4 von 6 Zellen weniger Mehrkosten" in text and "p = 1: ab n = 30" in text and "p = 10: ab n = 300" in text and "Median über 30 Seeds" in text


def test_verdict_grid_kinds():
    base = {"ns": (30, 100), "ps": (1, 3), "m": 30, "wins": np.zeros((2, 2), dtype=int)}
    assert ev.verdict_grid({**base, "diff": np.array([[1.0, 2.0], [3.0, 4.0]])}, "linear", 0.95)[0] == "warning"        # nirgends besser
    kind, text = ev.verdict_grid({**base, "diff": np.array([[1.0, 2.0], [3.0, -4.0]])}, "linear", 0.95)                 # 1 von 4: weniger als die Hälfte
    assert kind == "info" and "in 1 von 4 Zellen" in text and "p = 1: in keiner Datenmenge" in text
    assert ev.verdict_grid({**base, "diff": np.array([[-1.0, 2.0], [3.0, -4.0]])}, "linear", 0.95)[0] == "success"      # 2 von 4: die Hälfte reicht


def _penalty(direkt, cv, fest, emp, lam_cv=0.05):
    m = 30
    return {"direkt": np.full(m, direkt), "direkt_cv": np.full(m, cv), "direkt_l1": np.full(m, fest), "emp": np.full(m, emp), "lam_cv": np.full(m, lam_cv)}


def test_verdict_penalty_branches():
    kind, text = ev.verdict_penalty(_penalty(33.0, 45.0, 60.0, 12.0), "linear", 10, 100, 0.95, 0.02)
    assert kind == "warning" and "ohne Strafe 33,0 %" in text and "mit Stärke 0,02 60,0 %" in text and "gewählter Stärke 45,0 %" in text and "KQ plus Quantil 12,0 %" in text
    assert "verschlechtert" in text and "Deutung, nicht gemessen" in text
    kind, text = ev.verdict_penalty(_penalty(200.0, 60.0, 70.0, 50.0), "linear", 10, 30, 0.95, 0.05)
    assert kind == "info" and "erreicht KQ plus Quantil aber nicht" in text
    kind, text = ev.verdict_penalty(_penalty(200.0, 40.0, 70.0, 50.0), "linear", 10, 30, 0.95, 0.05)
    assert kind == "success" and "holt die direkte Regel KQ plus Quantil ein" in text
    assert ev.verdict_penalty(_penalty(33.0, 33.0, 60.0, 12.0), "linear", 10, 100, 0.95, 0.02)[0] == "info"    # gleich ohne Strafe: keine Verschlechterung


def _wrong(emp_q, dir_q, emp_b, dir_b):
    return {"quadratisch": {"emp": np.full(30, emp_q), "direkt": np.full(30, dir_q)}, "betrag": {"emp": np.full(30, emp_b), "direkt": np.full(30, dir_b)}}


def test_verdict_wrong_class_close_and_apart():
    kind, text = ev.verdict_wrong_class(_wrong(141.0, 143.0, 19.5, 20.0), 3, 1000, 0.95)
    assert kind == "info" and "141,0 % (KQ plus Quantil) gegen 143,0 % (direkt)" in text and "19,5 %" in text and "p = 3, n = 1000, τ = 0,95" in text
    assert "repariert die falsche Klasse nicht" in text
    kind, text = ev.verdict_wrong_class(_wrong(141.0, 143.0, 24.0, 30.0), 3, 100, 0.95)      # 6 von 24: mehr als ein Zehntel
    assert kind == "info" and "mehr als ein Zehntel" in text and "repariert" not in text
    assert "repariert" in ev.verdict_wrong_class(_wrong(100.0, 110.0, 20.0, 22.0), 3, 100, 0.95)[1]            # genau ein Zehntel zählt als nah
    assert "mehr als ein Zehntel" in ev.verdict_wrong_class(_wrong(100.0, 110.5, 20.0, 22.0), 3, 100, 0.95)[1]
