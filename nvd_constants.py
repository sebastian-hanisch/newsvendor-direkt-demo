"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

WELTEN = ("linear", "streuung-linear", "betrag", "quadratisch")
WELT_NAMEN = {
    "linear": "Wahrheit linear, Streuung konstant",
    "streuung-linear": "Streuung wächst linear mit x₁",
    "betrag": "Streuung hängt von |x₁| ab (nicht monoton)",
    "quadratisch": "Wahrheit quadratisch in x₁",
}
TAU_OPTIONS = (0.5, 0.8, 0.9, 0.95)
DEFAULT_TAU = 0.95
N_OPTIONS = (30, 100, 300, 1000)
DEFAULT_N = 100
P_OPTIONS = (1, 3, 5, 10)
DEFAULT_P = 3
LAM_OPTIONS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5)   # Strafstärken (dasselbe Raster wie die Kreuzvalidierung, nvd_algorithm.LAMS)
DEFAULT_LAM = 0.02
N_STRAFE_OPTIONS = (30, 100, 300)                 # Datenmengen des Strafe-Experiments (die Kreuzvalidierung rechnet 30 Lösungen je Seed)
DEFAULT_N_STRAFE = 100
SEED_MAX, DEFAULT_SEED = 999999, 7
SWEEP_SEEDS = tuple(range(30))                    # feste Wiederholungen für alle Bänder: nie ein einzelner Seed als Befund
GRID_N = (30, 100, 300, 1000)                     # Raster des Experiments „Datenmenge und Merkmalszahl“
GRID_P = (1, 3, 10)

# Handbeispiel (Plan, von Hand gerechnet): ein Merkmal x in {0, 1}; x = 0: Nachfragen 4, 5, 6; x = 1: 8, 12, 16
HAND_X = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
HAND_Y = (4.0, 5.0, 6.0, 8.0, 12.0, 16.0)
DEFAULT_TAU_HAND = 0.8

COLOR_MEAN = "#9a9a9a"
COLOR_EMP = "#4c78a8"
COLOR_DIRECT = "#2F6B65"
COLOR_L1 = "#e08a1e"
COLOR_TRUTH = "#555555"
RULE_NAMES = {
    "mittel": "Mittel bestellen",
    "emp": "KQ + Quantil der Residuen",
    "direkt": "direkt (Quantilregression)",
    "direkt_l1": "direkt mit L1-Strafe",
}
RULE_COLORS = {"mittel": COLOR_MEAN, "emp": COLOR_EMP, "direkt": COLOR_DIRECT, "direkt_l1": COLOR_L1}

PRESETS = {
    "Streuung wächst mit x₁ (direkt gewinnt)": {"welt": "streuung-linear", "tau": 0.95, "n": 100, "p": 3, "lam": 0.02, "n_strafe": 100, "tau_hand": DEFAULT_TAU_HAND, "seed": DEFAULT_SEED},
    "Konstante Streuung (direkt gewinnt nicht)": {"welt": "linear", "tau": 0.95, "n": 1000, "p": 3, "lam": 0.02, "n_strafe": 100, "tau_hand": DEFAULT_TAU_HAND, "seed": DEFAULT_SEED},
    "Wenig Daten, viele Merkmale": {"welt": "linear", "tau": 0.95, "n": 30, "p": 10, "lam": 0.05, "n_strafe": 30, "tau_hand": DEFAULT_TAU_HAND, "seed": DEFAULT_SEED},
    "Falsche Klasse (quadratische Wahrheit)": {"welt": "quadratisch", "tau": 0.95, "n": 1000, "p": 3, "lam": 0.02, "n_strafe": 100, "tau_hand": DEFAULT_TAU_HAND, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Streuung wächst mit x₁ (direkt gewinnt)": "Welt „Streuung wächst linear mit x₁“, 3 Merkmale, 100 Beobachtungen, τ = 0,95 (kritisches Verhältnis).",
    "Konstante Streuung (direkt gewinnt nicht)": "Welt „Wahrheit linear, Streuung konstant“, 3 Merkmale, 1000 Beobachtungen, τ = 0,95.",
    "Wenig Daten, viele Merkmale": "Welt „Wahrheit linear, Streuung konstant“, 10 Merkmale, nur 30 Beobachtungen, τ = 0,95; das Strafe-Experiment läuft mit n = 30 und λ = 0,05.",
    "Falsche Klasse (quadratische Wahrheit)": "Welt „Wahrheit quadratisch in x₁“, 3 Merkmale, 1000 Beobachtungen, τ = 0,95; beide Regeln sind linear.",
}
