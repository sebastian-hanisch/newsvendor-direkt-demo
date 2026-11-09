"""Permalink (Begrenzen und Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Die Regler sind Auswahlregler mit festen Stufen: ein Wert aus der Adresszeile rastet auf die nächste erlaubte Stufe ein,
ein unbekannter Name einer Welt wird ignoriert."""

import random
from dataclasses import dataclass

import streamlit as st

import nvd_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None


SETTING_SPECS = {
    "welt_select": SettingSpec("welt", str, "streuung-linear", options=C.WELTEN),
    "tau_select": SettingSpec("tau", float, C.DEFAULT_TAU, options=C.TAU_OPTIONS),
    "n_select": SettingSpec("n", int, C.DEFAULT_N, options=C.N_OPTIONS),
    "p_select": SettingSpec("p", int, C.DEFAULT_P, options=C.P_OPTIONS),
    "lam_select": SettingSpec("lam", float, C.DEFAULT_LAM, options=C.LAM_OPTIONS),
    "nstrafe_select": SettingSpec("ns", int, C.DEFAULT_N_STRAFE, options=C.N_STRAFE_OPTIONS),
    "tauhand_select": SettingSpec("th", float, C.DEFAULT_TAU_HAND, options=C.TAU_OPTIONS),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"welt": "welt_select", "tau": "tau_select", "n": "n_select", "p": "p_select", "lam": "lam_select", "n_strafe": "nstrafe_select",
               "tau_hand": "tauhand_select", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def options_of(state_key):
    return SETTING_SPECS[state_key].options


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def snap(spec: SettingSpec, value):
    """Auf die erlaubte Stufe einrasten (Zahlen: nächste Stufe; Namen: nur gültige Namen) bzw. auf die Grenzen begrenzen; None, wenn der Wert unbrauchbar ist."""
    if spec.options is not None:
        if spec.caster is str:
            return value if value in spec.options else None
        return min(spec.options, key=lambda o: abs(o - value))
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf Stufen eingerastet bzw. auf die Grenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = snap(spec, spec.caster(qp[spec.url_param]))
            except (ValueError, TypeError):
                continue
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem
    Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS, dann hebt auch seine Änderung die Markierung auf
    (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden; die Regler bleiben danach frei.")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
