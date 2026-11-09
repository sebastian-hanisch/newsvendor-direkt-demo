"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart und allgemeine Einstellungen in der Seitenleiste, Regler einzelner Experimente im Hauptbereich, aktives Preset hervorgehoben.

Die Hervorhebung ist zustandslos
(`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset); der Test unten stellt sicher, dass wirklich JEDER Regler dort steht."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import nvd_constants as C
import nvd_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


SIDEBAR_REGLER = {"welt_select", "tau_select", "n_select", "p_select", "seed_input"}      # allgemeine Einstellungen
MAIN_REGLER = {"tauhand_select", "lam_select", "nstrafe_select"}                         # Regler einzelner Experimente


def _regler(at, key):
    bereich = at.sidebar if key in SIDEBAR_REGLER else at.main
    for art in REGLER_ARTEN:
        try:
            return getattr(bereich, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht im erwarteten Bereich")


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))  # das Stylesheet setzt an der Klasse des Containers an
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [next(iter(C.PRESETS))]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_setzt_alle_regler(name):
    at = _app()
    at.button(key=f"preset_{name}").click().run()
    for key, state_key in P.PRESET_KEYS.items():
        assert _regler(at, state_key).value == C.PRESETS[name][key], (name, key)


def test_allgemeine_regler_stehen_in_der_seitenleiste_experimentregler_im_hauptbereich():
    at = _app()
    seite = {w.key for art in REGLER_ARTEN for w in getattr(at.sidebar, art) if w.key}
    haupt = {w.key for art in REGLER_ARTEN for w in getattr(at.main, art) if w.key}
    assert seite == SIDEBAR_REGLER and haupt == MAIN_REGLER


def test_jeder_regler_steht_in_setting_specs():
    at = _app()
    schluessel = {w.key for bereich in (at.sidebar, at.main) for art in REGLER_ARTEN for w in getattr(bereich, art) if w.key}
    fehlt = schluessel - set(P.SETTING_SPECS)
    assert not fehlt, f"Regler {sorted(fehlt)} fehlen in SETTING_SPECS: ihre Änderung hebt das aktive Preset nicht auf"
    assert set(P.SETTING_SPECS) <= schluessel, "ein Eintrag in SETTING_SPECS hat keinen Regler"


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(state_key):
    at = _app()
    spec = P.SETTING_SPECS[state_key]
    regler = _regler(at, state_key)
    if spec.options is not None:
        neu = spec.options[-1] if regler.value != spec.options[-1] else spec.options[0]
    else:
        neu = spec.hi if regler.value != spec.hi else spec.lo
    regler.set_value(neu).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "primary" not in _typen(at).values()
    # erneutes Anklicken markiert wieder
    name = next(iter(C.PRESETS))
    at.button(key=f"preset_{name}").click().run()
    assert _typen(at)[name] == "primary"
