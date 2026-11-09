"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Button, fährt jeden Regler an
seine Grenzen und prüft, dass kein Python-Fehler auftritt und jeder Abschnitt sein Verdikt zeigt.

Die Regler der App sind Auswahlregler (`select_slider`, `selectbox`) und ein Zahlenfeld; die allgemeinen stehen in der Seitenleiste, die der Experimente 1 und 4 im Hauptbereich."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import nvd_constants as C
import nvd_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _verdicts(at) -> list[str]:
    return [e.value for kind in ("success", "info", "warning") for e in getattr(at, kind)]


def test_app_loads_without_exception_and_shows_every_verdict():
    at = _fresh_app()
    verdicts = _verdicts(at)
    for marker in ("Trainingswerte mit n = 6", "Seeds: Mehrkosten gegen das Orakel", "Zellen weniger Mehrkosten", "Kreuzvalidierung", "p = 3, n = 100, τ = 0,95, Mehrkosten gegen das Orakel"):
        assert any(marker in v for v in verdicts), marker


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def _control_values(state_key):
    spec = P.SETTING_SPECS[state_key]
    return list(spec.options) if spec.options is not None else [spec.lo, spec.hi]


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_every_control_at_its_levels_does_not_raise(state_key):
    for value in _control_values(state_key):
        at = _fresh_app()
        for art in ("select_slider", "selectbox", "number_input"):
            try:
                widget = getattr(at, art)(key=state_key)
                break
            except KeyError:
                continue
        else:
            raise AssertionError(state_key)
        widget.set_value(value).run(timeout=TIMEOUT)
        assert not at.exception, (state_key, value, [str(e) for e in at.exception])


def test_the_pdf_button_makes_a_download():
    at = _fresh_app()
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_permalink_sets_the_controls_and_snaps_to_levels():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["welt"] = "linear"
    at.query_params["n"] = "250"          # rastet auf 300 ein
    at.query_params["tau"] = "0.8"
    at.query_params["p"] = "10"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.selectbox(key="welt_select").value == "linear" and at.select_slider(key="n_select").value == 300
    assert at.select_slider(key="tau_select").value == 0.8 and at.select_slider(key="p_select").value == 10


def test_permalink_ignores_unusable_values():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["welt"] = "gibt-es-nicht"
    at.query_params["n"] = "abc"
    at.query_params["seed"] = "-4"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.selectbox(key="welt_select").value == C.PRESETS[next(iter(C.PRESETS))]["welt"]
    assert at.select_slider(key="n_select").value == C.DEFAULT_N and at.number_input(key="seed_input").value == 0


def test_headings_name_the_five_experiments():
    at = _fresh_app()
    titles = [s.value for s in at.subheader]
    for n in range(1, 6):
        assert any(f"Experiment {n}" in t for t in titles), n
    assert any("Wo die Annahmen enden" in t for t in titles)
    assert any(e.label == "📐 Mathematische Formulierung" for e in at.expander)
