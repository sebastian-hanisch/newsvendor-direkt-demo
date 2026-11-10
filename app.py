"""Direkt auf den Kosten lernen: Pinball-Verlust und Quantilregression - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Newsvendor mit Merkmalen: die Bestellregel zweistufig (Prognose plus Reststreuung) oder direkt auf den Kosten gelernt (Quantilregression, ein LP).
Schnellstart und allgemeine Einstellungen stehen in der Seitenleiste, experimentbezogene Regler beim Experiment; jedes Experiment rechnet über feste Seeds und zeigt ein Verdikt.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import nvd_constants as C
import nvd_evaluation as ev
from nvd_evaluation import Settings
from nvd_layout import dunkles_design, kopf_css
from nvd_pdf_export import build_pdf
from nvd_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    sidebar_presets,
    options_of,
    randomize_seed,
    sync_query_params,
)
from nvd_visualization import build_band, build_grid, build_hand_curves, build_hand_lines, build_penalty, build_single, build_wrong

st.set_page_config(page_title="Direkt auf den Kosten lernen – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _band(kind, p, n, tau):
    return ev.band(kind, p, n, tau)


@st.cache_data(show_spinner=False)
def _single(settings):
    return ev.single_run(settings)   # settings enthält nur Welt, tau, n, p und Seed (die übrigen Felder bleiben auf dem Standard)


@st.cache_data(show_spinner=False)
def _grid(kind, tau):
    return ev.grid_band(kind, tau)


@st.cache_data(show_spinner=False)
def _penalty(kind, p, n, tau, lam):
    return ev.penalty_band(kind, p, n, tau, lam)


@st.cache_data(show_spinner=False)
def _wrong(p, n, tau):
    return {kind: ev.band(kind, p, n, tau, ("emp", "direkt")) for kind in ("quadratisch", "betrag")}


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in nvd_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🎯 Direkt auf den Kosten lernen: Pinball-Verlust und Quantilregression")
st.markdown(
    """
Beim **Newsvendor** entscheidet eine Bestellmenge zwischen zu viel (Überschuss, Kosten je Einheit **h**) und zu wenig (Fehlmenge, Kosten je Einheit **b**). Der Kostenverlust ist bis auf einen Faktor der
**Pinball-Verlust**; deshalb lässt sich die Bestellregel per **Quantilregression** (ein lineares Programm) **direkt aus den Kosten lernen**, statt erst den Mittelwert zu prognostizieren und eine Reststreuung aufzuschlagen.
Die Demo misst in simulierten Welten, **wann sich das lohnt und wann nicht**: Alle Zahlen sind Mehrkosten gegen das Orakel (die wahre optimale Menge), im Median über 30 feste Zufalls-Seeds.

Weiter unten: fünf Experimente (🔬), der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert das direkte Lernen auf den Kosten", expanded=True):
    st.markdown(
        """
1. **Zweistufig (Prognose plus Reststreuung):** den Mittelwert der Nachfrage per Kleinste Quadrate (KQ) auf die Merkmale zurückführen und die Bestellmenge um das **τ-Quantil der Residuen** erhöhen, für alle Merkmalswerte gleich (τ = b / (b + h), das kritische Verhältnis).
2. **Direkt:** die Bestellmenge **q(x) = c₀ + x·c** so wählen, dass die mittleren Newsvendor-Kosten der Trainingsdaten minimal sind. Diese Kosten sind der Pinball-Verlust (bis auf den Faktor b + h); das Minimum ist die **lineare Quantilregression**, gelöst als lineares Programm.
3. **Bewertung:** beide Regeln werden an einem festen Testsatz gegen das **Orakel** (das wahre τ-Quantil der Nachfrage) gemessen; die **Mehrkosten** sind der Aufschlag in Prozent der Orakelkosten. Das Mittel zu bestellen ist die dritte, ungeschickte Vergleichsregel.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

# --- Schnellstart und allgemeine Einstellungen (Seitenleiste); experimentbezogene Regler stehen beim Experiment ---------------------------------

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Daten und Kosten**")
    welt = st.selectbox("Welt der Daten", options_of("welt_select"), key="welt_select", format_func=lambda k: C.WELT_NAMEN[k],
                        help="Wie die Nachfrage von den Merkmalen abhängt. Gilt für die Experimente 2 bis 4; Experiment 5 hat feste Welten.")
    tau = st.select_slider("Kritisches Verhältnis τ", options_of("tau_select"), key="tau_select", format_func=lambda t: ev.de(t, 2),
                           help="τ = b / (b + h): Anteil der Fehlmengenkosten an der Summe beider Kosten; h = 1 festgelegt. τ = 0,95 heißt: eine fehlende Einheit kostet 19-mal so viel wie eine überschüssige.")
    n = st.select_slider("Beobachtungen n", options_of("n_select"), key="n_select", help="Zahl der Trainingsdaten (Tage mit Merkmalen und beobachteter Nachfrage).")
    p = st.select_slider("Merkmale p", options_of("p_select"), key="p_select", help="Zahl der Merkmale; nur das erste (x₁) beeinflusst in den Welten die Streuung oder die Wahrheit.")
    st.markdown("**Einzellauf**")
    seed = st.number_input("Zufalls-Seed des Einzellaufs", *bounds("seed_input"), key="seed_input", step=1,
                           help="Wählt den einen Datensatz, den der Einzellauf in Experiment 2 zeigt. Alle Bänder rechnen über die 30 festen Seeds und hängen davon nicht ab.")
    st.button("🎲 Neuer Zufalls-Seed", width="stretch", on_click=randomize_seed)

# --- Experiment 1: Pinball von Hand ------------------------------------------------------------------------------------------------------

st.markdown("---")
st.subheader("🔬 Experiment 1: Pinball-Verlust von Hand")
st.caption("Sechs Beobachtungen, ein Merkmal x ∈ {0, 1}: bei x = 0 waren die Nachfragen 4, 5, 6, bei x = 1 waren sie 8, 12, 16; h = 1 und b = τ / (1 − τ). Die Rechnung ist von Hand nachvollziehbar (Voreinstellung τ = 0,8: b = 4).")
st.select_slider("Kritisches Verhältnis τ im Handbeispiel", options_of("tauhand_select"), key="tauhand_select", format_func=lambda t: ev.de(t, 2),
                 help="Nur für dieses Experiment; Voreinstellung 0,8 ist das Handbeispiel mit b = 4, h = 1.")
tau_hand = st.session_state["tauhand_select"]
rules1 = ev.hand_rules(tau_hand)
col_a, col_b = st.columns(2)
col_a.plotly_chart(build_hand_curves(tau_hand, rules1), width="stretch", key="hand_curves")
col_b.plotly_chart(build_hand_lines(rules1), width="stretch", key="hand_lines")
st.caption("Links: Kosten der drei Beobachtungen einer Gruppe über der Bestellmenge (durchgezogen x = 0, gestrichelt x = 1); die Marken sind die Mengen der drei Regeln. "
           "Rechts: die gelernten Geraden. Im Handbeispiel setzt direkt die Menge in jeder Gruppe auf das τ-Quantil der Beobachtungen der Gruppe, KQ plus Quantil verschiebt die Mittel-Gerade nur um einen gemeinsamen Aufschlag.")
st.dataframe(
    [
        {"Regel": C.RULE_NAMES[r], "Achsenabschnitt": round(rules1[r]["intercept"], 2), "Steigung": round(rules1[r]["slope"], 2), "Menge bei x = 0": round(rules1[r]["q0"], 2),
         "Menge bei x = 1": round(rules1[r]["q1"], 2), "Trainingskosten (Mittel)": round(rules1[r]["train_cost"], 3), "Pinball-Summe": round(rules1[r]["pinball_sum"], 3)}
        for r in ("mittel", "emp", "direkt")
    ],
    hide_index=True,
)
kind, text = ev.verdict_hand(rules1, tau_hand)
getattr(st, kind)(text)

# --- Ergebnis und Kernfrage (Experiment 2) -----------------------------------------------------------------------------------------------

st.markdown("---")
st.markdown("## 🎯 Ergebnis: Mehrkosten der drei Regeln")
with st.spinner("Rechnet die 30 Seeds (je Seed ein lineares Programm)"):
    extra = _band(welt, int(p), int(n), float(tau))
m1, m2, m3 = st.columns(3)
med = {r: float(np.median(extra[r])) for r in ("mittel", "emp", "direkt")}
m1.metric("Mittel bestellen", ev.pct(med["mittel"]), delta="Referenz", delta_color="off")
# Delta immer als „mein Wert minus Referenz“ in Prozentpunkten; kleiner ist besser, die Farbe regelt delta_color="inverse" (Vorzeichen-Falle, DEMO-PLAYBOOK §2).
m2.metric("KQ + Quantil der Residuen", ev.pct(med["emp"]), delta=f"{med['emp'] - med['mittel']:+.1f} Prozentpunkte gegen Mittel".replace(".", ","), delta_color="inverse")
m3.metric("Direkt (Quantilregression)", ev.pct(med["direkt"]), delta=f"{med['direkt'] - med['emp']:+.1f} Prozentpunkte gegen KQ + Quantil".replace(".", ","), delta_color="inverse")
st.plotly_chart(build_band(extra), width="stretch", key="band_chart")
st.caption("Balken: Median der Mehrkosten gegen das Orakel über 30 feste Seeds, Fehlerbalken: Minimum bis Maximum. Kleiner ist besser.")

st.markdown("## ❓ Kernfrage: Lohnt sich das direkte Lernen auf den Kosten?")
kind, text = ev.verdict_streuung(extra, welt, int(p), int(n), float(tau))
getattr(st, kind)(text)
settings = Settings(welt=welt, tau=float(tau), n=int(n), p=int(p), seed=int(seed))
if st.button("📄 PDF der aktuellen Einstellung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, extra, text))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="newsvendor-direkt.pdf", mime="application/pdf", key="pdf_download")

# --- Experiment 2: Einzellauf ----------------------------------------------------------------------------------------------------------

st.markdown("---")
st.subheader("🔬 Experiment 2: Wann die Streuung zählt")
st.caption("Ein Einzellauf zum gewählten Zufalls-Seed: Bestellmenge minus optimale Menge (wahres Quantil) über dem ersten Merkmal x₁, auf dem festen Testsatz. "
           "Die Null-Linie ist das Orakel. Wählen Sie die Welt „Streuung wächst linear mit x₁“ und danach „Wahrheit linear, Streuung konstant“ und vergleichen Sie die Wolken beider Regeln.")
run = _single(settings)
st.plotly_chart(build_single(run), width="stretch", key="single_chart")
st.caption(f"Dieser eine Lauf (Seed {int(seed)}): Mehrkosten {ev.pct(run['extra']['emp'])} für KQ plus Quantil, {ev.pct(run['extra']['direkt'])} für direkt. Ein einzelner Seed ist kein Befund; die Bänder oben rechnen über 30 Seeds.")

# --- Experiment 3: Datenmenge und Merkmalszahl ----------------------------------------------------------------------------------------

st.markdown("---")
st.subheader("🔬 Experiment 3: Datenmenge und Merkmalszahl")
st.caption("Raster aus n und p in der gewählten Welt und bei dem gewählten τ: Farbe und Zahl sind Mehrkosten von direkt **minus** Mehrkosten von KQ plus Quantil (Prozentpunkte, Median über 30 Seeds; in Klammern: Seeds, in denen direkt besser ist). Blau und grüner Rahmen: direkt besser.")
with st.spinner("Rechnet das Raster (12 Zellen mit je 30 Seeds)"):
    g = _grid(welt, float(tau))
st.plotly_chart(build_grid(g), width="stretch", key="grid_chart")
kind, text = ev.verdict_grid(g, welt, float(tau))
getattr(st, kind)(text)

# --- Experiment 4: Mit Strafe ----------------------------------------------------------------------------------------------------------

st.markdown("---")
st.subheader("🔬 Experiment 4: Mit L1-Strafe")
st.caption("Die direkte Regel mit Strafe λ·|c|₁ auf den mittleren Pinball-Verlust. Verglichen werden: keine Strafe, die feste Stärke aus dem Regler und die Stärke, die eine 5-fache Kreuzvalidierung aus dem Raster 0 / 0,02 / 0,05 / 0,1 / 0,2 / 0,5 wählt. "
           "Welt, p und τ stammen aus den Einstellungen in der Seitenleiste; die Datenmenge hat einen eigenen Regler, weil die Kreuzvalidierung je Seed 30 Lösungen rechnet.")
e1, e2 = st.columns(2)
e1.select_slider("Beobachtungen n im Strafe-Experiment", options_of("nstrafe_select"), key="nstrafe_select")
e2.select_slider("Feste Strafstärke λ", options_of("lam_select"), key="lam_select", format_func=lambda x: ev.de(x, 2),
                 help="Gewicht der L1-Strafe auf den mittleren Pinball-Verlust; 0 = keine Strafe.")
n_strafe = st.session_state["nstrafe_select"]
lam = st.session_state["lam_select"]
with st.spinner("Rechnet die Kreuzvalidierung (30 Seeds, je Seed 30 Lösungen)"):
    pen = _penalty(welt, int(p), int(n_strafe), float(tau), float(lam))
st.plotly_chart(build_penalty(pen), width="stretch", key="penalty_chart")
st.caption(f"Die Kreuzvalidierung wählte im Median λ = {ev.de(float(np.median(pen['lam_cv'])), 2)}; über die Seeds streut die Wahl von {ev.de(float(pen['lam_cv'].min()), 2)} bis {ev.de(float(pen['lam_cv'].max()), 2)}.")
kind, text = ev.verdict_penalty(pen, welt, int(p), int(n_strafe), float(tau), float(lam))
getattr(st, kind)(text)

# --- Experiment 5: Falsche Klasse ------------------------------------------------------------------------------------------------------

st.markdown("---")
st.subheader("🔬 Experiment 5: Falsche Klasse")
st.caption("Zwei Welten, in denen die lineare Klasse die Wahrheit verfehlt: Mittelwert quadratisch in x₁, und Streuung abhängig von |x₁| (nicht monoton). p, n und τ stammen aus den Einstellungen in der Seitenleiste.")
with st.spinner("Rechnet beide Welten (je 30 Seeds)"):
    wrong = _wrong(int(p), int(n), float(tau))
st.plotly_chart(build_wrong(wrong), width="stretch", key="wrong_chart")
kind, text = ev.verdict_wrong_class(wrong, int(p), int(n), float(tau))
getattr(st, kind)(text)

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist (30 Seeds, Median der Mehrkosten gegen das Orakel, τ = 0,95) | Ausweg (soweit gemessen) |
|---|---|---|
| **Die Streuung hängt von den Merkmalen ab** | Bei konstanter Streuung (Welt „Wahrheit linear, Streuung konstant“, p = 3, n = 1000) kostet direkt 0,9 % gegen 0,3 % für KQ plus Quantil; direkt ist in 4 von 30 Seeds besser. | Zweistufig bleibt die einfache Regel. |
| **Genug Daten für die Zahl der Merkmale** | Mit p = 10 und n = 30 (Welt linear) kostet direkt 197,9 % gegen 47,3 % für KQ plus Quantil; direkt ist in 1 von 30 Seeds besser. | L1-Strafe holt einen Teil: 65,7 % mit per Kreuzvalidierung gewählter Strafe, KQ plus Quantil erreicht sie nicht. |
| **Die Wahl der Strafe ist verlässlich** | Mit p = 10 und n = 100 (Welt linear) steigt der Median von 33,2 % ohne Strafe auf 45,3 % mit per Kreuzvalidierung gewählter Strafe. | Nicht gemessen; eine andere Wahlregel wurde nicht geprüft. |
| **Die lineare Klasse trifft die Wahrheit** | Bei quadratischem Mittelwert (p = 3, n = 1000) 143,4 % gegen 141,3 %; bei Streuung, die vom Betrag von x₁ abhängt, 20,0 % gegen 19,5 %. In diesen beiden Welten repariert der Verlust allein die Klasse nicht. | Nicht gemessen; eine flexiblere Klasse ist hier nicht umgesetzt. |
| **Eine Familie von Instanzen** | Nur Normalrauschen, bei 0 gestutzt, vier Welten, simulierte Daten; keine Echtdaten, keine anderen Verteilungen. | Eigene Daten prüfen. |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $n$ Beobachtungen $(x_i, D_i)$ mit Merkmalen $x_i \in \mathbb{R}^p$ und Nachfragen $D_i \ge 0$ sowie die Kosten $b$ (je fehlende) und $h$ (je überschüssige Einheit); gesucht ist eine Bestellregel $q(x)$, die die erwarteten Kosten auf neuen Daten klein hält.

| Symbol | Bedeutung |
|---|---|
| $D$, $x$ | Nachfrage und Merkmalsvektor |
| $b$, $h$ | Kosten je fehlende bzw. überschüssige Einheit (in der Demo $h = 1$) |
| $\tau = b/(b+h)$ | kritisches Verhältnis |
| $q(x) = c_0 + x^\top c$ | lineare Bestellregel |
| $\rho_\tau$ | Pinball-Verlust |
| $\lambda$ | Stärke der L1-Strafe |
| $q^*(x)$ | optimale Bestellmenge (Orakel) |

**Kosten und Pinball-Verlust.** Mit $u = D - q$ und $\rho_\tau(u) = \max(\tau u, (\tau - 1)u)$ gilt

$$C(q, D) = b\,(D - q)_+ + h\,(q - D)_+ = (b + h)\,\rho_\tau(D - q).$$

**Optimum.** Die Bestellmenge, die die erwarteten Kosten gegeben $x$ minimiert, ist das $\tau$-Quantil der bedingten Verteilung der Nachfrage:

$$q^*(x) = F_{D \mid x}^{-1}(\tau).$$

**Zweistufig (KQ plus Quantil).** $\hat c$ ist die Kleinste-Quadrate-Lösung, $r_i = D_i - \hat c_0 - x_i^\top \hat c$ sind die Residuen und $r_{(k)}$ ihre aufsteigende Ordnung:

$$\hat q(x) = \hat c_0 + x^\top \hat c + r_{(\lceil \tau n \rceil)}.$$

**Direkt (Quantilregression).**

$$\min_{c_0,\, c} \; \frac{1}{n} \sum_{i=1}^{n} \rho_\tau\bigl(D_i - c_0 - x_i^\top c\bigr) + \lambda \,\lVert c \rVert_1 .$$

Als lineares Programm mit $c = c^+ - c^-$:

$$\min \; \frac{\tau}{n}\,\mathbf{1}^\top u + \frac{1 - \tau}{n}\,\mathbf{1}^\top v + \lambda\,\mathbf{1}^\top (c^+ + c^-) \quad \text{u. d. N.} \quad D - c_0 \mathbf{1} - X (c^+ - c^-) = u - v, \;\; u, v, c^+, c^- \ge 0 .$$

**Mehrkosten.** Die Nachfrage ist $D = \max(0, Y)$ mit $Y \sim \mathcal{N}(m(x), s(x)^2)$. Für $q \ge 0$ und $z = (q - m)/s$ gilt

$$\mathbb{E}\,(D - q)_+ = s\,\varphi(z) - (q - m)\,(1 - \Phi(z)),$$

$$\mathbb{E}\,(q - D)_+ = s\,\varphi(z) + (q - m)\,\Phi(z) - \bigl[\, s\,\varphi(m/s) + m\,\Phi(m/s) \,\bigr].$$

Daraus folgen die erwarteten Kosten $\bar C(q)$ gemittelt über den Testsatz; die Mehrkosten sind $100\,\bigl(\bar C(\hat q) - \bar C(q^*)\bigr) / \bar C(q^*)$.

**Literatur.** Ban, G.-Y. und Rudin, C.: *The Big Data Newsvendor: Practical Insights from Machine Learning*, Operations Research 67(1), 2019. **Nicht im Volltext gelesen**; laut veröffentlichtem Abstract wird die Bestellmenge direkt aus Merkmalen und Nachfragen gelernt statt über eine Prognose. Weitere Inhalte (mit und ohne Regularisierung, hohe Dimension, inkonsistente Entscheidungen beim Weglassen von Merkmalen) stammen aus der Zusammenfassung einer Websuche, der Abstract im Wortlaut ist dazu nicht geprüft. Zahlen und Satzbedingungen der Quelle sind hier nicht übernommen; alle Zahlen dieser Demo stammen aus den eigenen Messungen (README).
        """
    )

st.markdown("---")
sync_query_params({
    "welt_select": welt, "tau_select": tau, "n_select": int(n), "p_select": int(p), "lam_select": lam, "nstrafe_select": int(n_strafe),
    "tauhand_select": tau_hand, "seed_input": int(seed),
})
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Entscheidungsorientiert: Regret bis Wahl](https://sebastianhanisch.net/konzepte-entscheidungsorientiert.html)."
)
