"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

import numpy as np
from fpdf import FPDF

import nvd_constants as C
import nvd_evaluation as ev

_REPLACE = {"→": "->", "–": "-", "—": "-", "τ": "tau", "λ": "lambda", "σ": "sigma", "μ": "mu", "β": "beta", "₁": "1", "×": "x", "−": "-", "„": '"', "“": '"', "≥": ">=", "·": "*", "…": "..."}


def _s(text) -> str:
    out = str(text).replace("**", "")
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(settings: ev.Settings, extra: dict, verdict_text: str) -> bytes:
    """Einstellung, Mehrkosten je Regel (Median, Minimum, Maximum über die Seeds) und das Verdikt der Kernfrage."""
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Direkt auf den Kosten lernen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Welt: {C.WELT_NAMEN[settings.welt]}; p = {settings.p} Merkmale, n = {settings.n} Beobachtungen, tau = {ev.tau_text(settings.tau)} (kritisches Verhältnis), "
                            f"{len(C.SWEEP_SEEDS)} feste Seeds. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.multi_cell(0, 5, _s(verdict_text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    widths = [70, 30, 30, 30]
    pdf.set_font("Helvetica", "B", 9)
    for head, w in zip(["Regel", "Median (%)", "Minimum (%)", "Maximum (%)"], widths):
        pdf.cell(w, 6, _s(head), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for rule in ("mittel", "emp", "direkt"):
        v = np.asarray(extra[rule])
        row = [C.RULE_NAMES[rule], ev.de(float(np.median(v))), ev.de(float(v.min())), ev.de(float(v.max()))]
        for cell, w in zip(row, widths):
            pdf.cell(w, 5.5, _s(cell), border=1)
        pdf.ln()
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Mehrkosten gegen das Orakel (wahres Quantil) in Prozent der Orakelkosten. Quelle: newsvendor-direkt-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
