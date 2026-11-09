"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import nvd_constants as C
import nvd_evaluation as ev


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_hand_curves(tau: float, rules: dict):
    """Experiment 1 links: Kosten der drei Beobachtungen je Gruppe über der Bestellmenge; die Marken zeigen, was jede Regel bestellt."""
    qs = np.linspace(0.0, 20.0, 201)
    curves = ev.hand_cost_curve(tau, qs)
    fig = go.Figure()
    for g, dash in ((0.0, "solid"), (1.0, "dash")):
        fig.add_trace(go.Scatter(x=qs, y=curves[g], mode="lines", line=dict(color=C.COLOR_TRUTH, width=2, dash=dash), name=f"Kosten Gruppe x = {int(g)}",
                                 hovertemplate="Menge %{x:.1f}: Kosten %{y:.1f}<extra></extra>"))
    for rule in ("mittel", "emp", "direkt"):
        xs = [rules[rule]["q0"], rules[rule]["q1"]]
        ys = [float(ev.hand_cost_curve(tau, np.array([xs[0]]))[0.0][0]), float(ev.hand_cost_curve(tau, np.array([xs[1]]))[1.0][0])]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=11, color=C.RULE_COLORS[rule], line=dict(width=1, color="white")), name=C.RULE_NAMES[rule],
                                 hovertemplate=C.RULE_NAMES[rule] + ": Menge %{x:.2f}, Kosten %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Bestellmenge q")
    fig.update_yaxes(title_text="Kosten der 3 Beobachtungen einer Gruppe")
    return _base(fig)


def build_hand_lines(rules: dict):
    """Experiment 1 rechts: die Beobachtungen und die drei gelernten Geraden über dem Merkmal x."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(C.HAND_X), y=list(C.HAND_Y), mode="markers", marker=dict(size=10, color="rgba(60,60,60,0.8)"), name="Nachfragen",
                             hovertemplate="x = %{x:.0f}: Nachfrage %{y:.0f}<extra></extra>"))
    for rule in ("mittel", "emp", "direkt"):
        r = rules[rule]
        fig.add_trace(go.Scatter(x=[0, 1], y=[r["q0"], r["q1"]], mode="lines+markers", line=dict(color=C.RULE_COLORS[rule], width=3), name=C.RULE_NAMES[rule],
                                 hovertemplate=C.RULE_NAMES[rule] + ": x = %{x:.0f}, Menge %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Merkmal x", tickvals=[0, 1], range=[-0.2, 1.2])
    fig.update_yaxes(title_text="Nachfrage bzw. Bestellmenge")
    return _base(fig)


def build_band(extra: dict, rules=("mittel", "emp", "direkt"), log=False):
    """Mehrkosten gegen das Orakel je Regel: Median als Balken, Minimum bis Maximum über die Seeds als Fehlerbalken."""
    stats = {r: ev.summarize(extra[r]) for r in rules}
    fig = go.Figure(go.Bar(
        x=[C.RULE_NAMES[r] for r in rules], y=[stats[r]["median"] for r in rules], marker_color=[C.RULE_COLORS[r] for r in rules],
        error_y=dict(type="data", symmetric=False, array=[stats[r]["high"] - stats[r]["median"] for r in rules], arrayminus=[stats[r]["median"] - stats[r]["low"] for r in rules]),
        text=[ev.pct(stats[r]["median"]) for r in rules], textposition="outside",
        hovertemplate="%{x}: Median %{y:.1f} %<extra></extra>", showlegend=False,
    ))
    fig.update_yaxes(title_text="Mehrkosten gegen das Orakel (%)", type="log" if log else "linear")
    return _base(fig)


def build_single(run: dict):
    """Ein Lauf: Abweichung der Bestellmenge von der optimalen (wahres Quantil) über dem ersten Merkmal, auf dem Testsatz; 0 ist das Orakel."""
    fig = go.Figure()
    x1 = run["x1"]
    keep = np.arange(0, len(x1), max(1, len(x1) // 400))
    for rule in ("emp", "direkt"):
        dev = run["q"][rule] - run["q_true"]
        fig.add_trace(go.Scatter(x=x1[keep], y=dev[keep], mode="markers", marker=dict(size=5, color=C.RULE_COLORS[rule], opacity=0.55), name=C.RULE_NAMES[rule],
                                 hovertemplate="x₁ = %{x:.2f}: %{y:.1f}<extra></extra>"))
    fig.add_hline(y=0.0, line=dict(color=C.COLOR_TRUTH, width=1.5, dash="dot"))
    fig.update_xaxes(title_text="erstes Merkmal x₁")
    fig.update_yaxes(title_text="Bestellmenge minus optimale Menge")
    return _base(fig)


def build_grid(g: dict):
    """Raster n × p: Farbe = Mehrkosten direkt minus Mehrkosten KQ + Quantil (Median über die Seeds); grüne Rahmen markieren die Zellen, in denen direkt im Median besser ist."""
    ns, ps, diff = g["ns"], g["ps"], g["diff"]
    lim = float(max(1e-9, np.max(np.abs(diff))))
    text = [[f"{diff[i, j]:+.1f}".replace(".", ",") + f"<br>({int(g['wins'][i, j])}/{g['m']})" for j in range(len(ns))] for i in range(len(ps))]
    fig = go.Figure(go.Heatmap(
        z=diff, x=list(range(len(ns))), y=list(range(len(ps))), zmin=-lim, zmax=lim, colorscale="RdBu_r", text=text, texttemplate="%{text}", showscale=True,
        colorbar=dict(title=dict(text="direkt minus KQ+Quantil (Prozentpunkte)", side="top"), nticks=4),
        hovertemplate="p = %{y}, n = %{x}: %{z:.1f} Prozentpunkte<extra></extra>",
    ))
    for i in range(len(ps)):
        for j in range(len(ns)):
            if diff[i, j] < 0:
                fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5, line=dict(color=C.COLOR_DIRECT, width=3))
    fig.update_xaxes(title_text="Beobachtungen n", tickvals=list(range(len(ns))), ticktext=[str(n) for n in ns])
    fig.update_yaxes(title_text="Merkmale p", tickvals=list(range(len(ps))), ticktext=[str(p) for p in ps])
    return _base(fig, 330)


def build_penalty(b: dict):
    """Experiment 4: Mehrkosten von KQ + Quantil, direkt ohne Strafe, mit fester Strafe und mit per Kreuzvalidierung gewählter Strafe."""
    rules = ("emp", "direkt", "direkt_l1", "direkt_cv")
    names = {**C.RULE_NAMES, "direkt_cv": "direkt, Strafe per Kreuzvalidierung"}
    colors = {**C.RULE_COLORS, "direkt_cv": "#8e5ea2"}
    stats = {r: ev.summarize(b[r]) for r in rules}
    fig = go.Figure(go.Bar(
        x=[names[r] for r in rules], y=[stats[r]["median"] for r in rules], marker_color=[colors[r] for r in rules],
        error_y=dict(type="data", symmetric=False, array=[stats[r]["high"] - stats[r]["median"] for r in rules], arrayminus=[stats[r]["median"] - stats[r]["low"] for r in rules]),
        text=[ev.pct(stats[r]["median"]) for r in rules], textposition="outside", hovertemplate="%{x}: Median %{y:.1f} %<extra></extra>", showlegend=False,
    ))
    fig.update_yaxes(title_text="Mehrkosten gegen das Orakel (%)")
    return _base(fig)


def build_wrong(rows: dict):
    """Experiment 5: je Welt Mehrkosten von KQ + Quantil und direkt (Median, Minimum bis Maximum)."""
    fig = go.Figure()
    kinds = list(rows)
    for rule in ("emp", "direkt"):
        stats = [ev.summarize(rows[k][rule]) for k in kinds]
        fig.add_trace(go.Bar(
            x=[C.WELT_NAMEN[k] for k in kinds], y=[s["median"] for s in stats], name=C.RULE_NAMES[rule], marker_color=C.RULE_COLORS[rule],
            error_y=dict(type="data", symmetric=False, array=[s["high"] - s["median"] for s in stats], arrayminus=[s["median"] - s["low"] for s in stats]),
            text=[ev.pct(s["median"]) for s in stats], textposition="outside", hovertemplate="%{x}, " + C.RULE_NAMES[rule] + ": Median %{y:.1f} %<extra></extra>",
        ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Mehrkosten gegen das Orakel (%)")
    return _base(fig)
