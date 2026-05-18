"""
visualizations/charts.py
Reusable Plotly chart builders — all colors valid rgba(), no 8-digit hex.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ── THEME ──────────────────────────────────────────────────────────────────────
DARK_BG    = "rgba(0,0,0,0)"
PLOT_BG    = "#0d0d1a"
GRID_COLOR = "#1e1e3a"
AXIS_COLOR = "#888888"
GOLD   = "#f0b429"
GREEN  = "#4CAF50"
RED    = "#e74c3c"
BLUE   = "#3498db"
PURPLE = "#9b59b6"

_RGBA_BASE = {
    GREEN:     "rgba(76,175,80,",
    RED:       "rgba(231,76,60,",
    BLUE:      "rgba(52,152,219,",
    PURPLE:    "rgba(155,89,182,",
    GOLD:      "rgba(240,180,41,",
    "#e67e22": "rgba(230,126,34,",
    "#1abc9c": "rgba(26,188,156,",
}

def _rgba(hex_color: str, alpha: float = 0.15) -> str:
    base = _RGBA_BASE.get(hex_color, "rgba(240,180,41,")
    return f"{base}{alpha})"

def _base(extra: dict = None):
    """Build layout kwargs, merging optional extras (including margin)."""
    d = dict(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color="white"),
        margin=dict(t=30, b=30, l=40, r=20),
    )
    if extra:
        d.update(extra)
    return d

def _axis(title="", gridcolor=GRID_COLOR, color=AXIS_COLOR):
    return dict(title=title, color=color, gridcolor=gridcolor, showgrid=True)


# ── CHARTS ─────────────────────────────────────────────────────────────────────

def win_probability_donut(labels, values, colors=None):
    if colors is None:
        colors = [GREEN, GOLD, RED]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=[v * 100 for v in values],
        hole=0.62,
        marker=dict(colors=colors, line=dict(color="#0d0d1a", width=3)),
        textinfo="label+percent",
        textfont=dict(size=13, color="white"),
        rotation=90,
    ))
    fig.update_layout(**_base(), showlegend=False, height=340,
        annotations=[dict(text=f"{max(values):.0%}", x=0.5, y=0.5,
                          font=dict(size=26, color=GOLD), showarrow=False)])
    return fig


def radar_chart(teams_attrs: dict, categories: list, range_: tuple = (50, 90)):
    palette = [GREEN, RED, BLUE, PURPLE, GOLD]
    fig = go.Figure()
    for idx, (team, values) in enumerate(teams_attrs.items()):
        color = palette[idx % len(palette)]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself", name=team,
            line=dict(color=color, width=2),
            fillcolor=_rgba(color, 0.15),
        ))
    fig.update_layout(**_base(), height=400,
        polar=dict(bgcolor="#0a0a18",
                   radialaxis=dict(visible=True, range=list(range_), color="#555", gridcolor=GRID_COLOR),
                   angularaxis=dict(color="#888", gridcolor=GRID_COLOR)),
        legend=dict(font=dict(color="white")))
    return fig


def horizontal_bar(labels, values, color_scale="Oranges",
                   title="", x_title="", label_suffix="%", range_pad=10):
    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h",
        marker=dict(color=values, colorscale=color_scale, line=dict(color="#0d0d1a", width=0.5)),
        text=[f"{v:.1f}{label_suffix}" for v in values],
        textposition="outside", textfont=dict(color="white", size=11),
    ))
    fig.update_layout(
        **_base({"margin": dict(l=160, r=90, t=40 if title else 20, b=20)}),
        title=dict(text=title, font=dict(color="white", size=14)) if title else None,
        xaxis=dict(**_axis(x_title), range=[0, max(values) + range_pad]),
        yaxis=dict(color="white"),
        height=max(350, len(labels) * 28),
    )
    return fig


def line_chart(x_series, y_series_dict: dict, title="", y_title="", height=300):
    palette = [GOLD, GREEN, RED, BLUE, PURPLE]
    fig = go.Figure()
    for idx, (name, y) in enumerate(y_series_dict.items()):
        color = palette[idx % len(palette)]
        fig.add_trace(go.Scatter(
            x=x_series, y=y, mode="lines+markers", name=name,
            line=dict(color=color, width=2), marker=dict(size=4),
            fill="tozeroy" if idx == 0 else "none",
            fillcolor=_rgba(color, 0.09),
        ))
    fig.update_layout(**_base(), xaxis=_axis(), yaxis=_axis(y_title),
                      legend=dict(font=dict(color="white")), height=height)
    return fig


def grouped_bar(categories, series_dict: dict, title="", y_title="", height=350):
    palette = [GOLD, GREEN, BLUE, RED, PURPLE]
    fig = go.Figure()
    for idx, (name, values) in enumerate(series_dict.items()):
        fig.add_trace(go.Bar(
            name=name, x=categories, y=values,
            marker_color=palette[idx % len(palette)],
            text=[f"{v:.1f}" for v in values],
            textposition="outside", textfont=dict(color="white", size=11),
        ))
    fig.update_layout(**_base(), barmode="group",
        title=dict(text=title, font=dict(color="white")) if title else None,
        xaxis=dict(color="white"), yaxis=dict(**_axis(y_title)),
        legend=dict(font=dict(color="white")), height=height)
    return fig


def histogram(values, title="", x_title="", color=GOLD, nbins=15, height=300):
    fig = go.Figure(go.Histogram(
        x=values, nbinsx=nbins,
        marker=dict(color=color, opacity=0.85, line=dict(color="#0d0d1a", width=0.5)),
    ))
    fig.update_layout(**_base(),
        title=dict(text=title, font=dict(color="white")) if title else None,
        xaxis=_axis(x_title), yaxis=_axis("Count"), height=height)
    return fig


def elo_timeline(dates, elo_values, team_name="", height=280):
    fig = go.Figure(go.Scatter(
        x=dates, y=elo_values, mode="lines",
        line=dict(color=GOLD, width=2),
        fill="tozeroy", fillcolor=_rgba(GOLD, 0.09),
        name=team_name,
    ))
    fig.update_layout(**_base(), xaxis=_axis(), yaxis=_axis("Elo Rating"), height=height)
    return fig
