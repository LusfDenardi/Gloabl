"""Tema Carbon Pro unificado para GlobalStat."""

import plotly.graph_objects as go
import plotly.io as pio
from typing import Dict, Any, Optional

# Cores do tema Carbon Pro (versão consolidada)
COLORS = {
    "bg": "#0B0C0D",
    "panel": "#0F1114",
    "panel_alt": "#0C0F14",
    "panel_2": "#12161B",
    "border": "#1E2633",
    "edge": "#1E293B",
    "line": "#1E293B",
    "grid": "rgba(255,255,255,0.08)",
    "grid_alt": "#1B2330",
    "text": "#E6F1EE",
    "text_alt": "#E8EDF2",
    "muted": "#9AA7A1",
    "muted_alt": "#9AA7B5",
    "primary": "#C4FFE4",
    "accent": "#59F0C8",
    "accent_alt": "#6FB4FF",
    "positive": "#4ADE80",
    "good": "#10B981",
    "negative": "#F87171",
    "danger": "#EF4444",
    "warning": "#FBBF24",
    "warn": "#FFD84D",
    "shadow": "0 10px 30px rgba(0,0,0,.35)",
}

# Template Plotly Carbon Pro
def create_plotly_template() -> go.layout.Template:
    """Cria e retorna o template Plotly Carbon Pro."""
    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=COLORS["bg"],
            plot_bgcolor=COLORS["panel"],
            font=dict(
                color=COLORS["text"],
                family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica Neue, Arial, Noto Sans, sans-serif"
            ),
            margin=dict(l=40, r=24, t=56, b=40),
            xaxis=dict(
                showgrid=True,
                gridcolor=COLORS["grid"],
                zeroline=False,
                linecolor=COLORS["line"],
                tickcolor=COLORS["line"],
                color=COLORS["text"]
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=COLORS["grid"],
                zeroline=False,
                linecolor=COLORS["line"],
                tickcolor=COLORS["line"],
                color=COLORS["text"]
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                font=dict(color=COLORS["text"])
            ),
            hoverlabel=dict(
                bgcolor=COLORS["panel_alt"],
                bordercolor=COLORS["border"],
                font_color=COLORS["text"]
            ),
            colorway=[
                COLORS["accent_alt"],
                COLORS["text"],
                COLORS["primary"],
                COLORS["positive"],
                COLORS["warning"],
                COLORS["negative"]
            ],
        )
    )

# Template alternativo (carbonpro) usado em macro_app.py
def create_plotly_template_alt() -> go.layout.Template:
    """Cria template alternativo com cores ligeiramente diferentes."""
    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=COLORS["bg"],
            plot_bgcolor=COLORS["panel"],
            font=dict(color=COLORS["text_alt"]),
            margin=dict(l=30, r=20, t=40, b=30),
            xaxis=dict(
                showgrid=True,
                gridcolor=COLORS["grid_alt"],
                zeroline=False,
                tickcolor=COLORS["text_alt"]
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=COLORS["grid_alt"],
                zeroline=False,
                tickcolor=COLORS["text_alt"]
            ),
            legend=dict(
                bordercolor=COLORS["border"],
                borderwidth=1,
                bgcolor="rgba(0,0,0,0.0)"
            ),
            hoverlabel=dict(
                bgcolor=COLORS["panel_alt"],
                bordercolor=COLORS["border"],
                font_color=COLORS["text_alt"]
            ),
        )
    )

# Inicializa templates
pio.templates["carbon_pro"] = create_plotly_template()
pio.templates["carbonpro"] = create_plotly_template_alt()
pio.templates.default = "carbon_pro"

# Funções auxiliares de estilo
def cp_card(style: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Retorna estilo de card Carbon Pro."""
    base = {
        "background": f"linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.00)), {COLORS['panel']}",
        "border": f"1px solid {COLORS['border']}",
        "borderRadius": "14px",
        "boxShadow": COLORS["shadow"],
        "padding": "10px"
    }
    if style:
        base.update(style)
    return base

def get_toggle_btn_style() -> Dict[str, str]:
    """Retorna estilo de botão toggle."""
    return {
        'width': '42px',
        'height': '42px',
        'borderRadius': '9999px',
        'display': 'inline-flex',
        'alignItems': 'center',
        'justifyContent': 'center',
        'fontSize': '18px',
        'lineHeight': '1',
        'background': COLORS['panel_alt'],
        'color': COLORS['text_alt'],
        'border': f"1px solid {COLORS['border']}",
        'boxShadow': COLORS['shadow'],
        'cursor': 'pointer',
        'transition': 'transform .15s ease, box-shadow .2s ease, background .2s ease',
        'marginRight': '12px'
    }

def get_ghost_btn_style() -> Dict[str, str]:
    """Retorna estilo de botão ghost."""
    return {
        'padding': '10px 14px',
        'cursor': 'pointer',
        'marginRight': '10px',
        'background': 'transparent',
        'color': COLORS['text_alt'],
        'border': f"1px solid {COLORS['border']}",
        'borderRadius': '10px'
    }

def get_yellow_label_style() -> Dict[str, str]:
    """Retorna estilo de label amarelo."""
    return {
        'color': COLORS['warn'],
        'fontSize': '13px',
        'fontWeight': '700',
        'marginBottom': '6px'
    }

# Objeto de tema completo
CARBON_PRO_THEME = {
    "colors": COLORS,
    "plotly_template": "carbon_pro",
    "plotly_template_alt": "carbonpro",
    "css_path": "assets/css/carbon_pro.css",
    "functions": {
        "cp_card": cp_card,
        "get_toggle_btn_style": get_toggle_btn_style,
        "get_ghost_btn_style": get_ghost_btn_style,
        "get_yellow_label_style": get_yellow_label_style,
    }
}

def get_carbon_pro_theme() -> Dict[str, Any]:
    """Retorna o tema Carbon Pro completo."""
    return CARBON_PRO_THEME

