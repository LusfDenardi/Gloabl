# ===================== Análise Estatística — Beta / Alpha (Carbon Pro) =====================
import os
from pathlib import Path
import datetime as dt
import time
import requests

import dash
from dash import dcc, html, Input, Output
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from fitter import Fitter
from scipy.stats import gaussian_kde, anderson, normaltest
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

# ===============================================================
# === Paleta (Carbon Pro) — tokens equivalentes ao CSS
# ===============================================================
BG         = "#0B0C0D"
PANEL      = "#0F1114"
PANEL_2    = "#12161B"
LINE       = "#1E293B"
TEXT       = "#E6F1EE"
MUTED      = "#9AA7A1"
PRIMARY    = "#C4FFE4"  # mint institucional
ACCENT     = "#6FB4FF"
POSITIVE   = "#4ADE80"
NEGATIVE   = "#F87171"
WARNING    = "#FBBF24"
GRID_COLOR = "rgba(255,255,255,0.06)"

# ===============================================================
# === Template Plotly (Carbon Pro)
# ===============================================================
pio.templates["carbon_pro"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica Neue, Arial, Noto Sans, sans-serif"),
        xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, linecolor=LINE, tickcolor=LINE, color=TEXT),
        yaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, linecolor=LINE, tickcolor=LINE, color=TEXT),
        margin=dict(l=40, r=24, t=56, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(color=TEXT)),
        colorway=[ACCENT, TEXT, PRIMARY, POSITIVE, NEGATIVE, WARNING],
    )
)
pio.templates.default = "carbon_pro"

# ===============================================================
# === Parâmetros / Opções
# ===============================================================
ativos_opcoes = [
    # CRYPTO
    {'label': 'Crypto • Bitcoin (BTC-USD)', 'value': 'BTC-USD'},
    {'label': 'Crypto • Ethereum (ETH-USD)', 'value': 'ETH-USD'},
    {'label': 'Crypto • BNB (BNB-USD)', 'value': 'BNB-USD'},
    {'label': 'Crypto • Solana (SOL-USD)', 'value': 'SOL-USD'},
    {'label': 'Crypto • XRP (XRP-USD)', 'value': 'XRP-USD'},
    {'label': 'Crypto • Dogecoin (DOGE-USD)', 'value': 'DOGE-USD'},
    # COMMODITIES
    {'label': 'Commodity • Gold (GC=F)', 'value': 'GC=F'},
    {'label': 'Commodity • Silver (SI=F)', 'value': 'SI=F'},
    {'label': 'Commodity • WTI (CL=F)', 'value': 'CL=F'},
    {'label': 'Commodity • Brent (BZ=F)', 'value': 'BZ=F'},
    {'label': 'Commodity • Natural Gas (NG=F)', 'value': 'NG=F'},
    {'label': 'Commodity • Copper (HG=F)', 'value': 'HG=F'},
    {'label': 'Commodity • Coffee (KC=F)', 'value': 'KC=F'},
    {'label': 'Commodity • Soybeans (ZS=F)', 'value': 'ZS=F'},
    {'label': 'Commodity • Corn (ZC=F)', 'value': 'ZC=F'},
    {'label': 'Commodity • Wheat (ZW=F)', 'value': 'ZW=F'},
    # FOREX
    {'label': 'Forex • EUR/USD (EURUSD=X)', 'value': 'EURUSD=X'},
    {'label': 'Forex • GBP/USD (GBPUSD=X)', 'value': 'GBPUSD=X'},
    {'label': 'Forex • USD/JPY (USDJPY=X)', 'value': 'USDJPY=X'},
    {'label': 'Forex • USD/BRL (BRL=X)', 'value': 'BRL=X'},
    {'label': 'Forex • USD/CAD (CAD=X)', 'value': 'CAD=X'},
    {'label': 'Forex • USD/CHF (CHFUSD=X)', 'value': 'CHFUSD=X'},
    # ÍNDICES
    {'label': 'Equity Index • S&P 500 (^GSPC)', 'value': '^GSPC'},
    {'label': 'Equity Index • NASDAQ 100 (^NDX)', 'value': '^NDX'},
    {'label': 'Equity Index • Dow Jones (^DJI)', 'value': '^DJI'},
    {'label': 'Equity Index • Russell 2000 (^RUT)', 'value': '^RUT'},
    {'label': 'Equity Index • VIX (^VIX)', 'value': '^VIX'},
    {'label': 'Equity Index • Ibovespa (^BVSP)', 'value': '^BVSP'},
    {'label': 'Equity Index • Mini Índice Futuro (WINJ25.SA)', 'value': 'WINJ25.SA'},
    {'label': 'Equity Index • Mini Índice Futuro (WINM25.SA)', 'value': 'WINM25.SA'},
    {'label': 'Equity Index • Mini Índice Futuro (WINQ25.SA)', 'value': 'WINQ25.SA'},
    {'label': 'Equity Index • Mini Índice Futuro (WINV25.SA)', 'value': 'WINV25.SA'},
    # ETFs
    {'label': 'ETF • SPY (S&P 500)', 'value': 'SPY'},
    {'label': 'ETF • QQQ (Nasdaq)', 'value': 'QQQ'},
    {'label': 'ETF • IWM (Russell 2000)', 'value': 'IWM'},
    {'label': 'ETF • EWZ (Brazil)', 'value': 'EWZ'},
    {'label': 'ETF • GLD (Gold)', 'value': 'GLD'},
    {'label': 'ETF • TLT (Long Bonds)', 'value': 'TLT'},
    {'label': 'ETF • HYG (High Yield Bonds)', 'value': 'HYG'}
]

PERIODOS = {"6M": "6mo", "YTD": "ytd", "1Y": "1y", "2Y": "2y", "5Y": "5y", "Max": "max"}
FREQUENCIAS = {"Diária": "1d", "Semanal": "1wk", "Mensal": "1mo"}

# cache: chave = (symbol, range/period, interval) ou (symbol, start,end,interval)
_CHART_CACHE = {}
_CHART_TTL = 300.0  # 5 min

def _cache_get(key):
    item = _CHART_CACHE.get(key)
    if not item: return None
    ts, obj = item
    return obj if (time.time() - ts) < _CHART_TTL else None

def _cache_set(key, obj):
    _CHART_CACHE[key] = (time.time(), obj)

def _yahoo_chart_period(symbol: str, period: str, interval: str):
    key = ("period", symbol, period, interval)
    hit = _cache_get(key)
    if hit is not None:
        return hit
    try:
        r = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"range": period, "interval": interval, "events": "div,splits", "includeAdjustedClose": "true"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        r.raise_for_status()
        js = r.json().get("chart", {}).get("result", [])
        if not js: return pd.Series(dtype=float)
        r0 = js[0]
        ts = r0.get("timestamp", [])
        ind = r0.get("indicators", {})
        quote = (ind.get("quote") or [{}])[0]
        closes = quote.get("close")
        if (not closes) and ("adjclose" in ind):
            closes = (ind.get("adjclose") or [{}])[0].get("adjclose")
        if not ts or not closes: return pd.Series(dtype=float)
        idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert("UTC").tz_localize(None)
        s = pd.Series(closes, index=idx, name="Close").astype(float).dropna()
        _cache_set(key, s)
        return s
    except Exception:
        return pd.Series(dtype=float)

def _yahoo_chart_range(symbol: str, start: dt.date, end: dt.date, interval: str = "1d"):
    key = ("range", symbol, str(start), str(end), interval)
    hit = _cache_get(key)
    if hit is not None:
        return hit
    try:
        p1 = int(time.mktime(dt.datetime(start.year, start.month, start.day, 0, 0).timetuple()))
        p2 = int(time.mktime(dt.datetime(end.year, end.month, end.day, 23, 59).timetuple()))
        r = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"period1": p1, "period2": p2, "interval": interval, "events": "div,splits", "includeAdjustedClose": "true"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        r.raise_for_status()
        js = r.json().get("chart", {}).get("result", [])
        if not js: return pd.DataFrame()
        r0 = js[0]
        ts = r0.get("timestamp", [])
        ind = r0.get("indicators", {})
        if not ts or not ind: return pd.DataFrame()
        idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert("UTC").tz_localize(None)
        quote = (ind.get("quote") or [{}])[0]
        data = {
            "Open":  quote.get("open"),
            "High":  quote.get("high"),
            "Low":   quote.get("low"),
            "Close": quote.get("close"),
            "Volume": quote.get("volume"),
        }
        adjc = (ind.get("adjclose") or [{}])
        if adjc and adjc[0].get("adjclose") is not None:
            data["Adj Close"] = adjc[0]["adjclose"]
        df = pd.DataFrame({k: v for k, v in data.items() if v is not None}, index=idx).astype(float)
        df = df.dropna(how="all")
        _cache_set(key, df)
        return df
    except Exception:
        return pd.DataFrame()

# ===============================================================
# === Funções Auxiliares
# ===============================================================
def baixar_serie(ticker, periodo, frequencia):
    # 1) tenta Yahoo Chart API (estável no Railway)
    s = _yahoo_chart_period(ticker, periodo, frequencia)
    if not s.empty:
        return s

    # 2) fallback para yfinance (casos raros)
    df = yf.download(ticker, period=periodo, interval=frequencia,
                     progress=False, auto_adjust=False, threads=False)
    if df.empty:
        return pd.Series(dtype=float)

    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df.columns.get_level_values(0):
            df = df.xs('Adj Close', axis=1, level=0)
        else:
            df = df.xs('Close', axis=1, level=0)
        return pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    else:
        col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        return pd.to_numeric(df[col], errors="coerce").dropna()

def analisar_lag(df, max_lag=10):
    melhor_corr = -np.inf
    melhor_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = df['ret_ativo'][:lag].corr(df['ret_bench'][-lag:])
        elif lag > 0:
            corr = df['ret_ativo'][lag:].corr(df['ret_bench'][:-lag])
        else:
            corr = df['ret_ativo'].corr(df['ret_bench'])
        if pd.notna(corr) and abs(corr) > abs(melhor_corr):
            melhor_corr = corr
            melhor_lag = lag
    if melhor_lag > 0:
        interpretacao = f"O ativo segue o benchmark com {melhor_lag} período(s) de atraso."
    elif melhor_lag < 0:
        interpretacao = f"O ativo antecipa o benchmark em {abs(melhor_lag)} período(s)."
    else:
        interpretacao = "Sem defasagem detectada (lag = 0)."
    return melhor_lag, melhor_corr, interpretacao

# --- Guardas robustas ---
def _finite_nonconst(x, min_len):
    x = np.asarray(x)
    return (len(x) >= min_len) and np.isfinite(x).all() and (np.nanvar(x) > 0)

def _safe_normaltest(y):
    if _finite_nonconst(y, 8):
        try:
            _, p = normaltest(y)
            return float(p)
        except Exception:
            return np.nan
    return np.nan

def _safe_anderson(y):
    try:
        if _finite_nonconst(y, 3):
            res = anderson(y)
            return f"A² = {res.statistic:.4f}"
    except Exception:
        pass
    return "—"

def _safe_kde(x, grid):
    try:
        if _finite_nonconst(x, 5):
            kde = gaussian_kde(x)
            return kde(grid)
    except Exception:
        pass
    return None

def _safe_corr(a, b):
    if _finite_nonconst(a, 2) and _finite_nonconst(b, 2):
        try:
            return float(np.corrcoef(a, b)[0, 1])
        except Exception:
            return np.nan
    return np.nan

# ===============================================================
# === Componente CSS (inline) — base Carbon Pro
# ===============================================================
CARBON_PRO_CSS = r"""
<style>
:root{
  --bg:#0b0c0e;
  --panel:rgba(255,255,255,0.02);
  --panel-2:rgba(255,255,255,0.03);
  --line:#1a1b1e;
  --text:#e8edf2;
  --muted:#aab3c0;
  --accent:#59f0c8;
  --positive:#59f0c8;
  --negative:#ef4444;
  --warning:#fbbf24;
  --shadow:0 0 28px rgba(255,255,255,0.015),0 6px 20px rgba(0,0,0,0.45);
  --radius:16px;
  --font:"Inter",system-ui,-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}

html, body{
  height:100%;
  margin:0;
  background:var(--bg)!important;
  color:var(--text)!important;
  font-family:var(--font)!important;
  overflow-x:hidden;
}

/* ============================================================
   HEADERS / TÍTULOS
   ============================================================ */
h1,h2,h3,h4,h5,label{
  color:var(--text)!important;
  text-shadow:none!important;
}

h2{
  font-size:24px;
  font-weight:600;
}

h4{
  font-size:14px;
  font-weight:700;
  color:var(--accent)!important;
  letter-spacing:.03em;
  margin-bottom:6px;
}

/* ============================================================
   WRAPPERS / LAYOUT ORIGINAL PRESERVADO
   ============================================================ */
.app{
  width:100%!important;
  padding:24px!important;
  box-sizing:border-box!important;
}

.controls-grid{
  display:grid;
  gap:12px;
  grid-template-columns:repeat(4,minmax(220px,1fr));
}

@media (max-width:1200px){
  .controls-grid{ grid-template-columns:repeat(3,1fr); }
}
@media (max-width:900px){
  .controls-grid{ grid-template-columns:repeat(2,1fr); }
}
@media (max-width:600px){
  .controls-grid{ grid-template-columns:1fr; }
}

/* ============================================================
   CARDS (tema igual ao painel principal)
   ============================================================ */
.card{
  background:var(--panel);
  border:1px solid rgba(255,255,255,0.05);
  border-radius:var(--radius);
  box-shadow:var(--shadow);
  padding:14px;
  display:flex;
  flex-direction:column;
  overflow:hidden;
}

/* ============================================================
   BOTÕES — "← Voltar"
   ============================================================ */
.btn-tab{
  background:rgba(255,255,255,0.03);
  border:1px solid rgba(255,255,255,0.05);
  color:var(--text)!important;
  padding:8px 16px;
  border-radius:10px;
  cursor:pointer;
  font-size:13px;
  font-weight:600;
  transition:all .24s ease;
}

.btn-tab:hover{
  background:rgba(89,240,200,0.15);
  border-color:var(--accent);
  color:var(--accent)!important;
  transform:translateY(-1px);
  box-shadow:0 0 14px rgba(89,240,200,0.35);
}

/* ============================================================
   DROPDOWNS — estilo mesmo do painel principal
   ============================================================ */
.retro-dropdown .Select-control{
  background:transparent!important;
  border:none!important;
  border-bottom:1px solid rgba(255,255,255,0.18)!important;
  border-radius:0!important;
  height:34px!important;
  box-shadow:none!important;
}

.retro-dropdown .Select-control:hover{
  border-bottom-color:var(--accent)!important;
}

/* Texto das opções */
.retro-dropdown .Select-option{
  color:#fff!important;
  background:rgba(15,17,20,0.98)!important;
  padding:8px 12px!important;
  font-size:13px!important;
}

.retro-dropdown .Select-option:hover{
  background:rgba(89,240,200,0.15)!important;
  color:var(--accent)!important;
}

/* Texto selecionado */
.retro-dropdown .Select-value-label{
  color:#fff!important;
  font-weight:600!important;
}

/* ============================================================
   AREA DE MÉTRICAS (aside)
   ============================================================ */
.aside{
  padding:14px;
  overflow-y:auto;
  max-height:calc(100vh - 220px)!important;
}

.aside .metric-row{
  display:grid;
  grid-template-columns:1fr auto;
  gap:6px;
  padding:6px 0;
  border-bottom:1px dashed rgba(255,255,255,0.06);
}

.aside .metric-row:last-child{
  border-bottom:none;
}

.metric-row .k{
  color:var(--muted);
  font-size:12px;
}

.metric-row .v{
  font-variant-numeric:tabular-nums;
}

/* BADGES */
.badge{
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding:4px 8px;
  font-size:12px;
  border-radius:999px;
  border:1px solid rgba(255,255,255,0.08);
  background:rgba(255,255,255,0.04);
}

.badge--pos{
  background:rgba(89,240,200,0.12);
  border-color:rgba(89,240,200,0.35);
  color:var(--positive);
}

.badge--neg{
  background:rgba(239,68,68,0.12);
  border-color:rgba(239,68,68,0.35);
  color:var(--negative);
}

.badge--neu{
  background:rgba(148,163,184,0.12);
  border-color:rgba(148,163,184,0.25);
  color:var(--muted);
}

/* ============================================================
   PLOTLY — fundo transparente + grid suave
   ============================================================ */
.dash-graph, .js-plotly-plot, .plot-container{
  background:transparent!important;
}

svg{ background:transparent!important; fill:none!important; }

.xgrid, .ygrid{
  stroke:rgba(255,255,255,0.05)!important;
}

/* ============================================================
   LOADING
   ============================================================ */
._dash-loading{
  background-color:var(--bg)!important;
  color:var(--accent)!important;
  font-family:Consolas, monospace!important;
}

._dash-loading::before{
  content:"Carregando..."!important;
  display:flex!important;
  justify-content:center!important;
  align-items:center!important;
  height:100vh!important;
  font-size:18px!important;
  color:var(--accent)!important;
}
.main-grid{
  display:grid !important;
  grid-template-columns: minmax(0,1fr) 360px !important;
  gap:16px !important;
}

.aside{
  max-height: calc(100vh - 200px) !important;
  overflow-y:auto !important;
}
</style>
"""


# ===============================================================
# === Factory: monta o sub-app em /app/beta/  (ÚNICO mount_beta)
# ===============================================================
def mount_beta(shared_server):
    """
    Monta o app Dash Beta/Alpha sob /app/beta/ no Flask `shared_server`.
    Força o tema Carbon Pro via asset físico + <link> no <head> com cache-buster.
    """
    import time as _time
    PREFIX = "/app/beta/"
    assets_dir = Path(__file__).parent / "assets"
    assets_dir.mkdir(exist_ok=True)

    # --- 1) Garante o CSS como asset físico (nome com 'z_' para carregar por último) ---
    def _write_css(filename: str, css_block: str):
        css = (css_block or "").strip()
        if css.startswith("<style>"): css = css[len("<style>"):]
        if css.endswith("</style>"):  css = css[:-len("</style>")]
        path = assets_dir / filename
        try:
            if not path.exists() or path.read_text(encoding="utf-8") != css:
                path.write_text(css, encoding="utf-8")
                print(f"[beta/assets] CSS escrito: {path}")
        except Exception as e:
            print(f"[beta/assets] Falha ao gravar {filename}: {e}")

    _write_css("z_carbon_pro.css", CARBON_PRO_CSS)

    # --- 2) Cria o sub-app (assets_url_path respeitando o prefixo) ---
    app = dash.Dash(
        name="beta_app",
        server=shared_server,
        requests_pathname_prefix=PREFIX,
        routes_pathname_prefix=PREFIX,
        assets_url_path=f"{PREFIX}assets",
        assets_folder=str(assets_dir),
        suppress_callback_exceptions=True,
        title="Análise de correlação",
        serve_locally=True,
        # Se quiser bloquear CSS legados, ative a linha abaixo. Por padrão, deixamos OFF.
        # assets_ignore="^(?!z_carbon_pro\\.css$).*",
    )

    # --- 3) Injeta <link> explícito no <head> com cache-buster ---
    try:
        buster = int(_time.time())
        link_tag = f'<link rel="stylesheet" href="{PREFIX}assets/z_carbon_pro.css?v={buster}">'
        if link_tag not in app.index_string:
            app.index_string = app.index_string.replace("</head>", link_tag + "</head>")
        # Fundo do body
        app.index_string = app.index_string.replace(
            "<body>", '<body style="background-color:#0B0C0D!important;">'
        )
    except Exception as e:
        print(f"[beta/index] Falha ao injetar link do CSS: {e}")

    # ------------------------ Layout ------------------------
    app.layout = html.Div(
        className="app",
        style={
            "height": "100vh", 
            "width": "100vw",
            "display": "flex",
            "flexDirection": "column",
            "backgroundColor": "var(--bg)",
            "overflow": "hidden",
        },
        children=[
            html.Div(
                className="beta-header",
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "gap": "14px",
                    "marginBottom": "16px"
                },
                children=[
                    # <<< CORRIGIDO AQUI: target="_top"
                    html.A(
                        html.Button("← Voltar", className="btn-tab back-btn"),
                        href="/app/",
                        target="_top"
                    ),
                    html.H2(
                        "Análise Beta / Alpha",
                        className="beta-title",
                        style={
                            "textAlign": "center",
                            "color": "var(--primary)",
                            "fontWeight": "600",
                            "fontSize": "26px",
                            "margin": "0",
                            "flex": "1"
                        },
                    ),
                    html.Div()
                ],
            ),
            html.Div(
                className="controls-grid",
                style={"marginBottom": "24px"},
                children=[
                    html.Div(className="field", children=[
                        html.Label("Ativo"),
                        dcc.Dropdown(id="ativo-dropdown", options=ativos_opcoes, value="ETH-USD", className="carbon-dropdown"),
                    ]),
                    html.Div(className="field", children=[
                        html.Label("Benchmark"),
                        dcc.Dropdown(id="benchmark-dropdown", options=ativos_opcoes, value="BTC-USD", className="carbon-dropdown"),
                    ]),
                    html.Div(className="field", children=[
                        html.Label("Frequência"),
                        dcc.Dropdown(
                            id="freq-dropdown",
                            options=[{"label": k, "value": v} for k, v in {"Diária": "1d", "Semanal": "1wk", "Mensal": "1mo"}.items()],
                            value="1wk", className="carbon-dropdown"
                        ),
                    ]),
                    html.Div(className="field", children=[
                        html.Label("Período"),
                        dcc.RadioItems(
                            id="periodo-radio",
                            options=[{"label": k, "value": v} for k, v in {"6M": "6mo", "YTD": "ytd", "1Y": "1y", "2Y": "2y", "5Y": "5y", "Max": "max"}.items()],
                            value="2y",
                            labelStyle={"display": "inline-block", "marginRight": "10px"},
                            inputStyle={"marginRight": "4px"},
                            className="dash-radio-items"
                        ),
                    ]),
                ],
            ),
            html.Div(
                className="main-grid",
                style={"flex": "1", "minHeight": 0},
                children=[
                    html.Div(className="graph-wrapper card", children=[
                        dcc.Graph(id="scatter-grafico", style={"height": "100%", "width": "100%"})
                    ]),
                    html.Div(id="metricas-beta", className="aside card"),
                ],
            ),
        ],
    )

    @app.callback(
        [Output('scatter-grafico', 'figure'),
         Output('metricas-beta', 'children')],
        [Input('ativo-dropdown', 'value'),
         Input('benchmark-dropdown', 'value'),
         Input('periodo-radio', 'value'),
         Input('freq-dropdown', 'value')]
    )
    def gerar_regressao(ativo, benchmark, periodo, frequencia):
        if not ativo or not benchmark:
            return go.Figure(), html.Div("Selecione um ativo e um benchmark.", className="card p-12")

        # ================== DOWNLOAD SÉRIES ==================
        s_ativo = baixar_serie(ativo, periodo, frequencia)
        s_bench = baixar_serie(benchmark, periodo, frequencia)

        if s_ativo.empty or s_bench.empty:
            fig_empty = go.Figure()
            fig_empty.update_layout(title="Sem dados suficientes", template="carbon_pro")
            return fig_empty, html.Div(
                "Não encontramos dados para a combinação escolhida.",
                className="card p-12"
            )

        # ================== ALINHAMENTO “ESTILO 3D” ==================
        # Igual à superfície: concatena, interpola e remove NaN
        df = pd.concat(
            [s_ativo.rename(ativo), s_bench.rename(benchmark)],
            axis=1
        )
        # Interpola buracos e joga fora linhas ainda problemáticas
        df = df.interpolate().dropna()

        # Se ainda tiver poucos pontos, aborta
        if len(df) < 10:
            fig_empty = go.Figure()
            fig_empty.update_layout(title="Sem dados suficientes", template="carbon_pro")
            return fig_empty, html.Div(
                "Poucos dados em comum entre o ativo e o benchmark "
                "para o período/frequência escolhidos.",
                className="card p-12"
            )

        # ================== PREPARO DOS DADOS ==================
        # Retornos (ainda usados para estatísticas)
        df['ret_ativo']  = df[ativo].pct_change()
        df['ret_bench']  = df[benchmark].pct_change()
        
        # Normalização (base 100) — usado para regressão e gráfico
        df['norm_ativo']  = df[ativo]     / df[ativo].iloc[0]     * 100
        df['norm_bench']  = df[benchmark] / df[benchmark].iloc[0] * 100
        
        df = df.dropna()
        n = len(df)
        
        # ================== REGRESSÃO (AGORA EM PREÇOS NORMALIZADOS) ==================
        X = df['norm_bench'].values.reshape(-1, 1)
        y = df['norm_ativo'].values                  

        reg = LinearRegression().fit(X, y) if n >= 2 else None
        if reg is not None:
            beta = float(reg.coef_[0])
            alpha = float(reg.intercept_)
            try:
                r2 = float(reg.score(X, y)) if n >= 2 else np.nan
            except Exception:
                r2 = np.nan
            try:
                y_pred = reg.predict(X)
            except Exception:
                y_pred = None
        else:
            beta = alpha = r2 = np.nan
            y_pred = None

        corr = _safe_corr(df['norm_bench'].values, df['norm_ativo'].values)
        p_value = _safe_normaltest(y)
        ad_text = _safe_anderson(y)

        # Melhor distribuição (Fitter)
        best_dist = "—"
        if _finite_nonconst(y, 10):
            try:
                f = Fitter(y, distributions=['norm', 't', 'laplace', 'lognorm', 'expon', 'gamma', 'beta'])
                try:
                    f.fit(progress=False, n_jobs=1, max_workers=1, prefer='threads')
                except TypeError:
                    f.fit(progress=False)
                best_dist = list(f.get_best(method='sumsquare_error').keys())[0]
            except Exception:
                best_dist = "—"

        # Lag ótimo
        melhor_lag, melhor_corr_lag, interpretacao = analisar_lag(df)

        # ================== FIGURA ==================
        fig = make_subplots(
            rows=4, cols=3,
            specs=[
                [None, None, None],
                [{"type": "histogram"}, {"type": "scatter"}, None],
                [None, {"type": "histogram"}, None],
                [None, {"type": "scatter"}, None]
            ],
            column_widths=[0.15, 0.85, 0.0],
            row_heights=[0.0, 0.9, 0.2, 0.5],
            vertical_spacing=0.04,
            horizontal_spacing=0.02
        )

        # Histogramas + KDE
        x_min = float(np.nanmin(df['ret_bench'])) if n else -1e-3
        x_max = float(np.nanmax(df['ret_bench'])) if n else  1e-3
        span = x_max - x_min
        if not np.isfinite(span) or span <= 0:
            x_min, x_max = -1e-3, 1e-3
        kde_x = np.linspace(x_min, x_max, 200)

        kde_y_bench = _safe_kde(df['ret_bench'].values, kde_x)
        kde_y_ativo = _safe_kde(df['ret_ativo'].values, kde_x)

        fig.add_trace(go.Histogram(
            y=df['norm_ativo'], nbinsy=30,
            marker=dict(color=ACCENT, line=dict(color=BG, width=1)),
            opacity=0.75, showlegend=False
        ), row=2, col=1)
        if kde_y_ativo is not None:
            fig.add_trace(go.Scatter(
                x=kde_y_ativo, y=kde_x, mode='lines',
                line=dict(color=PRIMARY, width=2), showlegend=False
            ), row=2, col=1)

        fig.add_trace(go.Histogram(
            x=df['norm_bench'], nbinsx=30,
            marker=dict(color=ACCENT, line=dict(color=BG, width=1)),
            opacity=0.75, showlegend=False
        ), row=3, col=2)
        if kde_y_bench is not None:
            fig.add_trace(go.Scatter(
                x=kde_x, y=kde_y_bench, mode='lines',
                line=dict(color=PRIMARY, width=2), showlegend=False
            ), row=3, col=2)

        # Dispersão + Regressão
        fig.add_trace(go.Scatter(
            x=df['norm_bench'], y=df['norm_ativo'],
            mode='markers',
            marker=dict(color=ACCENT, size=6, line=dict(color=PANEL_2, width=0.5)),
            name='Retornos'
        ), row=2, col=2)

        if y_pred is not None and np.isfinite(beta) and np.isfinite(alpha):
            fig.add_trace(go.Scatter(
                x=df['norm_bench'], y=y_pred, mode='lines',
                line=dict(color=PRIMARY, width=3),
                name=f"Y = {beta:.3f}X + {alpha:.3f}"
            ), row=2, col=2)

        # Linhas normalizadas (gráfico inferior)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['norm_ativo'], mode='lines',
            name=ativo, line=dict(color=ACCENT, width=2, dash='dot')
        ), row=4, col=2)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['norm_bench'], mode='lines',
            name=benchmark, line=dict(color=TEXT, width=2, dash='dot')
        ), row=4, col=2)

        if n > 0:
            hoje = pd.Timestamp.today().normalize()
            fig.add_vline(x=hoje, line_width=1.2, line_dash="dash", line_color=MUTED, row=4, col=2)

            ultimo_idx = df.index[-1]
            desloc = pd.Timedelta(days=10)
            fig.add_annotation(
                x=ultimo_idx + desloc, y=df['norm_ativo'].iloc[-1],
                text=f"<b>{ativo}</b>", showarrow=False, font=dict(color=ACCENT, size=11),
                bgcolor="rgba(0,0,0,0.45)", bordercolor=ACCENT, borderwidth=1, borderpad=3,
                xanchor="left", yanchor="middle", row=4, col=2
            )
            fig.add_annotation(
                x=ultimo_idx + desloc, y=df['norm_bench'].iloc[-1],
                text=f"<b>{benchmark}</b>", showarrow=False, font=dict(color=TEXT, size=11),
                bgcolor="rgba(0,0,0,0.45)", bordercolor=TEXT, borderwidth=1, borderpad=3,
                xanchor="left", yanchor="middle", row=4, col=2
            )

        if np.isfinite(corr):
            y_top = float(np.nanmax(df[['norm_ativo', 'norm_bench']].values)) * 1.08 if n > 0 else 100.0
            badge_bg = POSITIVE if corr >= 0.7 else (NEGATIVE if corr <= -0.7 else ACCENT)
            fig.add_annotation(
                x=df.index[0], y=y_top,
                text=f"<b>{corr:.4f}</b>",
                showarrow=False, font=dict(size=14, color="#000"),
                bordercolor=badge_bg, borderwidth=2, borderpad=4,
                bgcolor=badge_bg, opacity=1, row=4, col=2
            )

        fig.update_layout(
            title=f"Regressão de {ativo} vs {benchmark}",
            template="carbon_pro",
            showlegend=False,
            height=None
        )

        # ================== MÉTRICAS ==================
        def row(k, v):
            return html.Div(className="metric-row", children=[
                html.Span(k, className="k"),
                html.Span(v, className="v")
            ])

        corr_badge_class = (
            "badge badge--pos" if np.isfinite(corr) and corr >= 0.7
            else "badge badge--neg" if np.isfinite(corr) and corr <= -0.7
            else "badge badge--neu"
        )
        lag_sign = "atraso" if melhor_lag > 0 else ("adianto" if melhor_lag < 0 else "sem defasagem")

        r2_disp   = f"{r2:.4f}"   if np.isfinite(r2) else "—"
        p_disp    = f"{p_value:.4f}" if np.isfinite(p_value) else "—"
        beta_disp = f"{beta:.4f}" if np.isfinite(beta) else "—"
        alpha_disp = f"{alpha:.4f}" if np.isfinite(alpha) else "—"
        corr_disp  = f"{corr:.4f}" if np.isfinite(corr) else "—"

        metricas = html.Div(children=[
            html.H4("Estatísticas"),
            row("Alpha (intercepto)", alpha_disp),
            row("Beta (coef.)", beta_disp),
            row("R²", r2_disp),
            html.Div(
                [html.Span("Correlação"), html.Span(corr_disp, className=f"{corr_badge_class} v")],
                className="metric-row"
            ),
            row("Normalidade (D’Agostino p)", p_disp),
            row("Anderson-Darling", ad_text),
            row("Melhor distribuição", best_dist if best_dist != "—" else "—"),
            row("Observações", f"{n}"),
            html.Hr(),
            html.H4("Análise de Lag"),
            row("Lag ótimo", f"{melhor_lag} ({lag_sign})"),
            row(
                "Correlação (lag ótimo)",
                f"{melhor_corr_lag:.4f}" if np.isfinite(melhor_corr_lag) else "—"
            ),
            html.Div(
                interpretacao,
                style={"fontSize": "12.5px", "color": MUTED, "marginTop": "6px"}
            ),
        ])

        return fig, metricas


    print("[INFO] Beta/Alpha em /app/beta/ — CSS forçado via <link> + assets/z_carbon_pro.css")
    return app


# ===============================================================
# === Execução Direta (opcional para teste A/B local isolado)
#     -> Em produção/railway use apenas painel_inicial_2.py
# ===============================================================
if __name__ == "__main__":
    from flask import Flask
    dev_server = Flask(__name__)
    mount_beta(dev_server)
    port = int(os.environ.get("PORT", "8076"))
    dev_server.run(host="0.0.0.0", port=port, debug=True)
