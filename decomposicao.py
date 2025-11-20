# ===============================================================
# === GLOBALSTAT — EDA / Decomposição (Carbon Pro) como sub-app /eda/
# ===============================================================
import os
os.environ["DASH_IGNORE_JUPYTER"] = "1"

from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

import dash
from dash import Dash, html, dcc, Input, Output, State, no_update
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from scipy import stats
from string import Template
# === OpenAI (opcional) ===
from openai import OpenAI
_client = None
try:
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("✅ Hidra A.I.: online")
except Exception as e:
    print(f"⚠️ Hidra A.I.: offline ({e})")
    _client = None

# === Tema Carbon Pro (tokens) ===
BG = "#0B0C0D"
TEXT = "#E6F1EE"
PANEL      = "#0F1114"
PANEL_2    = "#12161B"
EDGE_COLOR = "#1E293B"
LINE       = EDGE_COLOR
MUTED      = "#9AA7A1"
PRIMARY    = "#C4FFE4"
ACCENT     = "#6FB4FF"
POSITIVE   = "#4ADE80"
NEGATIVE   = "#F87171"
WARNING    = "#FBBF24"
GRID_COLOR = "rgba(255,255,255,0.08)"

# === Template Plotly (Carbon Pro) ===
pio.templates["carbon_pro"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(
            color=TEXT,
            family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica Neue, Arial, Noto Sans, sans-serif"
        ),
        xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, linecolor=LINE, tickcolor=LINE, color=TEXT),
        yaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, linecolor=LINE, tickcolor=LINE, color=TEXT),
        margin=dict(l=40, r=24, t=56, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(color=TEXT)),
        colorway=[ACCENT, PRIMARY, POSITIVE, TEXT, WARNING, NEGATIVE],
    )
)
pio.templates.default = "carbon_pro"

CARBON_CSS = Template("""
<style>
:root {
  --bg:$BG; --panel:$PANEL; --panel-2:$PANEL_2; --line:$EDGE_COLOR;
  --text:$TEXT; --muted:$MUTED; --primary:$PRIMARY; --accent:$ACCENT;
  --positive:$POSITIVE; --negative:$NEGATIVE; --warning:$WARNING;
  --shadow:0 8px 24px rgba(0,0,0,.45); --radius:12px;
}
html,body{
  height:100%; margin:0; padding:0;
  background:var(--bg)!important; color:var(--text)!important;
  font-family:Inter,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans",sans-serif!important;
  overflow-x:hidden; -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale;
}
.card{
  background:linear-gradient(180deg,var(--panel),var(--panel-2));
  border:1px solid var(--line); border-radius:var(--radius); box-shadow:var(--shadow);
}
.card__title{
  font-size:12px; font-weight:700; color:var(--muted);
  padding:8px; border-bottom:1px solid rgba(255,255,255,.06);
}
.dash-graph, .dash-graph-container, .plot-container, .js-plotly-plot{ background:transparent!important; }
.gs-header{ display:flex; align-items:center; justify-content:space-between; gap:14px; margin:0 0 8px 0; }
.gs-title{ flex:1; text-align:center; }
.btn-tab{
  background:var(--panel-2); border:1px solid var(--line); color:var(--text);
  padding:8px 16px; font-size:13.5px; border-radius:10px; cursor:pointer;
  transition:all .25s ease; box-shadow:var(--shadow);
}
.btn-tab:hover{ background:var(--accent); color:#fff; box-shadow:0 0 12px var(--accent); transform:translateY(-1px); }
.btn-tab:active{ transform:scale(.97); opacity:.9; }

/* Loader / anti-flash */
#_dash-loading { position:fixed; inset:0; background:$BG!important; z-index:99999; }
.dash-loading, .dash-spinner { background-color:$BG!important; color:$TEXT!important; }
.dash-loading .js-plotly-plot, .dash-spinner .js-plotly-plot,
.dash-loading .plot-container, .dash-spinner .plot-container { background-color:$BG!important; }
.dash-loading .js-plotly-plot svg rect.bg, .dash-spinner .js-plotly-plot svg rect.bg { fill:$BG!important; }

/* Campos do chat */
#user-input{
  background:linear-gradient(180deg,var(--panel),var(--panel-2));
  border:1px solid var(--line); color:var(--text); border-radius:8px; box-shadow:none;
}
#send-btn{
  background:linear-gradient(180deg, rgba(111,255,233,0.15), rgba(59,255,176,0.12));
  border:1px solid rgba(111,255,233,0.35); color:#CDEDE8; border-radius:10px; font-weight:600; font-size:18px;
}

/* Quick range */
button.quick{
  background:linear-gradient(180deg,rgba(255,255,255,.04),rgba(255,255,255,.015));
  color:var(--text); border:1px solid rgba(255,255,255,.08); border-radius:8px; padding:4px 10px;
  cursor:pointer; box-shadow:0 4px 12px rgba(0,0,0,.4); transition:all .18s ease-in-out;
}
button.quick:hover{
  transform:translateY(-1px);
  background:linear-gradient(180deg,rgba(111,255,233,.08),rgba(111,255,233,.05));
  box-shadow:0 6px 18px rgba(111,255,233,.15); border-color:rgba(111,255,233,.25);
}

/* Card do chat + neutralização do antigo #chat-history */
.chat-card{
  background: linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.03));
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: inset 0 0 6px rgba(111,255,233,.08);
  display: flex; flex-direction: column;
}
.chat-card__title{
  padding: 8px 10px; font-weight: 700; color: var(--muted);
  border-bottom: 1px solid rgba(255,255,255,.06); font-size: 14px;
}
.chat-card__body{ flex:1 1 auto; padding:12px; overflow-y:auto; background:transparent; }
#chat-history{ border:none; background:transparent; padding:0; box-shadow:none; }
</style>
<script>
document.addEventListener("DOMContentLoaded", () => {
  const bg = "$BG";
  function recolorPlots(){
    document.querySelectorAll('.js-plotly-plot').forEach(plot => {
      const svg = plot.querySelector('svg'); if(!svg) return;
      svg.style.backgroundColor = bg; const rect = svg.querySelector('rect.bg');
      if(rect) rect.setAttribute('fill', bg);
    });
  }
  if(window.Plotly){ window.Plotly.on('plotly_afterplot', () => recolorPlots()); }
  setTimeout(recolorPlots, 300);
});
</script>
""").substitute(
    BG=BG, TEXT=TEXT, PANEL=PANEL, PANEL_2=PANEL_2, EDGE_COLOR=EDGE_COLOR,
    MUTED=MUTED, PRIMARY=PRIMARY, ACCENT=ACCENT, POSITIVE=POSITIVE,
    NEGATIVE=NEGATIVE, WARNING=WARNING
)


# ==== Yahoo Chart API (resiliente) ====
import requests
import time

YCHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YCHART_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}
YCACHE_TTL = 300  # segundos (5 min)

# cache simples em memória: {key: (timestamp, DataFrame)}
_ycache = {}

def _cache_key(symbol, start, end, interval):
    return f"{symbol}|{start.isoformat()}|{end.isoformat()}|{interval}"

def _cache_get(key):
    item = _ycache.get(key)
    if not item:
        return None
    ts, df = item
    if time.time() - ts > YCACHE_TTL:
        _ycache.pop(key, None)
        return None
    return df.copy()

def _cache_set(key, df):
    _ycache[key] = (time.time(), df.copy())

def _yahoo_chart_range(symbol: str, start: dt.date, end: dt.date, interval: str = "1d", max_tries: int = 4) -> pd.DataFrame:
    """
    Puxa OHLC via Yahoo Chart API por data (period1/period2) com backoff + cache curto (TTL=5min).
    Retorna DataFrame com colunas ['Open','High','Low','Close','Adj Close','Volume'] quando possível;
    caso contrário, ao menos 'Close'. Fallback fica no _safe_download.
    """
    key = _cache_key(symbol, start, end, interval)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        p1 = int(time.mktime(dt.datetime(start.year, start.month, start.day, 0, 0).timetuple()))
        p2 = int(time.mktime(dt.datetime(end.year,   end.month,   end.day,   23, 59).timetuple()))
    except Exception:
        # fallback simples de datas (caso algum tz/local falhe)
        p1 = int(time.time()) - 86400 * 365 * 2
        p2 = int(time.time())

    params = {
        "period1": p1,
        "period2": p2,
        "interval": interval,                     # '1d','1wk','1mo'
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }

    delay = 0.6
    for attempt in range(1, max_tries + 1):
        try:
            r = requests.get(
                YCHART_URL.format(symbol=symbol),
                params=params,
                headers=YCHART_HEADERS,
                timeout=8,
            )
            # Trate busy/limiter como erro recuperável
            if r.status_code in (429, 502, 503, 504):
                raise RuntimeError(f"yahoo_busy:{r.status_code}")
            r.raise_for_status()

            js = r.json()
            res = js.get("chart", {}).get("result", [])
            if not res:
                raise ValueError("empty_result")

            r0 = res[0]
            ts = r0.get("timestamp", [])
            ind = r0.get("indicators", {})
            if not ts or not ind:
                raise ValueError("no_ts_or_indicators")

            idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert("UTC").tz_localize(None)

            quote = (ind.get("quote") or [{}])[0]
            data = {}
            if quote.get("open")   is not None: data["Open"]  = quote["open"]
            if quote.get("high")   is not None: data["High"]  = quote["high"]
            if quote.get("low")    is not None: data["Low"]   = quote["low"]
            if quote.get("close")  is not None: data["Close"] = quote["close"]
            if quote.get("volume") is not None: data["Volume"] = quote["volume"]

            adjc = (ind.get("adjclose") or [{}])
            adjc = adjc[0].get("adjclose") if adjc else None
            if adjc is not None:
                data["Adj Close"] = adjc

            if not data:
                raise ValueError("no_price_fields")

            df = pd.DataFrame(data, index=idx).astype(float)
            df = df.dropna(how="all")
            if not df.empty:
                _cache_set(key, df)
            return df

        except Exception:
            if attempt >= max_tries:
                return pd.DataFrame()
            time.sleep(delay)
            delay *= 1.8  # backoff exponencial leve

    return pd.DataFrame()

# ===============================================================
# Helpers: download & series
# ===============================================================
def _safe_download(symbol, start, end):
    """
    Estratégia de download:
      1) Yahoo Chart API (resiliente, com cache curto).
      2) Fallback: yfinance com janela start/end.
      3) Fallback final: yfinance por 'period' heurístico.
    """
    # 1) Yahoo Chart API
    df = _yahoo_chart_range(symbol, start, end, interval="1d")
    if df is not None and not df.empty:
        return df

    # 2) yfinance com start/end
    try:
        df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False, threads=False)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    # 3) yfinance por period heurístico
    try:
        delta = (end - start).days if isinstance(end, dt.date) and isinstance(start, dt.date) else 730
        if   delta > 1825: period = "5y"
        elif delta > 1095: period = "3y"
        elif delta > 730:  period = "2y"
        elif delta > 365:  period = "1y"
        elif delta > 182:  period = "6mo"
        elif delta > 90:   period = "3mo"
        else:              period = "1mo"

        df = yf.download(symbol, period=period, auto_adjust=False, progress=False, threads=False)
    except Exception:
        df = pd.DataFrame()

    return df if df is not None else pd.DataFrame()


def _ensure_series_close(df):
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return pd.Series(dtype="float64", name="Close")

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            s = df["Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
        else:
            s = df.iloc[:, 0]
    else:
        s = df["Close"] if "Close" in df.columns else df.iloc[:, 0]

    s = pd.to_numeric(s, errors="coerce").dropna()
    try:
        s.index = pd.to_datetime(df.index)
    except Exception:
        pass
    return s

def best_distribution_ks(data):
    dists = {
        "Normal": stats.norm,
        "Lognormal": stats.lognorm,
        "Exponencial": stats.expon,
        "Laplace": stats.laplace,
        "T-Student": stats.t
    }
    best = {"name": None, "stat": np.inf, "p": 0.0, "params": None}
    for name, dist in dists.items():
        try:
            params = dist.fit(data)
            stat, p = stats.kstest(data, dist.name, params)
            if stat < best["stat"] or (np.isclose(stat, best["stat"]) and p > best["p"]):
                best.update({"name": name, "stat": stat, "p": p, "params": params})
        except Exception:
            continue
    return best["name"], best["stat"], best["p"], best["params"]

# ------------------ AJUSTE AUTOMÁTICO: quantis, k e renko ------------------
def _theoretical_sigma(dist_name, params):
    try:
        if dist_name == "Normal":
            mu, sigma = params[0], params[1]
            return sigma
        if dist_name == "Laplace":
            loc, b = params
            return np.sqrt(2.0) * b
        if dist_name == "T-Student":
            dfv, loc, scale = params
            return np.nan if dfv <= 2 else scale * np.sqrt(dfv / (dfv - 2.0))
    except Exception:
        pass
    return np.nan

def _q_halfwidth(dist_name, params, alpha):
    """retorna half-width q_alpha (valor positivo) da dist ajustada."""
    p = 0.5 * (1.0 + float(alpha))
    try:
        if dist_name == "Normal":
            loc, scale = params[0], params[1]
            return float(stats.norm.ppf(p, loc, scale) - loc)
        if dist_name == "Laplace":
            loc, b = params
            return float(stats.laplace.ppf(p, loc, b) - loc)
        if dist_name == "T-Student":
            dfv, loc, scale = params
            return float(stats.t.ppf(p, dfv, loc, scale) - loc)
    except Exception:
        pass
    return np.nan

def _k_from(alpha, dist_name, params):
    q = _q_halfwidth(dist_name, params, alpha)
    sig = _theoretical_sigma(dist_name, params)
    return (q / sig if np.isfinite(q) and np.isfinite(sig) and sig > 0 else np.nan, q, sig)

def _renko_brick_to_log(brick_type, brick_value, last_price):
    """brick_type in {'percent','abs'}; brick_value ex.: 0.005 (0.5%) ou 10.0 (R$10)"""
    if brick_type == "percent":
        return float(np.log1p(float(brick_value)))
    # absoluto em preço
    brick_value = float(brick_value)
    last_price  = float(last_price if last_price else 0.0)
    if last_price <= 0:
        return np.nan
    return float(np.log1p(brick_value / last_price))

# ===============================================================
# Builders (figuras)
# ===============================================================
def build_combo(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    s = _ensure_series_close(df)
    r = np.log(s / s.shift(1)).dropna()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.08,
                        subplot_titles=("Série Original (Candlestick)", "Retorno Logarítmico"))

    if all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            increasing_line_color=POSITIVE, decreasing_line_color=NEGATIVE,
            increasing_fillcolor="rgba(74,222,128,0.20)", decreasing_fillcolor="rgba(248,113,113,0.20)",
            name="Preço"
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=s.index, y=s, mode="lines", line=dict(color=ACCENT, width=2), name="Preço"),
                      row=1, col=1)

    fig.add_trace(go.Scatter(x=r.index, y=r, mode="lines", line=dict(color=PRIMARY, width=1.5), name="Retorno log"),
                  row=2, col=1)
    fig.add_hline(y=0, line_color=MUTED, line_width=1, opacity=0.7, row=2, col=1)

    fig.update_layout(template="carbon_pro", height=400, showlegend=False)
    return fig

def build_histogram(s):
    r = np.log(s / s.shift(1)).dropna()
    desc = {"Mínimo": s.min(), "Máximo": s.max(), "Média": s.mean(), "Mediana": s.median(), "Desvio Padrão": s.std()}

    resumo_txt = (
        f"<b>Resumo Estatístico</b><br>"
        f"Mínimo: {desc['Mínimo']:,.4f}<br>"
        f"Máximo: {desc['Máximo']:,.4f}<br>"
        f"Média: {desc['Média']:,.4f}<br>"
        f"Mediana: {desc['Mediana']:,.4f}<br>"
        f"Desvio Padrão: {desc['Desvio Padrão']:,.4f}"
    )

    fig = go.Figure(go.Histogram(x=r, nbinsx=60, marker=dict(color=PRIMARY, line=dict(color=BG, width=0.5))))
    fig.add_annotation(
        text=resumo_txt, xref="paper", yref="paper", x=0.97, y=0.95, xanchor="right", yanchor="top",
        showarrow=False, align="left",
        font=dict(color=TEXT, size=12, family="Consolas, monospace"),
        bordercolor=EDGE_COLOR, borderwidth=1, borderpad=8, bgcolor=PANEL_2, opacity=0.9
    )
    fig.update_layout(template="carbon_pro", height=400, title="Histograma dos Retornos Logarítmicos",
                      xaxis_title="Retorno (log)", yaxis_title="Frequência")
    return fig

def build_boxplot(series):
    s = series[series > 0]
    s = np.log(s / s.shift(1)).dropna()

    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = s[(s < lb) | (s > ub)]

    fig = go.Figure()
    fig.add_trace(go.Box(
        x=["Retornos"] * len(s),
        y=s,
        boxpoints='all', jitter=0.4, pointpos=-1.8,
        marker=dict(size=4, color=ACCENT, opacity=0.75, line=dict(width=0)),
        line=dict(color=PRIMARY, width=2),
        fillcolor="rgba(196,255,228,0.10)",
        whiskerwidth=0.8, width=0.5, name="Retornos"
    ))
    fig.add_trace(go.Scatter(
        x=["Retornos"] * len(outliers),
        y=outliers, mode='markers',
        marker=dict(size=9, color=['rgba(204,255,51,0.25)' if v > ub else 'rgba(179,0,89,0.25)' for v in outliers],
                    line=dict(width=0)),
        hoverinfo='skip', showlegend=False
    ))
    fig.update_layout(
        template="carbon_pro", height=400, title="Boxplot dos Retornos Logarítmicos",
        yaxis_title="Retorno (log)", showlegend=False, margin=dict(l=40, r=40, t=60, b=60), boxmode='group'
    )
    fig.add_annotation(
        text=f"IQR: {iqr:.6f} | Outliers: {len(outliers)}",
        xref="paper", yref="paper", x=0.5, y=-0.12, showarrow=False,
        font=dict(color=TEXT, size=12, family="Consolas, monospace")
    )
    return fig

def gerar_comentario_boxplot(series: pd.Series) -> str:
    s = series[series > 0]
    r = np.log(s / s.shift(1)).dropna()
    if len(r) < 10:
        return "Amostra muito pequena para interpretar o boxplot com confiança."

    q1, q3 = r.quantile(0.25), r.quantile(0.75)
    iqr = q3 - q1
    med = r.median()
    mean = r.mean()
    std = r.std(ddof=1)
    skew = r.skew()
    ek = r.kurt()

    lb, ub = q1 - 1.5*iqr, q3 + 1.5*iqr
    outliers = r[(r < lb) | (r > ub)]
    out_pct = 100 * len(outliers) / len(r)

    skew_txt = "assimetria positiva (cauda à direita)" if skew > 0.5 else ("assimetria negativa (cauda à esquerda)" if skew < -0.5 else "baixa assimetria")
    kurt_txt = "caudas pesadas (mais eventos extremos que o normal)" if ek > 1.0 else ("caudas leves (menos extremos que o normal)" if ek < -0.5 else "curtose próxima do normal")

    vol_idx = (std / iqr) if iqr > 0 else np.inf
    vol_txt = "dispersão baixa" if vol_idx < 0.8 else ("dispersão moderada" if vol_idx < 1.5 else "dispersão alta")

    trend_txt = "mediana próxima de zero (tendência diária neutra)" if abs(med) < 0.001 else ("mediana positiva (tendência ligeiramente altista)" if med > 0 else "mediana negativa (tendência ligeiramente baixista)")

    if out_pct >= 10: out_txt = f"muitos outliers ({out_pct:.1f}% da amostra)"
    elif out_pct >= 3: out_txt = f"alguns outliers ({out_pct:.1f}% da amostra)"
    else: out_txt = f"poucos outliers ({out_pct:.1f}% da amostra)"

    comentario = (
        f"O boxplot indica {vol_txt} e {skew_txt}. "
        f"A {kurt_txt}. {trend_txt}. "
        f"Foram detectados {out_txt}, coerentes com volatilidade "
        f"{'elevada' if vol_idx >= 1.5 or out_pct >= 10 else 'normal a moderada'}."
    )
    comentario += f" (IQR≈{iqr:.4f}, média≈{mean:.4f}, mediana≈{med:.4f})."
    return comentario

def build_decomp(series):
    s = pd.Series(series).dropna()
    if len(s) < 60:
        fig = go.Figure()
        fig.update_layout(template="carbon_pro", height=360,
                          title=f"Decomposição indisponível (≥ 60 pontos, atual = {len(s)})")
        fig.add_annotation(text="Amplie o período (ex.: 1 ano+) ou use ACF/PACF/Diff.",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    s = s.copy().interpolate(limit=10)
    try:
        dec = seasonal_decompose(s, model="additive", period=30)
    except Exception:
        fig = go.Figure()
        fig.update_layout(template="carbon_pro", height=360, title="Decomposição indisponível para esta série/período")
        fig.add_annotation(text="Tente aumentar a janela ou usar ACF/PACF/Diferenças.",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    parts = {"Nível": dec.observed, "Tendência": dec.trend, "Sazonalidade": dec.seasonal, "Resíduo": dec.resid}
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=list(parts.keys()))
    colors = [PRIMARY, ACCENT, POSITIVE, TEXT]
    for i, (k, v) in enumerate(parts.items(), 1):
        fig.add_trace(go.Scatter(x=s.index, y=v, mode="lines", line=dict(color=colors[i - 1], width=2)), row=i, col=1)
    fig.update_layout(template="carbon_pro", height=360, showlegend=False, title="Decomposição Aditiva da Série")
    return fig

def build_acf_pacf(series):
    s = series.dropna()
    acf_vals = acf(s, nlags=40)
    pacf_vals = pacf(s, nlags=40)
    n = len(s); conf = 1.96 / np.sqrt(n); lags = np.arange(len(acf_vals))

    acf_colors  = [PRIMARY if abs(v) <= conf else WARNING for v in acf_vals]
    pacf_colors = [PRIMARY if abs(v) <= conf else WARNING for v in pacf_vals]

    fig = make_subplots(rows=2, cols=1, subplot_titles=("ACF", "PACF"))
    fig.add_trace(go.Bar(x=lags, y=acf_vals, marker_color=acf_colors), row=1, col=1)
    fig.add_hline(y= conf, line_color=MUTED, line_width=1, row=1, col=1)
    fig.add_hline(y=-conf, line_color=MUTED, line_width=1, row=1, col=1)

    fig.add_trace(go.Bar(x=lags, y=pacf_vals, marker_color=pacf_colors), row=2, col=1)
    fig.add_hline(y= conf, line_color=MUTED, line_width=1, row=2, col=1)
    fig.add_hline(y=-conf, line_color=MUTED, line_width=1, row=2, col=1)

    fig.update_layout(template="carbon_pro", height=360, showlegend=False,
                      title="ACF / PACF com Intervalos de Confiança")
    fig.add_annotation(text="Lags significativos em destaque (|r| > 1.96/√N)",
                       xref="paper", yref="paper", x=0.5, y=-0.12, showarrow=False)
    return fig

def build_diff(series):
    diff = series.diff().dropna()
    fig = go.Figure(go.Scatter(x=diff.index, y=diff, mode="lines", line=dict(color=ACCENT, width=2)))
    fig.update_layout(template="carbon_pro", height=360, showlegend=False, title="Série Diferenciada (1ª Ordem)")
    return fig

def build_diff2(series):
    diff2 = series.diff().diff().dropna()
    fig = go.Figure(go.Scatter(x=diff2.index, y=diff2, mode="lines", line=dict(color=PRIMARY, width=2)))
    fig.update_layout(template="carbon_pro", height=360, showlegend=False, title="Série Diferenciada (2ª Ordem)")
    return fig

# ===============================================================
# Factory: monta o sub-app em /eda/
# ===============================================================
def mount_eda(shared_server):
    assets_dir = Path(__file__).parent / "assets"
    app = Dash(
        __name__,
        server=shared_server,
        requests_pathname_prefix="/app/eda/",
        routes_pathname_prefix="/app/eda/",
        suppress_callback_exceptions=True,
        title="Análise exploratória"
    )

    # === Corrige o fundo preto no iframe (tema dark visível)
    app.index_string = app.index_string.replace(
        "<body>", '<body style="background-color:#0B0C0D!important;">'
    )

    # === Injeta CSS Carbon Pro ===
    app.index_string = app.index_string.replace("</head>", CARBON_CSS + "</head>")

    # === Layout ===
    app.layout = html.Div(
        style={"backgroundColor": BG, "color": TEXT, "padding": "16px"},
        children=[
            # HEADER: voltar | [ativo + períodos] | título central | espaçador direito
            html.Div(
                className="gs-header",
                style={"alignItems": "center", "marginBottom": "10px"},
                children=[
                    html.A(html.Button("← Voltar", className="btn-tab back-btn"), href="/"),

                    # ESQUERDA: dropdown do ativo + botões de período
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "10px"},
                        children=[
                            dcc.Dropdown(
                                id='symbol-dropdown',
                                options=[
                                    {'label': 'Crypto • Bitcoin (BTC-USD)', 'value': 'BTC-USD'},
                                    {'label': 'Crypto • Ethereum (ETH-USD)', 'value': 'ETH-USD'},
                                    {'label': 'Commodity • Gold (GC=F)', 'value': 'GC=F'},
                                    {'label': 'Equity Index • S&P 500 (^GSPC)', 'value': '^GSPC'},
                                    {'label': 'ETF • SPY (S&P500)', 'value': 'SPY'},
                                    {'label': 'ETF • QQQ (Nasdaq)', 'value': 'QQQ'},
                                    {'label': 'ETF • TLT (Long Bonds)', 'value': 'TLT'},
                                    {'label': 'ETF • EWZ (Brazil)', 'value': 'EWZ'},
                                ],
                                value='BTC-USD', clearable=False, style={'width': '320px'}
                            ),
                            html.Div(
                                children=[
                                    html.Button("5 anos", id="btn-5y", className="quick"),
                                    html.Button("3 anos", id="btn-3y", className="quick"),
                                    html.Button("2 anos", id="btn-2y", className="quick"),
                                    html.Button("1 ano", id="btn-1y", className="quick"),
                                    html.Button("6 meses", id="btn-6m", className="quick"),
                                    html.Button("3 meses", id="btn-3m", className="quick"),
                                    html.Button("1 mês", id="btn-1mo", className="quick"),
                                ],
                                style={'display': 'flex', 'gap': '6px', 'alignItems': 'center'}
                            )
                        ],
                    ),

                    # CENTRO: título
                    html.Div(
                        style={"flex": 1, "display": "flex", "justifyContent": "center"},
                        children=[
                            html.H3(
                                "Decomposição / EDA — GlobalStat",
                                className="gs-title",
                                style={"color": TEXT, "margin": 0, "fontWeight": "600"}
                            )
                        ]
                    ),

                    # DIREITA: espaçador para balancear o header
                    html.Div(style={"width": "240px"})
                ],
            ),

            # Stores
            dcc.Store(id="range-active", data="btn-2y"),
            dcc.Store(id="returns-store"),
            dcc.Store(id="fit-store"),
            dcc.Store(id="last-price"),

            # Linha: chat (esq) | conteúdo (dir)
            html.Div(
                style={'display': 'flex', 'gap': '1%'},
                children=[
                    # --- CHAT (coluna esquerda) ---
                    html.Div(
                        style={'flex': '0.42', 'marginRight': '1.2%', 'display': 'flex', 'flexDirection': 'column', 'height': '82vh'},
                        children=[
                            # card do chat com título embutido (topo do card = topo do box)
                            html.Div(
                                className="chat-card",
                                style={'flex': '1 1 auto', 'display': 'flex', 'flexDirection': 'column'},
                                children=[
                                    html.Div("Hidra A.I.", className="chat-card__title"),
                                    html.Div(
                                        id='chat-history',
                                        className="chat-card__body",
                                        style={
                                            'flex': '1 1 auto', 'overflowY': 'auto',
                                            'fontFamily': 'Consolas, monospace', 'color': TEXT, 'fontSize': '14px'
                                        }
                                    ),
                                ]
                            ),
                            html.Div(
                                style={'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'marginTop': '8px'},
                                children=[
                                    dcc.Input(
                                        id='user-input', type='text', placeholder='Converse com o Hidra A.I.…',
                                        style={
                                            'flex': 1, 'height': '44px', 'backgroundColor': '#101418',
                                            'color': TEXT, 'border': f'1px solid {EDGE_COLOR}',
                                            'padding': '0 12px', 'borderRadius': '8px', 'fontSize': '15px'
                                        }
                                    ),
                                    html.Button("↑", id='send-btn', n_clicks=0, className="btn-tab",
                                                style={'height': '44px', 'padding': '0 14px'})
                                ]
                            )
                        ]
                    ),

                    # COLUNA DIREITA (grid 3× + sidebar do Ajuste Automático à direita)
                    html.Div(
                        style={'flex': '1.58', 'display': 'flex', 'flexDirection': 'column', 'minHeight': '82vh'},
                        children=[
                            # uma linha: [GRID 3 blocos] | [SIDEBAR Ajuste Automático]
                            html.Div(
                                style={
                                    'display': 'flex',
                                    'gap': '12px',
                                    'flex': '1 1 auto',
                                    'minHeight': 0,
                                    'height': '82vh',       # igual ao chat
                                    'alignItems': 'stretch'     # alinha topo com o chat
                                },
                                children=[
                                    # === GRID (3 blocos) ===
                                    html.Div(
                                        style={
                                            'display': 'flex',
                                            'gap': '0.9%',
                                            'alignItems': 'stretch',
                                            'flex': '1 1 auto',
                                            'minWidth': 0,
                                            'height': '100%'     # ocupa toda a altura disponível
                                        },
                                        children=[
                                            # ----- Bloco 1 -----
                                            html.Div(
                                                style={
                                                    'flex': '1 1 0',
                                                    'display': 'flex',
                                                    'flexDirection': 'column',
                                                    'minWidth': 0,
                                                    'height': '100%'
                                                },
                                                children=[
                                                    dcc.Graph(
                                                        id='graph-1',
                                                        config={'displayModeBar': False, 'displaylogo': False, 'scrollZoom': False},
                                                        style={
                                                            'flex': '1 1 auto',
                                                            'minHeight': '0',
                                                            'height': 'auto',
                                                            'border': f'1px solid {EDGE_COLOR}',
                                                            'borderRadius': '8px'
                                                        }
                                                    ),
                                                    html.Div(
                                                        id='info-box-1',
                                                        style={
                                                            'flex': '0 0 auto',
                                                            'backgroundColor': PANEL_2,
                                                            'border': f'1px solid {EDGE_COLOR}',
                                                            'borderRadius': '8px',
                                                            'padding': '8px',
                                                            'fontFamily': 'Consolas, monospace',
                                                            'color': TEXT,
                                                            'fontSize': '13px',
                                                            'marginTop': '6px',
                                                            'maxHeight': '24vh',
                                                            'overflowY': 'auto'
                                                        }
                                                    ),
                                                    dcc.Dropdown(
                                                        id='slot-type-1',
                                                        options=[
                                                            {'label': 'Série + Retornos', 'value': 'combo'},
                                                            {'label': 'Histograma (ret. log)', 'value': 'hist'},
                                                            {'label': 'Boxplot (ret. log)', 'value': 'box'},
                                                            {'label': 'Decomposição', 'value': 'decomp'},
                                                            {'label': 'ACF/PACF', 'value': 'acf_pacf'},
                                                            {'label': 'Diff (1ª)', 'value': 'diff'},
                                                            {'label': 'Diff (2ª)', 'value': 'diff2'},
                                                        ],
                                                        value='combo', clearable=False, style={'marginTop': '6px'}
                                                    ),
                                                ]
                                            ),

                                            # ----- Bloco 2 -----
                                            html.Div(
                                                style={
                                                    'flex': '1 1 0',
                                                    'display': 'flex',
                                                    'flexDirection': 'column',
                                                    'minWidth': 0,
                                                    'height': '100%'
                                                },
                                                children=[
                                                    dcc.Graph(
                                                        id='graph-2',
                                                        config={'displayModeBar': False, 'displaylogo': False, 'scrollZoom': False},
                                                        style={
                                                            'flex': '1 1 auto',
                                                            'minHeight': '0',
                                                            'height': 'auto',
                                                            'border': f'1px solid {EDGE_COLOR}',
                                                            'borderRadius': '8px'
                                                        }
                                                    ),
                                                    html.Div(
                                                        id='info-box-2',
                                                        style={
                                                            'flex': '0 0 auto',
                                                            'backgroundColor': PANEL_2,
                                                            'border': f'1px solid {EDGE_COLOR}',
                                                            'borderRadius': '8px',
                                                            'padding': '8px',
                                                            'fontFamily': 'Consolas, monospace',
                                                            'color': TEXT,
                                                            'fontSize': '13px',
                                                            'marginTop': '6px',
                                                            'maxHeight': '24vh',
                                                            'overflowY': 'auto'
                                                        }
                                                    ),
                                                    dcc.Dropdown(
                                                        id='slot-type-2', value='hist', clearable=False,
                                                        options=[
                                                            {'label': 'Série + Retornos', 'value': 'combo'},
                                                            {'label': 'Histograma (ret. log)', 'value': 'hist'},
                                                            {'label': 'Boxplot (ret. log)', 'value': 'box'},
                                                            {'label': 'Decomposição', 'value': 'decomp'},
                                                            {'label': 'ACF/PACF', 'value': 'acf_pacf'},
                                                            {'label': 'Diff (1ª)', 'value': 'diff'},
                                                            {'label': 'Diff (2ª)', 'value': 'diff2'},
                                                        ],
                                                        style={'marginTop': '6px'}
                                                    ),
                                                ]
                                            ),

                                            # ----- Bloco 3 -----
                                            html.Div(
                                                style={
                                                    'flex': '1 1 0',
                                                    'display': 'flex',
                                                    'flexDirection': 'column',
                                                    'minWidth': 0,
                                                    'height': '100%'
                                                },
                                                children=[
                                                    dcc.Graph(
                                                        id='graph-3',
                                                        config={'displayModeBar': False, 'displaylogo': False, 'scrollZoom': False},
                                                        style={
                                                            'flex': '1 1 auto',
                                                            'minHeight': '0',
                                                            'height': 'auto',
                                                            'border': f'1px solid {EDGE_COLOR}',
                                                            'borderRadius': '8px'
                                                        }
                                                    ),
                                                    html.Div(
                                                        id='info-box-3',
                                                        style={
                                                            'flex': '0 0 auto',
                                                            'backgroundColor': PANEL_2,
                                                            'border': f'1px solid {EDGE_COLOR}',
                                                            'borderRadius': '8px',
                                                            'padding': '8px',
                                                            'fontFamily': 'Consolas, monospace',
                                                            'color': TEXT,
                                                            'fontSize': '13px',
                                                            'marginTop': '6px',
                                                            'maxHeight': '24vh',
                                                            'overflowY': 'auto'
                                                        }
                                                    ),
                                                    dcc.Dropdown(
                                                        id='slot-type-3', value='box', clearable=False,
                                                        options=[
                                                            {'label': 'Série + Retornos', 'value': 'combo'},
                                                            {'label': 'Histograma (ret. log)', 'value': 'hist'},
                                                            {'label': 'Boxplot (ret. log)', 'value': 'box'},
                                                            {'label': 'Decomposição', 'value': 'decomp'},
                                                            {'label': 'ACF/PACF', 'value': 'acf_pacf'},
                                                            {'label': 'Diff (1ª)', 'value': 'diff'},
                                                            {'label': 'Diff (2ª)', 'value': 'diff2'},
                                                        ],
                                                        style={'marginTop': '6px'}
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),

                                    # === SIDEBAR: Ajuste Automático (largura fixa) ===
                                    html.Div(
                                        style={
                                            'flex': '0 0 380px', 'width': '380px',
                                            'display': 'flex', 'flexDirection': 'column', 'gap': '10px',
                                            'alignSelf': 'flex-start', 'position': 'sticky', 'top': '72px'
                                        },
                                        children=[
                                            html.Div(
                                                className="card",
                                                style={'padding': '10px 12px', 'borderRadius': '10px'},
                                                children=[
                                                    html.Div("Ajuste Automático — Bandas", className="card__title"),
                                    
                                                    # 🔽 CONTROLES EM COLUNA (VERTICAL)
                                                    html.Div(
                                                        style={'display': 'flex', 'flexDirection': 'column', 'gap': '10px'},
                                                        children=[
                                    
                                                            # Confiança
                                                            html.Div([
                                                                html.Div("Confiança", style={'color': EDGE_COLOR, 'fontWeight': 700, 'fontSize': '12px', 'marginBottom': '4px'}),
                                                                dcc.Dropdown(
                                                                    id='aa-alpha',
                                                                    options=[
                                                                        {'label': '90%', 'value': 0.90},
                                                                        {'label': '95%', 'value': 0.95},
                                                                        {'label': '99%', 'value': 0.99}
                                                                    ],
                                                                    value=0.95, clearable=False, style={'width': '100%'}
                                                                ),
                                                            ]),
                                    
                                                            # Renko (tipo + valor)
                                                            html.Div([
                                                                html.Div("Renko", style={'color': EDGE_COLOR, 'fontWeight': 700, 'fontSize': '12px', 'marginBottom': '4px'}),
                                                                dcc.Dropdown(
                                                                    id='aa-brick-type',
                                                                    options=[
                                                                        {'label': '% (ex.: 0,5%)', 'value': 'percent'},
                                                                        {'label': 'Absoluto', 'value': 'abs'}
                                                                    ],
                                                                    value='percent', clearable=False, style={'width': '100%', 'marginBottom': '8px'}
                                                                ),
                                                                dcc.Input(
                                                                    id='aa-brick-value', type='number', step=0.0001, value=0.005,
                                                                    style={
                                                                        'width': '94%', 'backgroundColor': PANEL_2, 'color': TEXT,
                                                                        'border': f'1px solid {EDGE_COLOR}', 'borderRadius': '8px',
                                                                        'height': '32px', 'padding': '0 10px', 'marginTop': '4px'
                                                                    }
                                                                ),
                                                            ]),
                                    
                                                            # Alvo (tijolos)
                                                            html.Div([
                                                                html.Div("Alvo (tijolos)", style={'color': EDGE_COLOR, 'fontWeight': 700, 'fontSize': '12px', 'marginBottom': '4px'}),
                                                                dcc.Input(
                                                                    id='aa-target-R', type='number', step=1, value=12,
                                                                    style={
                                                                        'width': '94%', 'backgroundColor': PANEL_2, 'color': TEXT,
                                                                        'border': f'1px solid {EDGE_COLOR}', 'borderRadius': '8px',
                                                                        'height': '32px', 'padding': '0 10px'
                                                                    }
                                                                ),
                                                            ]),
                                    
                                                            # ✅ Botão reaparece (100% largura)
                                                            html.Button(
                                                                "Ajustar", id='aa-run', n_clicks=0, className="btn-tab",
                                                                style={'height': '36px', 'padding': '0 12px', 'width': '100%', 'marginTop': '4px'}
                                                            ),
                                    
                                                            # Saída
                                                            html.Div(
                                                                id='aa-output',
                                                                style={
                                                                    'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px',
                                                                    'alignItems': 'center', 'marginTop': '8px',
                                                                    'background': PANEL_2, 'border': f'1px solid {EDGE_COLOR}',
                                                                    'borderRadius': '8px', 'padding': '8px 10px',
                                                                    'fontFamily': 'Consolas, monospace', 'fontSize': '13px'
                                                                }
                                                            ),
                                                        ]
                                                    ),
                                                ]
                                            )
                                        ]
                                    ),
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )


    # ===========================================================
    # Callback principal: atualiza 3 blocos conforme os seletores
    # ===========================================================
    @app.callback(
        Output('graph-1', 'figure'), Output('info-box-1', 'children'),
        Output('graph-2', 'figure'), Output('info-box-2', 'children'),
        Output('graph-3', 'figure'), Output('info-box-3', 'children'),
        Output('range-active', 'data'), Output('returns-store', 'data'),
        Output('fit-store', 'data'), Output('last-price', 'data'),
        Input('symbol-dropdown', 'value'),
        Input('slot-type-1', 'value'), Input('slot-type-2', 'value'), Input('slot-type-3', 'value'),
        Input('btn-5y', 'n_clicks'), Input('btn-3y', 'n_clicks'), Input('btn-2y', 'n_clicks'),
        Input('btn-1y', 'n_clicks'), Input('btn-6m', 'n_clicks'), Input('btn-3m', 'n_clicks'), Input('btn-1mo', 'n_clicks'),
    )
    def update_blocks(symbol, t1, t2, t3, *_btns):
        ctx = dash.callback_context
        end = dt.date.today()

        # janela ativa
        if not ctx.triggered:
            start = end - dt.timedelta(days=730); active_button = "btn-2y"
        else:
            btn = ctx.triggered[0]['prop_id'].split('.')[0]
            range_map = {
                'btn-5y': (1825, "btn-5y"), 'btn-3y': (1095, "btn-3y"), 'btn-2y': (730, "btn-2y"),
                'btn-1y': (365, "btn-1y"), 'btn-6m': (182, "btn-6m"), 'btn-3m': (90, "btn-3m"), 'btn-1mo': (30, "btn-1mo")
            }
            delta, active_button = range_map.get(btn, (730, "btn-2y"))
            start = end - dt.timedelta(days=delta)

        # dados base (uma vez)
        df = _safe_download(symbol, start, end)
        s = _ensure_series_close(df)
        r = np.log(s / s.shift(1)).dropna()
        last_price_val = float(s.iloc[-1]) if len(s) else np.nan

        # conteúdos comuns p/ caixas
        # resumo
        desc = {"Mínimo": s.min(), "Máximo": s.max(), "Média": s.mean(), "Mediana": s.median(), "Desvio Padrão": s.std()}
        box_resumo = dcc.Markdown("**Resumo Estatístico**\n\n" + "\n".join([f"**{k}:** {v:,.4f}" for k, v in desc.items()]))

        # ks
        best_name, best_stat, best_p, params = best_distribution_ks(r)
        def fmt_params(name, params):
            try:
                if name == "Normal":   mu, sigma = params[0], params[1]; return f"μ = {mu:.4f}, σ = {sigma:.4f}"
                if name == "Lognormal": shape, loc, scale = params;      return f"shape = {shape:.4f}, loc = {loc:.4f}, scale = {scale:.4f}"
                if name == "Exponencial": loc, scale = params;           return f"loc = {loc:.4f}, scale = {scale:.4f}"
                if name == "Laplace": loc, scale = params;               return f"loc = {loc:.4f}, scale = {scale:.4f}"
                if name == "T-Student": dfv, loc, scale = params;        return f"df = {dfv:.2f}, loc = {loc:.4f}, scale = {scale:.4f}"
                return ", ".join([f"{p:.4f}" for p in params])
            except Exception:
                return "—"
        verdict = "Boa aderência" if best_p >= 0.05 else "Aderência fraca"
        box_ks = dcc.Markdown(
            f"**Teste KS (Kolmogorov–Smirnov)**  \n**Melhor ajuste:** {best_name or '—'}  \n"
            f"**Parâmetros:** {fmt_params(best_name, params) if params else '—'}  \n"
            f"**KS:** {best_stat:.4f} | **p:** {best_p:.4f}  \n**Decisão:** {verdict}"
        )
        # comentário boxplot
        comentario_box = gerar_comentario_boxplot(s)
        box_boxplot = html.Div([html.P(comentario_box, style={"margin": 0})])

        # helpers de figura/caixa por tipo
        def fig_and_info(tipo):
            if tipo == 'combo':     return build_combo(df), box_resumo
            if tipo == 'hist':      return build_histogram(s), box_ks
            if tipo == 'box':       return build_boxplot(s), box_boxplot
            if tipo == 'decomp':    return build_decomp(s), dcc.Markdown("— Decomposição: avaliação qualitativa —")
            if tipo == 'acf_pacf':
                fig = build_acf_pacf(s)
                # diagnóstico rápido
                acf_vals = acf(s.dropna(), nlags=40); pacf_vals = pacf(s.dropna(), nlags=40)
                n = len(s.dropna()); conf = 1.96/np.sqrt(n)
                signif_acf = int(np.sum(np.abs(acf_vals[1:])  > conf))
                signif_pacf = int(np.sum(np.abs(pacf_vals[1:]) > conf))
                acf_text  = "ACF com decaimento lento → diferenciar" if signif_acf > 10 else ("Poucos lags na ACF → possivelmente estacionária" if signif_acf <= 3 else "ACF sugere sazonalidade")
                pacf_text = "PACF ~ AR(1)" if signif_pacf == 1 else (f"PACF ~ AR({signif_pacf})" if 2 <= signif_pacf <= 3 else "PACF sem picos → ruído branco")
                return fig, dcc.Markdown(f"**ACF:** {acf_text}   **PACF:** {pacf_text}")
            if tipo == 'diff':
                fig = build_diff(s)
                try: adf_stat, adf_p, *_ = adfuller(s.diff().dropna(), autolag="AIC")
                except Exception: adf_stat, adf_p = np.nan, np.nan
                try: kpss_stat, kpss_p, *_ = kpss(s.diff().dropna(), regression="c", nlags="auto")
                except Exception: kpss_stat, kpss_p = np.nan, np.nan
                decision = "✅ Estacionária" if (adf_p < 0.05 and kpss_p > 0.05) else "⚠️ Não estacionária"
                return fig, dcc.Markdown(f"**ADF:** {adf_stat:.3f}, p={adf_p:.3f}   **KPSS:** {kpss_stat:.3f}, p={kpss_p:.3f}   → {decision} (Δ¹)")
            if tipo == 'diff2':
                fig = build_diff2(s)
                try: adf_stat, adf_p, *_ = adfuller(s.diff().diff().dropna(), autolag="AIC")
                except Exception: adf_stat, adf_p = np.nan, np.nan
                try: kpss_stat, kpss_p, *_ = kpss(s.diff().diff().dropna(), regression="c", nlags="auto")
                except Exception: kpss_stat, kpss_p = np.nan, np.nan
                decision = "✅ Estacionária" if (adf_p < 0.05 and kpss_p > 0.05) else "⚠️ Não estacionária"
                return fig, dcc.Markdown(f"**ADF:** {adf_stat:.3f}, p={adf_p:.3f}   **KPSS:** {kpss_stat:.3f}, p={kpss_p:.3f}   → {decision} (Δ²)")
            return go.Figure(), "—"

        f1, i1 = fig_and_info(t1)
        f2, i2 = fig_and_info(t2)
        f3, i3 = fig_and_info(t3)

        returns_store = {"alpha_default": 0.95, "values": r.astype(float).tolist()}
        fit_store = {"name": best_name, "params": [float(p) for p in (params or [])],
                     "ks": float(best_stat) if np.isfinite(best_stat) else None,
                     "p": float(best_p) if np.isfinite(best_p) else None}

        return f1, i1, f2, i2, f3, i3, active_button, returns_store, fit_store, last_price_val


    # ===================== Ajuste Automático (resultado vertical + Profit) =====================
    @app.callback(
        Output('aa-output', 'children'),
        Input('aa-run', 'n_clicks'),
        State('aa-alpha', 'value'), State('aa-brick-type', 'value'),
        State('aa-brick-value', 'value'), State('aa-target-R', 'value'),
        State('fit-store', 'data'), State('last-price', 'data'),
        prevent_initial_call=True
    )
    def run_auto_adjust(n_clicks, alpha, brick_type, brick_value, target_R, fit_data, last_price):
        if not fit_data or not fit_data.get('name') or not fit_data.get('params'):
            return "— Ainda não há ajuste de distribuição disponível para este período."
    
        alpha  = float(alpha or 0.95)
        name   = fit_data['name']
        params = tuple(fit_data['params'] or ())
    
        # k, q½ e sigma (no domínio do log-retorno)
        k, q, sigma = _k_from(alpha, name, params)
    
        # ---- Conversões p/ escala Profit (preço) ----
        # σ_preço = exp(σ_log) - 1
        sigma_price   = (np.exp(sigma) - 1) if np.isfinite(sigma) else np.nan
        # Desvio (Profit) = k * σ_preço
        desvio_profit = (k * sigma_price) if (np.isfinite(k) and np.isfinite(sigma_price)) else np.nan
    
        # ---- Métricas relacionadas ao Renko ----
        r_brick = _renko_brick_to_log(brick_type, brick_value, last_price)
        N95     = (q / r_brick) if (np.isfinite(r_brick) and r_brick > 0) else np.nan
        k_target = ((target_R * r_brick) / sigma) if (
            np.isfinite(r_brick) and r_brick > 0 and np.isfinite(sigma) and sigma > 0
        ) else np.nan
    
        # item vertical (linha inteira)
        def item(lbl, val):
            return html.Div(
                [html.Span(lbl, style={'fontWeight': 700, 'color': EDGE_COLOR}),
                 html.Span(val)],
                style={
                    'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
                    'width': '100%', 'padding': '6px 10px',
                    'border': f'1px solid {EDGE_COLOR}', 'borderRadius': '8px',
                    'background': 'rgba(255,255,255,0.03)', 'marginBottom': '8px'
                }
            )
    
        return [
            item("Dist", name or "—"),
            item("α", f"{int(alpha*100)}%"),
            item("k (teórico)", f"{k:,.4f}" if np.isfinite(k) else "—"),
            item("σ (log)", f"{sigma:,.6f}" if np.isfinite(sigma) else "—"),
            item("σ (preço)", f"{sigma_price:,.6f}" if np.isfinite(sigma_price) else "—"),
            item("Desvio (Profit)", f"{desvio_profit:,.4f}" if np.isfinite(desvio_profit) else "—"),
            item("q½", f"{q:,.6f}" if np.isfinite(q) else "—"),
            item("Tijolo", f"{brick_value:.4f}{' %' if brick_type=='percent' else ''}"),
            item("N95", f"{N95:,.2f}" if np.isfinite(N95) else "—"),
            item(f"Alvo {int(target_R)}R → k", f"{k_target:,.4f}" if np.isfinite(k_target) else "—"),
        ]
