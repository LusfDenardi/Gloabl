# ===================== Macro_app.py (Carbon Pro — pronto p/ EMBED e Standalone) =====================
import os
os.environ["DASH_IGNORE_JUPYTER"] = "1"

import dash
from dash import dcc, html, State
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

import yfinance as yf
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime
import logging

from pathlib import Path
from flask import Flask, request, redirect
from werkzeug.middleware.proxy_fix import ProxyFix
from dash.exceptions import PreventUpdate
import requests
import datetime as dt

# ---- MetaTrader5 (não usado)
try:
    import MetaTrader5 as mt5
    MT5_OK = True
except Exception:
    MT5_OK = False
USE_MT5 = False

# ===================== TEMA CARBON PRO =====================
CARBON = {
    "bg": "#0A0B0D",
    "panel": "#101217",
    "panel_alt": "#0C0F14",
    "border": "#1E2633",
    "grid": "#1B2330",
    "hi": "#E8EDF2",
    "soft": "#B8C2CC",
    "muted": "#9AA7B5",
    "accent": "#59F0C8",
    "warn": "#FFD84D",
    "danger": "#EF4444",
    "good": "#10B981",
    "shadow": "0 10px 30px rgba(0,0,0,.35)"
}

def cp_card(style=None):
    base = {
        "background": f"linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.00)), {CARBON['panel']}",
        "border": f"1px solid {CARBON['border']}",
        "borderRadius": "14px",
        "boxShadow": CARBON["shadow"],
        "padding": "10px"
    }
    if style:
        base.update(style)
    return base

YELLOW_LABEL = {
    'color': CARBON['warn'], 'fontSize': '13px',
    'fontWeight': '700', 'marginBottom': '6px'
}

pio.templates["carbonpro"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=CARBON["bg"],
        plot_bgcolor=CARBON["panel"],
        font=dict(color=CARBON["hi"]),
        margin=dict(l=30, r=20, t=40, b=30),
        xaxis=dict(showgrid=True, gridcolor=CARBON["grid"], zeroline=False, tickcolor=CARBON["hi"]),
        yaxis=dict(showgrid=True, gridcolor=CARBON["grid"], zeroline=False, tickcolor=CARBON["hi"]),
        legend=dict(bordercolor=CARBON["border"], borderwidth=1, bgcolor="rgba(0,0,0,0.0)"),
        hoverlabel=dict(bgcolor=CARBON["panel_alt"], bordercolor=CARBON["border"], font_color=CARBON["hi"])
    )
)
pio.templates.default = "carbonpro"

TOGGLE_BTN_STYLE = {
    'width':'42px','height':'42px','borderRadius':'9999px',
    'display':'inline-flex','alignItems':'center','justifyContent':'center',
    'fontSize':'18px','lineHeight':'1',
    'background': CARBON['panel_alt'],'color': CARBON['hi'],
    'border': f"1px solid {CARBON['border']}", 'boxShadow': CARBON['shadow'],
    'cursor':'pointer','transition':'transform .15s ease, box-shadow .2s ease, background .2s ease',
    'marginRight':'12px'
}
GHOST_BTN = {
    'padding':'10px 14px','cursor':'pointer','marginRight':'10px',
    'background': 'transparent','color': CARBON['hi'],
    'border': f"1px solid {CARBON['border']}",'borderRadius':'10px'
}

# ===================== LOGIN (bypass por env) =====================
def load_basic_auth_pairs() -> dict:
    env_val = os.environ.get("BASIC_AUTH_PAIRS", "").strip()
    pairs = {}
    if env_val:
        for item in env_val.split(","):
            if ":" in item:
                u, p = item.split(":", 1)
                pairs[u.strip()] = p.strip()
    return pairs

VALID_USERNAME_PASSWORD_PAIRS = load_basic_auth_pairs()

# ===================== CONFIGS =====================
DEBUG_OPTIONS = False
_last_good_chain = {"SPY": None, "QQQ": None, "IWM": None}
_last_good_spot  = {"SPY": np.nan, "QQQ": np.nan, "IWM": np.nan}

cryptos = [
    "GC=F","BTC-USD","ETH-USD","BNB-USD","SOL-USD","ADA-USD",
    "XRP-USD","DOGE-USD","AVAX-USD","DOT-USD","LTC-USD","LINK-USD","ATOM-USD"
]
SPAN_VOL = 10
interval_ms = 300000
interval_sec = interval_ms // 1000
N_BARRAS = 365

if MT5_OK and USE_MT5:
    timeframes_dict = {'D1': mt5.TIMEFRAME_D1, 'H4': mt5.TIMEFRAME_H4}
else:
    timeframes_dict = {'D1': 'D1', 'H4': 'H4'}

ativos_comp = {
    'EURUSD.r': 1.0000, 'GBPUSD.r': 1.2000, 'USDJPY.r': 140.00,
    'USDCAD.r': 1.3500, 'USDCHF.r': 0.9000, 'USDSEK.r': 10.5000
}
pares_fx = ['EURUSD.r','GBPUSD.r','USDJPY.r','USDCAD.r','USDCHF.r','USDSEK.r']

cores_ativos = {
    'EURUSD.r': '#FF6B6B', 'GBPUSD.r': '#C084FC', 'USDJPY.r': '#22C55E',
    'USDCAD.r': '#F8FA8C', 'USDCHF.r': '#6EE7FF', 'USDSEK.r': '#A78BFA',
    'USDX': '#E8EDF2','EURX':'#FF6B6B','GBPX':'#C084FC','JPYX':'#22C55E',
    'CADX':'#F8FA8C','SEKX':'#6EE7FF','CHFX':'#A78BFA'
}
ativos = ['SPY','QQQ','IWM']

# === Yahoo Chart API (resiliente com cache curto) ===
YCHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YCHART_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}
YCACHE_TTL = 300  # 5 minutos

_ycache = {}  # cache em memória: {key: (timestamp, DataFrame)}

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
    """Puxa OHLC via Yahoo Chart API por data, com backoff e cache curto (TTL=5min)."""
    key = _cache_key(symbol, start, end, interval)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        p1 = int(time.mktime(dt.datetime(start.year, start.month, start.day, 0, 0).timetuple()))
        p2 = int(time.mktime(dt.datetime(end.year, end.month, end.day, 23, 59).timetuple()))
    except Exception:
        # fallback simples (últimos 2 anos)
        p1 = int(time.time()) - 86400 * 365 * 2
        p2 = int(time.time())

    params = {
        "period1": p1,
        "period2": p2,
        "interval": interval,
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
            if r.status_code in (429, 502, 503, 504):
                raise RuntimeError(f"yahoo_busy:{r.status_code}")
            if not r.text.strip():
                raise ValueError("empty_body")
            
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


# === Fallbacks com yfinance ===
def _safe_download(symbol, start, end, interval="1d"):
    """
    Estratégia de download:
      1) Yahoo Chart API (resiliente, com cache curto);
      2) Fallback: yfinance com janela start/end;
      3) Fallback final: yfinance por 'period' heurístico.
    """
    # 1) Tentativa com API Chart
    df = _yahoo_chart_range(symbol, start, end, interval=interval)
    if df is not None and not df.empty:
        return df

    # 2) Fallback yfinance com datas explícitas
    try:
        df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False, threads=False)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    # 3) Fallback heurístico de período
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


# === Utilitário: garantir Série Close ===
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


# === Interface principal compatível com coletar() ===
def _download_close_series(symbol: str, period: str, interval: str) -> pd.Series:
    """
    Substitui completamente o antigo método baseado em yf_download_retry.
    Usa _safe_download (com cache, backoff e fallback) mantendo compatibilidade de saída.
    """
    # Converter período textual ('1y','6mo','3mo','1mo',...) em datas
    end = dt.date.today()
    days_map = {"5y":1825,"3y":1095,"2y":730,"1y":365,"6mo":182,"3mo":90,"1mo":30,"1wk":7}
    delta_days = days_map.get(period, 365)
    start = end - dt.timedelta(days=delta_days)

    df = _safe_download(symbol, start, end, interval=interval)
    if df is None or df.empty:
        return pd.Series(dtype=float)

    s = _ensure_series_close(df)
    s = s[~s.index.duplicated(keep='last')]
    return s


# ========================= Auxiliares =====================
def calcular_b2(df, periodo=20):
    df = df.copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['MA'] = df['Close'].rolling(window=periodo).mean()
    df['STD'] = df['Close'].rolling(window=periodo).std()
    df['Upper'] = df['MA'] + 2 * df['STD']
    df['Lower'] = df['MA'] - 2 * df['STD']
    delta = (df['Upper'] - df['Lower']).replace(0, np.nan)
    df['B2'] = (df['Close'] - df['Lower']) / delta
    df['Date'] = df.index
    return df[['Close','B2','Date']].dropna()

def obter_dados_cripto(ticker):
    try:
        # converte YF → forma usada no macro
        yf_symbol = ticker
        if yf_symbol.endswith('.r'):
            yf_symbol = yf_symbol.replace('.r','=X')

        # usa o motor robusto do macro (Chart API + fallback)
        s = _download_close_series(yf_symbol, period="90d", interval="1d")

        if s is None or s.empty:
            return pd.DataFrame()

        df = pd.DataFrame({"Close": s})
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna()
        return calcular_b2(df)

    except Exception:
        return pd.DataFrame()

def coletar(ticker, timeframe):
    if not (MT5_OK and USE_MT5):
        periodo_map = {'D1': '1y', 'H4': '6mo'}
        intervalo_map = {'D1': '1d', 'H4': '1h'}
        tf_label = timeframe if timeframe in ('D1','H4') else 'D1'
        periodo = periodo_map.get(tf_label,'1y')
        intervalo = intervalo_map.get(tf_label,'1d')
        alias = {"VIX":"^VIX","US500":"^GSPC","USDX":"DX-Y.NYB","TLT":"TLT"}
        yf_symbol = alias.get(ticker, ticker)
        if yf_symbol.endswith('.r'):
            yf_symbol = yf_symbol.replace('.r','=X')
        s = _download_close_series(yf_symbol, periodo, intervalo)
        if s.empty:
            return pd.DataFrame()
        return pd.DataFrame({'time': s.index, 'close': s.values})
    try:
        if not mt5.initialize():
            return pd.DataFrame()
        hoje = datetime.now()
        df = mt5.copy_rates_from(ticker, timeframe, hoje, N_BARRAS)
        if df is None or len(df) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(df)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df.rename(columns={'close': 'close'})[['time','close']]
    except Exception:
        return pd.DataFrame()

# ========================= Opções / Gamma =========================
def get_options_data(ticker, tentativas=3, espera=1.2):
    try:
        stock = yf.Ticker(ticker)
        expirations = None
        for i in range(tentativas):
            try:
                expirations = stock.options
                if expirations:
                    break
            except Exception:
                pass
            time.sleep(espera * (1.5 ** i))
        if not expirations:
            spot = np.nan
            try:
                h = stock.history(period='5d', interval='1d')
                if not h.empty:
                    spot = float(h['Close'].dropna().iloc[-1])
                elif hasattr(stock,"fast_info") and getattr(stock,"fast_info",{}).get('last_price'):
                    spot = float(stock.fast_info['last_price'])
            except Exception:
                pass
            if ticker in _last_good_chain and _last_good_chain[ticker] is not None:
                calls, puts, exp_cached = _last_good_chain[ticker]
                spot_cached = _last_good_spot[ticker]
                return calls.copy(), puts.copy(), float(spot_cached), exp_cached
            return pd.DataFrame(), pd.DataFrame(), spot, None

        calls = puts = None
        exp_date = expirations[0]
        max_to_try = min(6, len(expirations))
        for idx_try in range(max_to_try):
            try:
                exp_date = expirations[idx_try]
                opt_chain = stock.option_chain(exp_date)
                calls, puts = opt_chain.calls, opt_chain.puts
                if calls is not None and not calls.empty and puts is not None and not puts.empty:
                    break
            except Exception:
                time.sleep(1.0 + 0.8 * idx_try)
                continue

        if calls is None or puts is None or calls.empty or puts.empty:
            spot = np.nan
            try:
                h = stock.history(period='5d', interval='1d')
                if not h.empty:
                    spot = float(h['Close'].dropna().iloc[-1])
            except Exception:
                pass
            if ticker in _last_good_chain and _last_good_chain[ticker] is not None:
                calls, puts, exp_cached = _last_good_chain[ticker]
                spot_cached = _last_good_spot[ticker]
                return calls.copy(), puts.copy(), float(spot_cached), exp_cached
            return pd.DataFrame(), pd.DataFrame(), spot, None

        calls = calls.copy(); puts = puts.copy()
        calls['expiration'] = pd.to_datetime(exp_date)
        puts['expiration'] = pd.to_datetime(exp_date)

        spot = np.nan
        try:
            h = stock.history(period='5d', interval='1d')
            if not h.empty:
                spot = float(h['Close'].dropna().iloc[-1])
        except Exception:
            pass
        if np.isnan(spot):
            try:
                h = stock.history(period='1d')
                if not h.empty:
                    spot = float(h['Close'].dropna().iloc[-1])
            except Exception:
                pass
        if np.isnan(spot):
            try:
                if hasattr(stock,"fast_info") and getattr(stock,"fast_info",{}).get('last_price'):
                    spot = float(stock.fast_info['last_price'])
            except Exception:
                pass

        if ticker in _last_good_chain:
            _last_good_chain[ticker] = (calls.copy(), puts.copy(), exp_date)
            _last_good_spot[ticker] = spot
        return calls, puts, spot, exp_date

    except Exception:
        if ticker in _last_good_chain and _last_good_chain[ticker] is not None:
            calls, puts, exp_cached = _last_good_chain[ticker]
            spot_cached = _last_good_spot[ticker]
            return calls.copy(), puts.copy(), float(spot_cached), exp_cached
        return pd.DataFrame(), pd.DataFrame(), np.nan, None

def gamma_exposure(options, spot, rate=0.05):
    try:
        S = float(spot)
        K = pd.to_numeric(options['strike'], errors='coerce')
        sigma = pd.to_numeric(options['impliedVolatility'], errors='coerce').replace(0,np.nan)
        T = options['expiration'].apply(lambda x: max((x - pd.Timestamp.today()).days/365, 1/365))
        d1 = (np.log(S / K) + (rate + (sigma**2)/2) * T) / (sigma * np.sqrt(T))
        gamma = np.exp(-d1**2/2) / (S * sigma * np.sqrt(2*np.pi*T))
        oi = pd.to_numeric(options['openInterest'], errors='coerce').fillna(0.0)
        gamma_exposure = gamma * oi * 100 * S * 0.01
        return gamma_exposure.fillna(0)
    except Exception:
        return pd.Series([0] * len(options))

# ========================= Gráficos macro =====================
def gerar_range_index_plotly(timeframe_label, ativos_visiveis=None):
    tf = timeframe_label
    df_final = pd.DataFrame()

    df_usdx = coletar('USDX', tf)
    if not df_usdx.empty:
        df_usdx = df_usdx.rename(columns={'close': 'USDX'})
        df_final = df_usdx.copy()

    indices_sinteticos = {
        'EURX': (['EURUSD.r','EURGBP.r','EURJPY.r','EURCHF.r','EURAUD.r'], [0.315,0.305,0.189,0.111,0.080]),
        'GBPX': (['GBPUSD.r','GBPJPY.r','GBPEUR.r','GBPCHF.r'], [0.347,0.207,0.339,0.107]),
        'JPYX': (['USDJPY.r','EURJPY.r','GBPJPY.r'], [0.5,0.3,0.2]),
        'CADX': (['USDCAD.r','EURCAD.r','GBPCAD.r'], [0.5,0.3,0.2]),
        'SEKX': (['USDSEK.r','EURSEK.r'], [0.6,0.4]),
        'CHFX': (['USDCHF.r','EURCHF.r','GBPCHF.r'], [0.5,0.3,0.2]),
    }

    def calcular_index_sintetico(pares, pesos):
        df_idx = pd.DataFrame()
        for par in pares:
            d = coletar(par, tf)
            if not d.empty:
                d = d.rename(columns={'close': par})
                df_idx = d if df_idx.empty else pd.merge(df_idx, d, on='time', how='outer')
        if df_idx.empty:
            return None
        df_idx = df_idx.sort_values('time').ffill().bfill()
        cols = [c for c in df_idx.columns if c != 'time']
        pow_list = []
        for par, peso in zip(pares, pesos):
            if par in cols:
                pow_list.append(np.power(df_idx[par].values, peso))
        if not pow_list:
            return None
        index_val = np.prod(pow_list, axis=0)
        out = pd.DataFrame({'time': df_idx['time'].values, 'Index': index_val})
        return out

    for nome, (pares, pesos) in indices_sinteticos.items():
        df_idx = calcular_index_sintetico(pares, pesos)
        if df_idx is not None:
            df_idx = df_idx.rename(columns={'Index': nome})
            df_final = df_idx if df_final.empty else pd.merge(df_final, df_idx, on='time', how='outer')

    if df_final.empty:
        return go.Figure()

    df_final = df_final.sort_values('time').ffill().bfill()
    ativos_cols = [c for c in df_final.columns if c != 'time']
    for ativo in ativos_cols:
        df_final[ativo] = 100 + 100*np.log(df_final[ativo] / df_final[ativo].iloc[0])

    df_final['MediaGrupo'] = df_final[ativos_cols].mean(axis=1)
    df_final['FaixaAltaSuave'] = df_final['MediaGrupo'] + 10
    df_final['FaixaBaixaSuave'] = df_final['MediaGrupo'] - 10
    df_final['FaixaAltaInternaSuave2'] = df_final['MediaGrupo'] + 9
    df_final['FaixaBaixaInternaSuave2'] = df_final['MediaGrupo'] - 9

    span = 100
    for col in ['MediaGrupo','FaixaAltaSuave','FaixaBaixaSuave','FaixaAltaInternaSuave2','FaixaBaixaInternaSuave2']:
        df_final[f'{col}_smooth'] = df_final[col].ewm(span=span).mean().ewm(span=span).mean()

    FUT_DAYS   = 40
    LABEL_DAYS = 5
    PROJ_DAYS  = 10
    DRAW_STUB  = True

    t_hist = df_final['time']
    last_t = pd.to_datetime(t_hist.iloc[-1])
    t_future = pd.bdate_range(start=last_t + pd.Timedelta(days=1), periods=FUT_DAYS, freq='B')

    def extend_flat(series, last_val):
        if len(t_future) == 0:
            return pd.Series(series.values, index=t_hist), t_hist
        s_ext = pd.Series([last_val]*len(t_future), index=t_future)
        s_full = pd.concat([pd.Series(series.values, index=t_hist), s_ext])
        return s_full, s_full.index

    out_sup_full, t_full = extend_flat(df_final['FaixaAltaSuave_smooth'],       float(df_final['FaixaAltaSuave_smooth'].iloc[-1]))
    out_inf_full, _      = extend_flat(df_final['FaixaBaixaSuave_smooth'],      float(df_final['FaixaBaixaSuave_smooth'].iloc[-1]))
    in_sup_full, _       = extend_flat(df_final['FaixaAltaInternaSuave2_smooth'], float(df_final['FaixaAltaInternaSuave2_smooth'].iloc[-1]))
    in_inf_full, _       = extend_flat(df_final['FaixaBaixaInternaSuave2_smooth'], float(df_final['FaixaBaixaInternaSuave2_smooth'].iloc[-1]))
    media_s_full, _      = extend_flat(df_final['MediaGrupo_smooth'],           float(df_final['MediaGrupo_smooth'].iloc[-1]))

    fig = go.Figure()
    visiveis = ativos_visiveis or [c for c in df_final.columns if c != 'time']
    x_max = t_full[-1] if len(t_full) else last_t

    for ativo in visiveis:
        if ativo not in df_final.columns:
            continue
        y = df_final[ativo]
        fig.add_trace(go.Scatter(
            x=t_hist, y=y, mode='lines',
            name=ativo, line=dict(color=cores_ativos.get(ativo,'gray')),
            hovertemplate=f"<b>{ativo}</b><br>Data: %{{x|%d/%m/%Y}}<br>Valor: %{{y:.2f}}",
            showlegend=False
        ))
        if DRAW_STUB and len(t_future):
            x_stub = [last_t, min(last_t + pd.Timedelta(days=PROJ_DAYS), x_max)]
            fig.add_trace(go.Scatter(
                x=x_stub, y=[y.iloc[-1], y.iloc[-1]],
                mode='lines', line=dict(color=cores_ativos.get(ativo,'gray'), dash='dot', width=1),
                showlegend=False, hoverinfo='skip'
            ))
        label_x = min(last_t + pd.Timedelta(days=LABEL_DAYS), x_max)
        fig.add_trace(go.Scatter(
            x=[label_x], y=[y.iloc[-1]],
            mode='text', text=[ativo], textposition='middle right',
            showlegend=False, textfont=dict(size=12, color=cores_ativos.get(ativo,'#E8EDF2')),
            hoverinfo='skip'
        ))

    fig.add_trace(go.Scatter(x=t_full, y=out_sup_full, mode='lines',
                             line=dict(color=CARBON['soft'], width=2), showlegend=False))
    fig.add_trace(go.Scatter(x=t_full, y=out_inf_full, mode='lines',
                             line=dict(color=CARBON['soft'], width=2), showlegend=False))
    fig.add_trace(go.Scatter(x=t_full, y=in_sup_full,  mode='lines',
                             line=dict(color=CARBON['muted'], dash='dash'), showlegend=False))
    fig.add_trace(go.Scatter(x=t_full, y=in_inf_full,  mode='lines',
                             line=dict(color=CARBON['muted'], dash='dash'), showlegend=False))
    fig.add_trace(go.Scatter(x=t_full, y=media_s_full, mode='lines',
                             line=dict(color='#BFBFBF', width=2, dash='dash'), showlegend=False))

    today_str = datetime.now().strftime("%Y-%m-%d")
    fig.add_shape(type="line", x0=today_str, x1=today_str, y0=0, y1=1,
                  xref='x', yref='paper', line=dict(color=CARBON["warn"], width=0.5, dash="dash"))
    fig.add_annotation(x=today_str, y=1.1, xref="x", yref="paper",
                       text=datetime.now().strftime("%d/%m/%Y"),
                       showarrow=False, font=dict(color=CARBON["warn"], size=8))

    fig.update_layout(title=dict(text=f"Index ({timeframe_label})", x=0.5, xanchor='center'))
    return fig

def gerar_range_comparativo_plotly(timeframe_label, ativos_visiveis=None):
    Q_CROSS     = 0.95
    SAFETY      = 2.00
    K_OUTER     = 1.00
    K_INNER     = 0.85
    SMOOTH_SPAN = 180
    MIN_HIST    = 60
    ROLL_WINDOW = 252

    FUT_DAYS      = 40
    LABEL_DAYS    = 5
    PROJ_DAYS     = 10
    DRAW_STUB     = True

    tf = timeframe_label
    df_final = pd.DataFrame()

    for ativo, base in ativos_comp.items():
        d = coletar(ativo, tf)
        if not d.empty:
            d = d.sort_values('time')
            d['norm'] = 100 + 100*np.log(d['close'] / float(base))
            d = d[['time','norm']].rename(columns={'norm': ativo})
            df_final = d if df_final.empty else pd.merge(df_final, d, on='time', how='outer')

    if df_final.empty:
        return go.Figure()

    df_final = df_final.sort_values('time').ffill().bfill()
    ativos_cols = list(ativos_comp.keys())

    media = df_final[ativos_cols].mean(axis=1)
    dev_abs = df_final[ativos_cols].sub(media, axis=0).abs()
    q_cs = dev_abs.quantile(Q_CROSS, axis=1)
    env_long = q_cs.rolling(ROLL_WINDOW, min_periods=MIN_HIST).mean()
    env_long = env_long.fillna(method='bfill') * SAFETY

    out_sup = (media + K_OUTER * env_long).ewm(span=SMOOTH_SPAN).mean()
    out_inf = (media - K_OUTER * env_long).ewm(span=SMOOTH_SPAN).mean()
    in_sup  = (media + K_INNER * env_long).ewm(span=SMOOTH_SPAN).mean()
    in_inf  = (media - K_INNER * env_long).ewm(span=SMOOTH_SPAN).mean()
    media_s = media.ewm(span=SMOOTH_SPAN).mean()

    t_hist = df_final['time']
    last_t = pd.to_datetime(t_hist.iloc[-1])
    t_future = pd.bdate_range(start=last_t + pd.Timedelta(days=1), periods=FUT_DAYS, freq='B')

    def extend_flat(series, last_val):
        if len(t_future) == 0:
            return series, t_hist
        s_ext = pd.Series([last_val]*len(t_future), index=t_future)
        s_full = pd.concat([pd.Series(series.values, index=t_hist), s_ext])
        return s_full, s_full.index

    out_sup_full, t_full = extend_flat(out_sup, float(out_sup.iloc[-1]))
    out_inf_full, _      = extend_flat(out_inf, float(out_inf.iloc[-1]))
    in_sup_full, _       = extend_flat(in_sup,  float(in_sup.iloc[-1]))
    in_inf_full, _       = extend_flat(in_inf,  float(in_inf.iloc[-1]))
    media_s_full, _      = extend_flat(media_s, float(media_s.iloc[-1]))

    fig = go.Figure()
    visiveis = ativos_visiveis or ativos_cols
    x_max = t_full[-1] if len(t_full) else last_t

    for ativo in visiveis:
        if ativo not in df_final.columns:
            continue
        y = df_final[ativo]
        fig.add_trace(go.Scatter(
            x=t_hist, y=y, mode='lines',
            name=ativo, line=dict(color=cores_ativos.get(ativo,'#E8EDF2')),
            hovertemplate=f"<b>{ativo}</b><br>Data: %{{x|%d/%m/%Y}}<br>Valor: %{{y:.2f}}",
            showlegend=False
        ))
        if DRAW_STUB and len(t_future):
            x_stub = [last_t, min(last_t + pd.Timedelta(days=PROJ_DAYS), x_max)]
            fig.add_trace(go.Scatter(
                x=x_stub, y=[y.iloc[-1], y.iloc[-1]],
                mode='lines', line=dict(color=cores_ativos.get(ativo,'#E8EDF2'), dash='dot', width=1),
                showlegend=False, hoverinfo='skip'
            ))
        label_x = min(last_t + pd.Timedelta(days=LABEL_DAYS), x_max)
        fig.add_trace(go.Scatter(
            x=[label_x], y=[y.iloc[-1]],
            mode='text', text=[ativo], textposition='middle right',
            showlegend=False, textfont=dict(color=cores_ativos.get(ativo,'#E8EDF2'), size=12),
            hoverinfo='skip'
        ))

    fig.add_trace(go.Scatter(x=t_full, y=out_sup_full, mode='lines',
                             line=dict(color=CARBON['soft'], width=2), showlegend=False))
    fig.add_trace(go.Scatter(x=t_full, y=out_inf_full, mode='lines',
                             line=dict(color=CARBON['soft'], width=2), showlegend=False))
    fig.add_trace(go.Scatter(x=t_full, y=in_sup_full,  mode='lines',
                             line=dict(color=CARBON['muted'], dash='dash'), showlegend=False))
    fig.add_trace(go.Scatter(x=t_full, y=in_inf_full,  mode='lines',
                             line=dict(color=CARBON['muted'], dash='dash'), showlegend=False))
    fig.add_trace(go.Scatter(x=t_full, y=media_s_full, mode='lines',
                             line=dict(color='#BFBFBF', width=2, dash='dash'), showlegend=False))

    today_str = datetime.now().strftime("%Y-%m-%d")
    fig.add_shape(type="line", x0=today_str, x1=today_str, y0=0, y1=1,
                  xref='x', yref='paper', line=dict(color=CARBON["warn"], width=0.5, dash="dash"))
    fig.add_annotation(x=today_str, y=1.1, xref="x", yref="paper",
                       text=datetime.now().strftime("%d/%m/%Y"),
                       showarrow=False, font=dict(color=CARBON["warn"], size=8))

    fig.update_layout(title=dict(text=f"Pares ({timeframe_label})", x=0.5, xanchor='center'))
    return fig

def calcular_hotelling_t2(_, timeframe):
    tf_label = timeframe if timeframe in ('D1','H4') else 'D1'
    df_final = pd.DataFrame()
    for ativo in pares_fx:
        d = coletar(ativo, tf_label)
        if not d.empty:
            d = d.rename(columns={'close': ativo})
            df_final = d if df_final.empty else pd.merge(df_final, d, on='time', how='outer')
    if df_final.empty:
        return go.Figure()
    df_final = df_final.sort_values('time').ffill().bfill()
    for ativo in pares_fx:
        df_final[f"{ativo}_ret"] = np.log(df_final[ativo] / df_final[ativo].shift(1))
    df_ret = df_final[[f"{ativo}_ret" for ativo in pares_fx]].dropna()
    means = df_ret.mean()
    stds = df_ret.std().replace(0,np.nan)
    zscores = (df_ret - means) / stds
    T2 = (zscores**3).sum(axis=1).replace([np.inf,-np.inf], np.nan).fillna(0)
    df_plot = df_final.loc[df_ret.index].copy()
    df_plot['T2'] = T2
    idx_red = df_plot[df_plot['T2'] > 30].index

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['T2'], mode='lines',
                             line=dict(color='rgb(200,200,200)', width=1),
                             name='Curva T2', showlegend=False))
    fig.add_trace(go.Scatter(x=df_plot.loc[idx_red,'time'], y=df_plot.loc[idx_red,'T2'],
                             mode='markers', marker=dict(color='red', size=3),
                             name='Ponto Crítico', showlegend=False))
    today_str_iso = datetime.now().strftime("%Y-%m-%d")
    fig.add_shape(type="line", x0=today_str_iso, x1=today_str_iso, y0=0, y1=1,
                  xref='x', yref='paper', line=dict(color=CARBON["warn"], width=0.5, dash="dash"))
    fig.add_annotation(x=today_str_iso, y=1.10, xref='x', yref='paper',
                       text=datetime.now().strftime("%d/%m/%Y"),
                       showarrow=False, font=dict(color=CARBON["warn"], size=8))
    fig.add_shape(type="line",
                  x0=df_plot['time'].min(), x1=df_plot['time'].max(), y0=30, y1=30,
                  line=dict(color=CARBON["good"], width=0.7, dash="dashdot"), name="Nível 30")
    fig.update_layout(title=dict(text="Ponto de Interesse", x=0.5, xanchor='center'))
    return fig

def gerar_grafico_vix(timeframe, divisor_macro):
    tf_label = timeframe if timeframe in ('D1','H4') else 'D1'
    df_vix = coletar("VIX", tf_label)
    df_sp = coletar("US500", tf_label)
    df_usdx = coletar("USDX", tf_label)
    df_tlt = coletar("TLT", tf_label)
    if any(df is None or df.empty for df in [df_vix, df_sp, df_usdx, df_tlt]):
        return go.Figure()
    df = df_vix.copy()
    df['media'] = df['close'].ewm(span=742).mean()
    df['z'] = (df['close'] - df['media']) / df['close'].rolling(200).std()

    df_sp = df_sp.copy()
    df_sp['ret'] = np.log(df_sp['close']).diff()
    df_sp['vol'] = df_sp['ret'].rolling(20).std() * np.sqrt(252) * 100
    df['vol_realizada'] = df_sp['vol'] / max(divisor_macro, 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time'], y=df['close'], mode='lines',
                             name='VIX (Preço)', line=dict(color='orange'), showlegend=False))
    fig.add_trace(go.Scatter(x=df['time'], y=df['media'], mode='lines',
                             name='Média Exponencial (742)', line=dict(color=CARBON['hi'], dash='dash'),
                             showlegend=False))
    fig.add_trace(go.Bar(x=df['time'], y=df['vol_realizada'],
                         name='Vol. Realizada (S&P 500)', marker_color='tan', showlegend=False))
    df['marker'] = np.where(df['z'] > 3.5, 'acima_3.5', np.where(df['z'] > 3, 'acima_3', ''))
    df_z3 = df[df['marker']=='acima_3']
    df_z35 = df[df['marker']=='acima_3.5']
    fig.add_trace(go.Scatter(x=df_z3['time'], y=df_z3['close'], mode='markers',
                             name='Z > 3', marker=dict(size=3, color='yellow'), showlegend=False))
    fig.add_trace(go.Scatter(x=df_z35['time'], y=df_z35['close'], mode='markers',
                             name='Z > 3.5', marker=dict(size=3, color='red'), showlegend=False))
    fig.add_trace(go.Scatter(x=df_usdx['time'], y=df_usdx['close']/max(divisor_macro,1),
                             mode='lines', name='DXY', line=dict(color='lime')))
    fig.add_trace(go.Scatter(x=df_tlt['time'], y=df_tlt['close']/max(divisor_macro,1),
                             mode='lines', name='TLT', line=dict(color='magenta')))
    today_str = datetime.now().strftime("%Y-%m-%d")
    fig.add_shape(type="line", x0=today_str, x1=today_str, y0=0, y1=1,
                  xref='x', yref='paper', line=dict(color=CARBON["warn"], width=0.5, dash="dash"))
    fig.add_annotation(x=today_str, y=1.03, xref="x", yref="paper",
                       text=datetime.now().strftime("%d/%m/%Y"),
                       showarrow=False, font=dict(color=CARBON["warn"], size=10))
    fig.update_layout(title=dict(text='VIX - Contexto Macro', x=0.5, xanchor='center'))
    return fig

# ===================== FACTORY / MOUNT =====================
def _normalize_prefix(prefix: str) -> str:
    if not prefix:
        prefix = "/app/macro/"  # padrão para embed no host
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    if not prefix.endswith("/"):
        prefix = prefix + "/"
    return prefix

def _build_app(shared_server: Flask, prefix: str):
    """Cria server+Dash já com prefixo normalizado."""
    server = shared_server or Flask(__name__)
    server.wsgi_app = ProxyFix(server.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    PREFIX = _normalize_prefix(prefix or os.environ.get("APP_PREFIX", "/app/macro/"))

    assets_dir = Path(__file__).parent / "assets"
    assets_dir.mkdir(exist_ok=True)

    app = dash.Dash(
        __name__,
        server=server,
        requests_pathname_prefix=PREFIX,
        routes_pathname_prefix=PREFIX,
        suppress_callback_exceptions=True,
        serve_locally=True,
        assets_url_path=f"{PREFIX}assets",
        assets_folder=str(assets_dir),
        title="Macro Visão • GlobalStat",
    )

    # Fundo do HTML base (evita “tela preta” enquanto carrega)
    try:
        app.index_string = app.index_string.replace(
            "<body>", '<body style="background-color:#0B0C0D!important;color:#E6F1EE;">'
        )
    except Exception:
        pass

    # Rotas utilitárias (redireciona raiz p/ prefixo)
    if __name__ == "__main__":
        @server.route("/")
        def _root_redirect():
            qs = f"?{request.query_string.decode()}" if request.query_string else ""
            return redirect(f"{PREFIX}{qs}")

    @server.route("/healthz")
    def _healthz():
        return "ok", 200

    # ===================== LAYOUT =====================
    SIDEBAR_BASE = {'position':'fixed','top':'0','left':'0','bottom':'0','width':'240px','padding':'14px',
                    'backgroundColor':CARBON['panel_alt'],'borderRight':f"1px solid {CARBON['border']}",
                    'overflowY':'auto','transition':'transform 0.25s ease','zIndex':1000}
    CONTENT_BASE = {'transition':'margin-left 0.25s ease'}

    app.layout = html.Div(style={"backgroundColor":CARBON['bg'],"padding":"10px","color":CARBON['hi']}, children=[
        dcc.Location(id="url", refresh=False),

        html.Div(id='main-wrapper', style={**CONTENT_BASE, 'marginLeft':'0px','backgroundColor':CARBON['bg']}, children=[

            html.Div(id='page-geral', children=[
                html.A(
                    html.Button("← Voltar", className="btn-tab back-btn"),
                    href="/app/",
                    style={"marginBottom": "12px", "display": "inline-block"}
                ),
                html.Div([
                    html.Div([
                        html.P("Buy and Hold", style={"color":CARBON['hi'],"marginBottom":"5px"}),
                        dcc.Interval(id='atualizacao-temporizada', interval=3*60*1000, n_intervals=0),
                        html.Div(id="mini-graficos", style={"whiteSpace":"nowrap","overflowX":"auto"}),
                        html.Div(id="mensagem-erro", style={"color":CARBON['danger'],"marginTop":"10px"})
                    ], style={'display':'inline-block','width':'100%'})
                ], style={'display':'flex','alignItems':'center','marginBottom':'10px'}),

                html.Hr(style={"margin":"20px 0","borderColor":CARBON['border']}),

                html.Div([
                    html.Button("Atualizar", id="refresh-button", style=GHOST_BTN),
                    dcc.Dropdown(
                        id="timeframe-dropdown",
                        options=[{"label":k,"value":k} for k in timeframes_dict],
                        value="D1",
                        clearable=False,
                        className="carbon-dropdown",
                        style={"width":"100px","display":"inline-block"}
                    ),
                    html.Div(id="last-update", style={"color":CARBON['muted'],"marginLeft":"20px","display":"inline-block"})
                ], style={'marginBottom':'8px'}),

                html.Div([
                    html.Div([
                        dcc.Checklist(id="checklist-index",
                            options=[{"label":a,"value":a} for a in ['USDX','EURX','GBPX','JPYX','CADX','SEKX','CHFX']],
                            value=['USDX','EURX','GBPX','JPYX','CADX','SEKX','CHFX'],
                            labelStyle={'display':'inline-block','marginRight':'12px','color':CARBON['hi'],'fontSize':'12px'},
                            style={'marginBottom':'5px','textAlign':'center'}),
                        dcc.Graph(id='graph-index', style={'height':'38vh','width':'100%'})
                    ], style={**cp_card({'width':'25%','padding':'8px','display':'flex','flexDirection':'column','alignItems':'center'})}),

                    html.Div([
                        dcc.Checklist(id='checklist-comparativo',
                            options=[{"label":k,"value":k} for k in ativos_comp.keys()],
                            value=list(ativos_comp.keys()),
                            labelStyle={'display':'inline-block','marginRight':'12px','color':CARBON['hi'],'fontSize':'12px'},
                            style={'marginBottom':'5px','textAlign':'center'}),
                        dcc.Graph(id='graph-comparativo', style={'height':'38vh','width':'100%'})
                    ], style={**cp_card({'width':'25%','padding':'8px','display':'flex','flexDirection':'column','alignItems':'center','marginLeft':'10px'})}),

                    html.Div([
                        html.Div(' ', style={'height':'27px'}),
                        dcc.Graph(id='graph-t2', style={'height':'38vh','width':'100%'})
                    ], style={**cp_card({'width':'25%','padding':'8px','display':'flex','flexDirection':'column','alignItems':'center','marginLeft':'10px'})}),

                    html.Div([
                        html.Div([
                            dcc.Dropdown(
                                id='divisor-dropdown',
                                options=[{'label':'3','value':3},{'label':'4','value':4}],
                                value=3,
                                clearable=False,
                                className="carbon-dropdown",
                                style={'width':'70px','height':'28px','borderRadius':'8px',
                                       'fontSize':'12px','padding':'0px','textAlign':'center',
                                       'marginBottom':'4px','float':'right'}
                            )
                        ], style={'width':'100%','display':'inline-block','textAlign':'right'}),
                        dcc.Graph(id='grafico-vix', style={'height':'38vh','width':'100%'})
                    ], style={**cp_card({'width':'25%','padding':'8px','display':'flex','flexDirection':'column','alignItems':'center','marginLeft':'10px'})}),
                ], style={'display':'flex'}),

                html.Hr(style={"margin":"20px 0","borderColor":CARBON['border']}),

                html.Div(id='gamma-charts-container', style={'display':'flex','flexDirection':'column','alignItems':'center','gap':'20px','padding':'10px 0'}),

                html.Hr(style={"margin":"20px 0","borderColor":CARBON['border']}),

                html.Div(id='crypto-charts-container', style={'display':'flex','flexWrap':'wrap','justifyContent':'center','gap':'10px','padding':'10px 0'}),

                html.Div(id="contador", style={"color":CARBON['muted'],"marginTop":"20px"}),

                dcc.Interval(id="interval-refresh", interval=interval_ms, n_intervals=0),
                dcc.Interval(id="interval-1s", interval=1000, n_intervals=0),
                dcc.Interval(id='interval-component', interval=300*1000, n_intervals=0)
            ]),
        ])
    ])

    # ===================== CALLBACKS =====================
    # Mini-gráficos (Cripto)
    @app.callback(
        [Output("mini-graficos","children"), Output("mensagem-erro","children")],
        Input("atualizacao-temporizada","n_intervals"))
    def atualizar_mini_graficos(n):
        graficos = []; erros = []
        for cripto in cryptos:
            df = obter_dados_cripto(cripto)
            time.sleep(0.4)
            if df.empty:
                erros.append("XAU" if cripto=="GC=F" else cripto.replace("-USD","").replace("=F","").replace("X",""))
                continue
            nome = "XAU" if cripto=="GC=F" else cripto.replace("-USD","").replace("=F","").replace("X","")
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(
                x=df['Date'], y=df['Close'], mode='lines',
                line=dict(color='yellow', width=1),
                name='Preço', showlegend=False
            ), secondary_y=True)
            df_sinal = df[df['B2'] < 0]
            fig.add_trace(go.Scatter(
                x=df_sinal['Date'], y=df_sinal['Close'],
                mode='markers',
                marker=dict(color='lime', size=8, symbol='triangle-up'),
                name="sinal", showlegend=False
            ), secondary_y=True)
            x_line = pd.Timestamp(df['Date'].iloc[-1]).to_pydatetime()
            ymin  = float(df['Close'].min()); ymax  = float(df['Close'].max())
            fig.add_trace(go.Scatter(x=[x_line, x_line], y=[ymin, ymax], mode='lines',
                    line=dict(color='#FFFACD', width=1.4, dash='dash'), hoverinfo='skip', showlegend=False), secondary_y=True)
            fig.add_annotation(x=x_line, y=0, xref="x", yref="paper",
                               text=x_line.strftime("%Y-%m-%d"), showarrow=False,
                               font=dict(color="#FFFACD", size=8), align="center", yanchor="top")
            fig.update_layout(title=nome, title_x=0.5, height=120, width=200,
                              margin=dict(l=5,r=5,t=25,b=20), plot_bgcolor=CARBON['panel'], paper_bgcolor=CARBON['panel'],
                              font=dict(color=CARBON['hi'], size=8),
                              xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                              yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                              yaxis2=dict(showticklabels=False, showgrid=False, zeroline=False))
            graficos.append(html.Div(dcc.Graph(figure=fig),
                            style={"display":"inline-block","marginRight":"10px",
                                   "border":f"1px solid {CARBON['border']}","borderRadius":"10px",
                                   "padding":"5px","backgroundColor":CARBON['panel']}))
        mensagem_erro = "Falha ao carregar: " + ", ".join(erros) if erros else ""
        return graficos, mensagem_erro

    # Principais (macro blocos)
    @app.callback(
        [Output('graph-index','figure'),
         Output('graph-comparativo','figure'),
         Output('graph-t2','figure'),
         Output('grafico-vix','figure'),
         Output('last-update','children')],
        [Input('refresh-button','n_clicks'),
         Input('interval-refresh','n_intervals'),
         Input('checklist-index','value'),
         Input('checklist-comparativo','value')],
        [State('timeframe-dropdown','value'),
         State('divisor-dropdown','value')])
    def atualizar_todos(n_clicks, n_intervals, index_sel, comp_sel, timeframe_label, divisor_macro):
        fig_index = gerar_range_index_plotly(timeframe_label, index_sel)
        fig_comp  = gerar_range_comparativo_plotly(timeframe_label, comp_sel)
        fig_t2    = calcular_hotelling_t2(None, timeframe_label)
        fig_vix   = gerar_grafico_vix(timeframe_label, divisor_macro)
        now = datetime.now().strftime("Última atualização: %d/%m/%Y %H:%M:%S")
        return fig_index, fig_comp, fig_t2, fig_vix, now

    # Contador
    @app.callback(Output('contador','children'), Input('interval-1s','n_intervals'))
    def atualizar_contador(n):
        tempo_restante = interval_sec - (n % interval_sec)
        return f'Atualização automática em: {tempo_restante} segundos'

    # Gamma
    @app.callback(Output('gamma-charts-container','children'), Input('interval-component','n_intervals'))
    def update_dashboard(n):
        columns = []
        for ativo in ativos:
            calls, puts, spot, exp_date = get_options_data(ativo)
            if calls.empty or puts.empty or np.isnan(spot):
                fig_strike = go.Figure(); fig_strike.add_annotation(text="Sem dados de opções no momento",
                                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                fig_strike.update_layout(height=400, title=f'{ativo} Gamma Exposure Strike')
                fig_profile = go.Figure(); fig_profile.add_annotation(text="Sem dados de opções no momento",
                                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                fig_profile.update_layout(height=400, title=f'{ativo} Gamma Exposure Profile')
            else:
                calls_gamma = gamma_exposure(calls, spot)
                puts_gamma  = gamma_exposure(puts,  spot)
                strikes_calls = pd.to_numeric(calls['strike'], errors='coerce')
                strikes_puts  = pd.to_numeric(puts['strike'],  errors='coerce')
                call_wall = float(calls.iloc[np.argmax(calls_gamma)]['strike']) if not calls_gamma.empty else float(spot)
                put_wall  = float(puts.iloc[np.argmax(puts_gamma)]['strike'])   if not puts_gamma.empty else float(spot)

                fig_strike = go.Figure()
                fig_strike.add_trace(go.Bar(x=strikes_calls, y=calls_gamma, name='Call Gamma'))
                fig_strike.add_trace(go.Bar(x=strikes_puts,  y=-puts_gamma, name='Put Gamma'))
                fig_strike.add_vline(x=call_wall, line_color='green', line_dash='dot')
                fig_strike.add_vline(x=put_wall,  line_color='red',   line_dash='dot')
                fig_strike.add_vline(x=spot,      line_color='yellow')
                fig_strike.add_annotation(text=f'Call Wall: {call_wall}', x=call_wall, y=1.15, xref='x', yref='paper', showarrow=False)
                fig_strike.add_annotation(text=f'Put Wall: {put_wall}', x=put_wall, y=1.10, xref='x', yref='paper', showarrow=False)
                fig_strike.add_annotation(text=f'Spot: {spot:.2f}', x=spot, y=1.00, xref='x', yref='paper', showarrow=False)
                fig_strike.update_layout(height=400, title=f'{ativo} Gamma Exposure Strike',
                                         xaxis_title='Strike', yaxis_title='Gamma Exposure')

                price_range = np.linspace(float(spot)*0.90, float(spot)*1.20, 100)
                gamma_profile = [gamma_exposure(calls, p).sum() - gamma_exposure(puts, p).sum() for p in price_range]
                gamma_flip_idx = np.where(np.diff(np.sign(gamma_profile)))[0]
                gamma_flip = price_range[gamma_flip_idx[0]] if len(gamma_flip_idx)>0 else float(spot)

                fig_profile = go.Figure()
                fig_profile.add_trace(go.Scatter(x=price_range, y=gamma_profile, mode='lines', name='Gamma Exposure'))
                if not np.isnan(gamma_flip):
                    fig_profile.add_vline(x=float(gamma_flip), line_color='red', line_dash='dot')
                fig_profile.add_vline(x=call_wall, line_color='green', line_dash='dot')
                fig_profile.add_vline(x=put_wall,  line_color='blue',  line_dash='dot')
                fig_profile.add_vline(x=float(spot), line_color='yellow')
                fig_profile.update_layout(height=400, title=f'{ativo} Gamma Exposure Profile',
                                          xaxis_title='Preço Simulado', yaxis_title='Gamma Exposure')

            column = html.Div(style={'display':'flex','flexDirection':'row','padding':'10px','width':'100%',
                                     'justifyContent':'space-between','alignItems':'flex-end'}, children=[
                html.Div(style={**cp_card({'width':'48%','padding':'10px'})}, children=[
                    html.H4(ativo, style={'textAlign':'center'}), dcc.Graph(figure=fig_strike, config={'displayModeBar': False})
                ]),
                html.Div(style={**cp_card({'width':'48%','padding':'10px'})}, children=[
                    dcc.Graph(figure=fig_profile, config={'displayModeBar': False})
                ])
            ])
            columns.append(column)
        return columns

    return server, app, PREFIX

def mount_macro(shared_server: Flask | None = None, prefix: str | None = None):
    """
    Uso no host:
        from macro_app import mount_macro
        server, app, prefix = mount_macro(shared_server=server_host, prefix="/app/macro/")
    O host sobe o gunicorn/uwsgi. Não rodar server.run() aqui.
    """
    return _build_app(shared_server, prefix)

# ===================== Modo Standalone (opcional) =====================
if __name__ == "__main__":
    # Standalone: respeita APP_PREFIX (default /app/macro/) e PORT/ HOST
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8059"))
    default_prefix = os.environ.get("APP_PREFIX", "/app/macro/")
    server, app, _ = _build_app(None, default_prefix)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    server.run(host=host, port=port, debug=False, use_reloader=False)
