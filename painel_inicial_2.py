# ===================== Carbon Pro • Home Dashboard (Código Limpo) — PARTE 1 =====================
import os, math, time, random, requests, json
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import quote

import dash
from dash import dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
import yfinance as yf
import feedparser
from flask import Flask, request, session, redirect, url_for, render_template_string


# Interpolação (3D surface)
try:
    from scipy.interpolate import griddata
except Exception:
    griddata = None

# ---------------- Cache de preço rápido (quotes) ----------------
_PRICE_CACHE = {"data": {}, "ts": 0.0}
_PRICE_TTL = int(os.environ.get("PRICE_TTL", "300"))  # 5 min

def _yahoo_v7_quotes(symbols, retries=2, timeout=8):
    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
        "Accept": "application/json,text/plain,*/*",
        "Connection": "close",
        "Referer": "https://finance.yahoo.com/",
    }
    params = {"symbols": ",".join(symbols)}
    last_err = None
    for k in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"busy {r.status_code}")
            r.raise_for_status()
            js = r.json().get("quoteResponse", {}).get("result", [])
            out = {}
            for row in js:
                sym = row.get("symbol")
                px = row.get("regularMarketPrice")
                chg = row.get("regularMarketChange")
                pct = row.get("regularMarketChangePercent")
                if sym is not None:
                    out[sym] = (float(px) if px is not None else None,
                                float(chg) if chg is not None else None,
                                float(pct) if pct is not None else None)
            if out:
                return out
        except Exception as e:
            last_err = e
            time.sleep(0.35 * (2 ** k))
    if last_err:
        print(f"[Yahoo v7] falha: {last_err}")
    return {}

def _yahoo_chart_range(ticker: str, range_: str = "5d", interval: str = "1d", retries: int = 3):
    """Yahoo v8 chart → Series Close (UTC)."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
        "Accept": "application/json,text/plain,*/*",
        "Connection": "close",
        "Referer": "https://finance.yahoo.com/",
    }
    params = {"range": range_, "interval": interval, "includePrePost": "false", "corsDomain": "finance.yahoo.com"}
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=8)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"busy {r.status_code}")
            r.raise_for_status()
            j = r.json()
            res = j.get("chart", {}).get("result", [])
            if not res:
                raise RuntimeError("empty result")
            timestamps = res[0].get("timestamp", []) or []
            close = res[0].get("indicators", {}).get("quote", [{}])[0].get("close", []) or []
            if not timestamps or not close:
                raise RuntimeError("no ts/close")
            idx = pd.to_datetime(np.array(timestamps, dtype="int64"), unit="s", utc=True)
            s = pd.Series(close, index=idx, name="Close").dropna()
            if len(s) == 0:
                raise RuntimeError("all NaN")
            return s
        except Exception as e:
            last_err = e
            time.sleep(0.35 * (2 ** attempt))
    raise last_err or RuntimeError("chart fetch failed")

def _calc_snapshot_from_series(s: pd.Series):
    s = s.dropna()
    if len(s) >= 2:
        px, px0 = float(s.iloc[-1]), float(s.iloc[-2])
        chg = px - px0
        pct = (chg / px0) * 100.0 if px0 else None
        return px, chg, pct
    elif len(s) == 1:
        px = float(s.iloc[-1]); return px, None, None
    return None, None, None

def get_snapshot_quotes(symbols, range_: str = "5d", interval: str = "1d"):
    """Snapshot robusto (v7 → chart → yfinance) com TTL."""
    now = time.time()
    if (now - _PRICE_CACHE["ts"] < _PRICE_TTL) and _PRICE_CACHE["data"]:
        return {tk: _PRICE_CACHE["data"].get(tk, (None, None, None)) for tk in symbols}

    out = {s: (None, None, None) for s in symbols}

    v7 = _yahoo_v7_quotes(symbols)
    for s, tup in v7.items(): out[s] = tup

    missing = [s for s, v in out.items() if v == (None, None, None)]
    for tk in missing:
        try:
            s = _yahoo_chart_range(tk, range_=range_, interval=interval)
            out[tk] = _calc_snapshot_from_series(s)
        except Exception as e:
            print(f"[chart] {tk}: {e}")
            out[tk] = (None, None, None)

    missing2 = [tk for tk, tup in out.items() if tup == (None, None, None)]
    if missing2:
        try:
            df = yf.download(missing2, period=range_, interval=interval,
                             auto_adjust=False, threads=False, progress=False)
            for tk in missing2:
                try:
                    s = (df["Close"][tk].dropna()
                         if isinstance(df.columns, pd.MultiIndex)
                         else df["Close"].dropna())
                    out[tk] = _calc_snapshot_from_series(pd.Series(s))
                except Exception as e:
                    print(f"[yfinance] {tk}: {e}")
        except Exception as e:
            print(f"[yfinance] falha batch: {e}")

    _PRICE_CACHE["data"] = out; _PRICE_CACHE["ts"] = time.time()
    return {tk: out.get(tk, (None, None, None)) for tk in symbols}

import json
from urllib.parse import quote

def tv_embed_src(symbol="BMFBOVESPA:IBOV", interval="D"):
    """Monta o src do TradingView Advanced Chart de forma segura.
    Obs: IBOV intraday costuma retornar 'Sem dados' no embed público; use 'D'."""
    cfg = {
        "symbol": symbol,
        "interval": interval,          # <- use "D" para funcionar com IBOV
        "theme": "dark",
        "locale": "br",
        "autosize": True,
        # (estes serão úteis nas próximas etapas)
        "hide_top_toolbar": True,
        "hide_side_toolbar": True,
        "withdateranges": False,
        "allow_symbol_change": False,
        "save_image": False,
        "calendar": False,
        "studies": [],
        # host padrão do widget
        "support_host": "https://www.tradingview.com"
    }
    return "https://s.tradingview.com/embed-widget/advanced-chart/?locale=br#" + quote(json.dumps(cfg))


# --------------- Forex (Yahoo Chart + fallback) ---------------
_CHART_CACHE, _CHART_TTL = {}, 300.0
def _cache_get(key):
    item = _CHART_CACHE.get(key); 
    return None if not item else (item[1] if (time.time() - item[0]) < _CHART_TTL else None)
def _cache_set(key, obj): _CHART_CACHE[key] = (time.time(), obj)

def _yahoo_chart_period(symbol: str, period: str, interval: str):
    key = ("period", symbol, period, interval)
    hit = _cache_get(key)
    if hit is not None: return hit
    try:
        r = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"range": period, "interval": interval, "includePrePost": "false", "events": "div,splits"},
            headers={"User-Agent": "Mozilla/5.0"}, timeout=8
        )
        r.raise_for_status()
        js = r.json().get("chart", {}).get("result", [])
        if not js: return pd.Series(dtype=float)
        r0 = js[0]
        ts = r0.get("timestamp", [])
        quote = (r0.get("indicators", {}).get("quote") or [{}])[0]
        closes = quote.get("close")
        if not ts or not closes: return pd.Series(dtype=float)
        idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(None)
        s = pd.Series(closes, index=idx, name="Close").astype(float).dropna()
        _cache_set(key, s); return s
    except Exception as e:
        print(f"[chart-period] {symbol}: {e}")
        return pd.Series(dtype=float)

def baixar_serie_forex(ticker, periodo="5d", frequencia="1d"):
    s = _yahoo_chart_period(ticker, periodo, frequencia)
    if not s.empty: return s
    try:
        df = yf.download(ticker, period=periodo, interval=frequencia,
                         progress=False, auto_adjust=False, threads=False)
        if df.empty: return pd.Series(dtype=float)
        col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        return pd.to_numeric(df[col], errors="coerce").dropna()
    except Exception as e:
        print(f"[forex yfinance] {ticker}: {e}")
        return pd.Series(dtype=float)

def get_forex_snapshot(symbols):
    results = {}
    for sym in symbols:
        try:
            s = baixar_serie_forex(sym)
            if len(s) >= 2:
                pct = ((s.iloc[-1] - s.iloc[-2]) / s.iloc[-2]) * 100
                results[sym] = float(pct)
            else:
                results[sym] = None
        except Exception as e:
            results[sym] = None
            print(f"[ERRO] {sym}: {e}")
    return {k: v for k, v in results.items() if v is not None}

def fmt_num(x, nd=2):
    if x is None or (isinstance(x, float) and (np.isnan(x))): return "—"
    s = f"{x:,.{nd}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def badge_color(v): 
    if v is None: return "badge--neu"
    return "badge--pos" if v >= 0 else "badge--neg"

# ===================== PARTE 2 =====================

# ---- Fear & Greed (Crypto) ----
# ---- Fear & Greed (Crypto) com fallback e cache ----
_FNG_LAST = {"val": 50, "cls": "Neutral", "dt": datetime.utcnow()}  # valor seguro p/ render

def fetch_crypto_fng():
    urls = [
        "https://api.alternative.me/fng/?limit=1&format=json",
        "https://api.alternative.me/fng/?limit=1",                 # fallback 1
    ]
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=8)
            r.raise_for_status()
            j = r.json()
            data = j.get("data") or []
            if not data:
                continue
            row = data[0]
            val = int(row.get("value"))
            cls = (row.get("value_classification") or "").title()
            ts  = int(row.get("timestamp") or time.time())
            dt  = datetime.fromtimestamp(ts)
            _FNG_LAST.update({"val": val, "cls": cls, "dt": dt})
            return val, cls, dt
        except Exception as e:
            # tenta próxima URL
            pass
    # Se tudo falhar, retorna último bom (ou 50 Neutral)
    return _FNG_LAST["val"], _FNG_LAST["cls"], _FNG_LAST["dt"]


# ---- News (RSS) — versão única efetiva ----
import email.utils as eut
_FEEDS = [
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",   # WSJ Markets
    "https://www.investing.com/rss/news_25.rss",        # Investing.com – Markets
    "https://www.reuters.com/finance/markets/rss",      # Reuters Markets
    "https://www.moneytimes.com.br/feed/",              # BR (MoneyTimes)
]

def _as_dt(s: str):
    if not s: return None
    try:
        tt = eut.parsedate_to_datetime(s)
        return tt.astimezone(timezone.utc) if tt.tzinfo else tt.replace(tzinfo=timezone.utc)
    except Exception:
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            return None

def _ago(dt: datetime):
    if not dt: return ""
    now = datetime.now(timezone.utc)
    sec = max(0, int((now - dt).total_seconds()))
    if sec < 60:   return f"{sec}s"
    if sec < 3600: return f"{sec//60}m"
    if sec < 86400:return f"{sec//3600}h"
    return f"{sec//86400}d"

def fetch_news(limit=12):
    items = []
    for url in _FEEDS:
        try:
            feed = feedparser.parse(url)
            src  = (feed.feed.get("title") or "").strip() or url
            for e in feed.entries:
                title = getattr(e, "title", "").strip()
                link  = getattr(e, "link", "")
                dt = _as_dt(getattr(e, "published", getattr(e, "updated", "")))
                img = ""
                for key in ("media_content", "media_thumbnail", "image", "thumbnail"):
                    m = getattr(e, key, None)
                    if isinstance(m, list) and m and isinstance(m[0], dict) and m[0].get("url"):
                        img = m[0]["url"]; break
                    if isinstance(m, dict) and m.get("url"):
                        img = m["url"]; break
                items.append({
                    "title": title or "(sem título)",
                    "link":  link,
                    "source": src,
                    "published_dt": dt,
                    "ago": _ago(dt),
                    "img": img,
                })
        except Exception as ex:
            print(f"[NEWS] erro em {url}: {ex}")
    items.sort(key=lambda x: x.get("published_dt") or datetime(1970,1,1,tzinfo=timezone.utc), reverse=True)
    return items[:limit]

def _get_series_surface(ticker: str, period="1y"):
    """Obtém série de fechamento robusta via Yahoo API + fallback yfinance."""
    import requests

    # 1️⃣ Tentativa direta (API Yahoo v8)
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"range": period, "interval": "1d"}
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, params=params, headers=headers, timeout=8)
        if r.status_code == 200:
            js = r.json().get("chart", {}).get("result", [])
            if js:
                data = js[0]
                ts = data.get("timestamp", [])
                quote = (data.get("indicators", {}).get("quote") or [{}])[0]
                close = quote.get("close", [])
                if ts and close:
                    s = pd.Series(close, index=pd.to_datetime(ts, unit="s"), name="Close").dropna()
                    s = s[~s.index.duplicated(keep="last")].sort_index()
                    if len(s) >= 30:
                        return s
    except Exception as e:
        print(f"[Yahoo v8 fail] {ticker}: {e}")

    # 2️⃣ Fallback via yfinance
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            progress=False,
            threads=False,
            auto_adjust=False
        )
        if not df.empty:
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            s.index = pd.to_datetime(s.index)
            s = s[~s.index.duplicated(keep="last")].sort_index()
            if len(s) >= 30:
                return s
    except Exception as e:
        print(f"[yfinance fail] {ticker}: {e}")

    # 3️⃣ Cache local (mínimo para render)
    import numpy as np
    idx = pd.date_range(end=datetime.today(), periods=60, freq="D")
    fake = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    print(f"[Fallback usado] {ticker}")
    return fake


def build_surface_3d(a1, a2, a3, l1, l2, l3):
    """Cria a superfície 3D Z=f(X,Y) com trilha temporal e pontos atuais."""
    s1, s2, s3 = _get_series_surface(a1), _get_series_surface(a2), _get_series_surface(a3)

    # --- Alinhamento flexível (mantém ~mesmo período sem exigir dias idênticos) ---
    df = pd.concat([s1.rename("x"), s2.rename("y"), s3.rename("z")], axis=1)
    df = df.interpolate().dropna()
    if len(df) < 30:
        raise ValueError("Séries insuficientes após alinhamento temporal.")

    # --- Normalização z-score ---
    def _norm(a):
        mu, sd = np.mean(a), np.std(a)
        return (a - mu) / (sd if sd else 1.0)

    x = _norm(df["x"].to_numpy())
    y = _norm(df["y"].to_numpy())
    z = _norm(df["z"].to_numpy())
    dates = df.index

    # --- Geração de grid regular ---
    xi = np.linspace(x.min(), x.max(), 60)
    yi = np.linspace(y.min(), y.max(), 60)
    X, Y = np.meshgrid(xi, yi)

    # --- Interpolação robusta ---
    try:
        Z = griddata((x, y), z, (X, Y), method="cubic")
    except Exception:
        Z = griddata((x, y), z, (X, Y), method="linear")

    if Z is None or np.isnan(Z).all():
        raise ValueError("Falha ao gerar superfície interpolada.")

    # === FIGURA ===
    fig = go.Figure()

    # Superfície principal
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale="Viridis",
        opacity=0.9,
        showscale=True,
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
        name="Superfície"
    ))

    # Trilha temporal (pontos reais)
    hover_text = [
        f"Data: {d.strftime('%d/%m/%Y')}<br>"
        f"{l1} (norm): {xv:.3f}<br>"
        f"{l2} (norm): {yv:.3f}<br>"
        f"{l3} (norm): {zv:.3f}"
        for d, xv, yv, zv in zip(dates, x, y, z)
    ]
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines+markers",
        marker=dict(size=2, color="white", opacity=0.4),
        line=dict(color="white", width=1),
        hovertext=hover_text,
        hoverinfo="text",
        name="Trajetória temporal"
    ))

    # Ponto final destacado
    fig.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]],
        mode="markers+text",
        marker=dict(size=9, color="#FBBF24", line=dict(color="white", width=2)),
        text=f"{l3} (atual)",
        textposition="top center",
        name=l3
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis_title=f"{l1} (norm)",
            yaxis_title=f"{l2} (norm)",
            zaxis_title=f"{l3} (norm)",
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
        ),
        showlegend=False,
    )

    return fig


# ---------------- Flask + Auth ----------------
server = Flask(__name__)

# Defina o secret_key primeiro
server.secret_key = os.environ.get(
    "SECRET_KEY",
    "chave_padrao_local_troque_em_producao"
)

# ========================================================
# 1) CRIA O DASH PRINCIPAL ANTES DOS SUB-APPS ← CRÍTICO
# ========================================================
ASSETS_DIR = Path(__file__).parent / "assets"

app = dash.Dash(
    __name__,
    server=server,
    url_base_pathname="/app/",
    assets_folder=str(ASSETS_DIR),
    suppress_callback_exceptions=True,
    title="Carbon Pro • Global Market Console",
)

# ========================================================
# 2) AGORA SIM — MONTA OS SUB-APPS
# ========================================================

# Beta / Alpha
from analise_beta_alpha import mount_beta
mount_beta(server)

# Previsão
from previsao import mount_previsao
mount_previsao(server)

# Macro visão
from macro_app import mount_macro
mount_macro(server)

# EDA / Decomposição
from decomposicao import mount_eda
mount_eda(server)



def load_basic_auth_pairs() -> dict:
    env_val = os.environ.get("BASIC_AUTH_PAIRS", "").strip()
    pairs = {}

    if env_val:
        for item in env_val.split(","):
            if ":" in item:
                u, p = item.split(":", 1)
                pairs[u.strip()] = p.strip()

    # Fallback apenas para uso local
    if not pairs:
        print("[AVISO] Nenhum BASIC_AUTH_PAIRS definido. Usando credenciais padrão (somente ambiente local).")
        pairs = {"admin@atrium.com": "admin123"}

    return pairs


VALID_USERNAME_PASSWORD_PAIRS = load_basic_auth_pairs()



LOGIN_TEMPLATE = """<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8"><title>Login</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#020817;--card:#050816;--text:#E6EDF3;--muted:#8B949E;--accent:#18A0FB;--accent-soft:rgba(24,160,251,0.12);--danger:#EF4444;--border:#151B23;--radius:14px;--shadow:0 18px 45px rgba(0,0,0,.65);--font:system-ui,-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;background:
radial-gradient(900px 600px at 0% 0%,rgba(24,160,251,0.08),transparent),
radial-gradient(900px 600px at 100% 0%,rgba(45,212,191,0.06),transparent),#020817;font-family:var(--font);color:var(--text);}
.card{width:100%;max-width:380px;background:radial-gradient(140% 160% at 0% 0%,rgba(24,160,251,0.12),transparent),
radial-gradient(140% 220% at 100% 0%,rgba(45,212,191,0.10),transparent),var(--card);border:1px solid rgba(148,163,253,0.16);
border-radius:var(--radius);padding:26px 24px 22px;box-shadow:var(--shadow);backdrop-filter:blur(18px);}
h1{font-size:21px;margin:0 0 6px 0;font-weight:600;letter-spacing:.01em}
p{margin:0 0 16px 0;color:var(--muted);font-size:13px}
.field{margin-bottom:10px}
label{display:block;font-size:11px;color:var(--muted);margin-bottom:5px;font-weight:500;letter-spacing:.03em;text-transform:uppercase}
input{width:100%;padding:10px 11px;border-radius:9px;border:1px solid rgba(148,163,253,0.24);background:rgba(3,7,18,0.96);color:var(--text);outline:none;font-size:13px;}
input:focus{border-color:var(--accent);box-shadow:0 0 0 1px rgba(24,160,251,0.28),0 10px 30px rgba(15,23,42,0.9);}
.btn{width:100%;padding:10px 12px;margin-top:8px;border-radius:9px;border:0;cursor:pointer;font-weight:600;font-size:13px;letter-spacing:.02em;text-transform:uppercase;background:linear-gradient(90deg,#18A0FB,#22C55E);color:#020817;}
.btn:hover{filter:brightness(1.07)}
.error{color:var(--danger);font-size:11px;margin-top:6px}
.brand{display:flex;align-items:center;justify-content:center;margin-bottom:10px}
.brand img.logo{height:40px;display:block;margin:0 auto 8px auto}
.foot{margin-top:10px;text-align:center;font-size:10px;color:var(--muted)}
.hint{font-size:10px;color:var(--muted);margin-top:8px;text-align:center}
</style>
</head>
<body>
  <div class="card">
    <div class="brand"><img src="{{ asset('globalstat_logo.png') }}" alt="GlobalStat" class="logo"></div>
    <h1>GlobalStat Console</h1>
    <p>Acesse o painel exclusivo da Atrium com visão profissional de mercados.</p>
    <form method="POST" action="{{ url_for('login') }}">
      <div class="field"><label>E-mail</label><input name="username" type="email" placeholder="voce@dominio.com" required autofocus></div>
      <div class="field"><label>Senha</label><input name="password" type="password" placeholder="********" required></div>
      <button class="btn" type="submit">Entrar</button>
      {% if error %}<div class="error">{{ error }}</div>{% endif %}
    </form>
    <div class="hint">Configure via <code>BASIC_AUTH_PAIRS</code> no ambiente.</div>
    <div class="foot">© {{ year }} GlobalStat • Atrium</div>
  </div>
</body></html>"""

@server.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("username"); pw = request.form.get("password")
        if user in VALID_USERNAME_PASSWORD_PAIRS and pw == VALID_USERNAME_PASSWORD_PAIRS[user]:
            session["logged_in"] = True; session["user"] = user; session.pop("last_page", None)
            resp = redirect("/app/"); resp.set_cookie("dash_pathname", "/", max_age=0)
            return resp
        return render_template_string(LOGIN_TEMPLATE, error="Credenciais inválidas",
                                      asset=app.get_asset_url, year=datetime.now().year)
    return render_template_string(LOGIN_TEMPLATE, error=None,
                                  asset=app.get_asset_url, year=datetime.now().year)

@server.route("/logout")
def logout(): session.clear(); return redirect("/login")
@server.route("/")
def root_redirect(): return redirect("/login")



# Tema Plotly (igual ao atual)
BG = "#020817"; PANEL = "#050816"; LINE = "#151B23"; TEXT = "#E6EDF3"
PRIMARY = "#18A0FB"; ACCENT = "#22C55E"; NEUTRAL = "#9CA3AF"; EMERALD = "#22C55E"; AMBER = "#FBBF24"; ROSE = "#F87171"
GRID_COLOR = "rgba(148,163,253,0.14)"
pio.templates["carbon_pro"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="system-ui, -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"),
        xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, linecolor=LINE, tickcolor=LINE, color=TEXT),
        yaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, linecolor=LINE, tickcolor=LINE, color=TEXT),
        margin=dict(l=40, r=24, t=40, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(color=TEXT)),
        colorway=[PRIMARY, ACCENT, EMERALD, NEUTRAL, AMBER, ROSE],
    )
)
pio.templates.default = "carbon_pro"
@server.before_request
def require_login():
    path = request.path or "/"

    # 1) Páginas livres
    if path in {"/login", "/favicon.ico"}:
        return

    # 2) Tudo que o Dash precisa para carregar (JS/CSS/API internas)
    if path.startswith((
        "/assets", "/static", "/app/assets",      # seus assets
        "/_dash", "/app/_dash",                   # layout, dependencies, updates
        "/_favicon", "/_dash-component-suites"    # bundles de componentes
    )):
        return

    # 3) Conteúdo do app
    if path.startswith("/app/"):
        return

    # 4) Proteção (se quiser exigir login para outras rotas)
    if not session.get("logged_in"):
        return redirect("/login")

@server.route("/app/")
def dash_root_redirect():
    """Garante que /app/ carrega o layout principal do Dash"""
    return app.index()
# ===================== PARTE 4 =====================

CURRENCIES = {
    "British Pound (GBPUSD)": "GBPUSD=X",
    "Japanese Yen (USDJPY)": "JPY=X",
    "Euro (EURUSD)": "EURUSD=X",
    "Bitcoin": "BTC-USD",
}
# --- World Markets por CLASSE financeira (substitui GLOBAL_MARKETS*) ---
GLOBAL_MARKETS_BY_CLASS = {
    "Equities / Indexes": {
        "USA – S&P 500": "^GSPC",
        "USA – Nasdaq 100": "^NDX",
        "USA – Dow Jones": "^DJI",
        "Brazil (Ibovespa)": "^BVSP",
        "Euro Stoxx 50": "^STOXX50E",
        "Germany (DAX)": "^GDAXI",
        "UK (FTSE 100)": "^FTSE",
        "France (CAC 40)": "^FCHI",
        "Japan (Nikkei 225)": "^N225",
        "Hong Kong (Hang Seng)": "^HSI",
        "India (Nifty 50)": "^NSEI",
        "Australia (ASX 200)": "^AXJO",
    },

    "Commodities": {
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Crude Oil (WTI)": "CL=F",
        "Brent": "BZ=F",
        "Natural Gas": "NG=F",
        "Copper": "HG=F",
    },

    "Forex": {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
        "USD/BRL": "BRL=X",
        "USD/CAD": "CAD=X",
        "USD/CHF": "CHF=X",
        "AUD/USD": "AUDUSD=X",
        "NZD/USD": "NZDUSD=X",
    },

    "Crypto": {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "Solana": "SOL-USD",
        "BNB": "BNB-USD",
        "XRP": "XRP-USD",
        "Cardano": "ADA-USD",
    },

    "Rates & Vol": {
        "US 10Y Yield": "^TNX",
        "VIX": "^VIX",
    },
}

COMMODITIES = {"Brent Crude": "BZ=F","Crude Oil (WTI)": "CL=F","Gold": "GC=F","Silver": "SI=F","Natural Gas": "NG=F"}
CENTER_DEFAULT = ["^DJI", "^GSPC", "^NDX"]
TV_SYMBOLS = {"^DJI": "TVC:DJI","^GSPC": "SP:SPX","^NDX": "NASDAQ:NDX","^BVSP": "BMFBOVESPA:IBOV","GC=F": "TVC:GOLD","BTC-USD":"CRYPTO:BTCUSD"}
KPI_SYMBOLS = [("^GSPC","S&P 500"),("^NDX","Nasdaq 100"),("BTC-USD","Bitcoin"),("BZ=F","Brent"),("^TNX","US 10Y"),("^VIX","VIX")]
FOREX_PAIRS = {"EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X","USD/JPY":"JPY=X","USD/BRL":"BRL=X","USD/CAD":"CAD=X","USD/CHF":"CHF=X",
               "AUD/USD":"AUDUSD=X","NZD/USD":"NZDUSD=X","USD/CNY":"CNY=X","USD/MXN":"MXN=X"}

# ===================== PARTE 5 — CSS EFETIVO (ATUALIZADA) =====================
CARBON_PRO_CSS = r"""
<style>
:root { --bg:#0b0c0e; --panel:rgba(255,255,255,0.02); --line:#1a1b1e; --text:#e8edf2; --muted:#aab3c0;
        --accent:#59f0c8; --positive:#59f0c8; --negative:#ef4444; --warning:#fbbf24;
        --shadow:0 0 28px rgba(255,255,255,0.015),0 6px 20px rgba(0,0,0,0.45);
        --radius:16px; --font:"Inter",system-ui,-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
html, body { min-height:100%; margin:0; padding:0; background:var(--bg)!important; color:var(--text)!important; font-family:var(--font)!important; overflow:hidden; }
.layout-shell { display:flex; flex-direction:row; height:100vh; width:100%; background:linear-gradient(135deg,#0e1117 0%,#1b1e24 100%); overflow:hidden; }
.grid { display:grid; grid-template-columns:310px minmax(640px, 2fr) 360px !important; grid-template-rows:auto 1fr; grid-gap:12px; flex:1; width:100%; overflow:hidden; gap:10px; height:100%; min-height:0; }
.col{
  display:flex; flex-direction:column; gap:10px;
  overflow-y:visible; overflow-x:hidden;
  padding-bottom:8px;
  min-height:0;
  scrollbar-width:thin; scrollbar-color:rgba(255,255,255,0.08) transparent;
}
.col::-webkit-scrollbar{ width:6px; } .col::-webkit-scrollbar-thumb{ background:rgba(255,255,255,0.08); border-radius:3px; }
.col--left{ grid-column:1; grid-row:2; } .col--center{ grid-column:2; grid-row:2; } .col--right{ grid-column:3; grid-row:1 / 3; }
.card { background:var(--panel); border:1px solid rgba(255,255,255,0.05); border-radius:var(--radius); box-shadow:var(--shadow);
        padding:10px 12px 12px; display:flex; flex-direction:column; overflow:hidden; flex:1 1 auto; }
.card__title{ font-size:12px; font-weight:600; color:var(--accent); margin-bottom:4px; letter-spacing:.05em; }
.card__body{ font-size:12px; color:var(--muted); }
.js-plotly-plot,.plot-container,.main-svg,.svg-container,.cartesianlayer,.plotly .bg,.plotly .main-svg rect.bg{ background:transparent!important; fill:transparent!important; }
.xgrid,.ygrid,.zgrid { stroke:rgba(255,255,255,0.05)!important; }
.tradingview-widget-container iframe, #tv-chart { background:#0b0c0e!important; border-radius:14px!important; }
#market-map-card { align-self:stretch!important; width:100%!important; max-width:100%!important; aspect-ratio:1/1!important; height:auto!important; min-height:0!important; flex:0 0 auto; }
#market-map-body { width:100%!important; height:100%!important; }
.col--right .card:first-child { height:100px!important; min-height:100px!important; max-height:100px!important; overflow:hidden; padding-bottom:4px; }
@media (max-width:1200px){ .grid{ grid-template-columns:1fr; height:auto; } .col--right{ flex-direction:row; } .col--right .card{ flex:1; min-height:240px; } }
</style>
"""

SIDEBAR_CSS = r"""
<style>
:root{
  --sidebar-w:220px;
  --sidebar-w-collapsed:64px;
  --accent:#59f0c8;
  --bg:#0b0c0e;
  --text:#e8edf2;
  --muted:#aab3c0;
  --radius:16px;
  --font:"Inter",system-ui,-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}

.sidebar{
  width:var(--sidebar-w);
  position:fixed;
  inset:0 auto 0 0;
  z-index:1000;
  background:#000;
  backdrop-filter:blur(10px);
  border-right:1px solid rgba(255,255,255,0.05);
  display:flex;
  flex-direction:column;
  box-shadow:0 0 28px rgba(255,255,255,0.015),0 6px 20px rgba(0,0,0,0.45);
  transition:width .25s ease;
  overflow:hidden;
}

.sidebar__head{
  display:flex;
  align-items:center;
  gap:10px;
  padding:14px 16px 10px;
  border-bottom:1px solid rgba(255,255,255,0.05);
}

.sidebar__brand{
  display:flex;
  align-items:center;
  gap:10px;
  color:var(--text);
  font-weight:800;
  font-size:19px;
  cursor:pointer;
}

.sidebar__brand img{
  height:42px;
  width:42px;
  border-radius:10px;
  box-shadow:0 0 18px rgba(89,240,200,0.3);
}

.sidebar__toggle{
  margin-left:auto;
  cursor:pointer;
  border:1px solid rgba(255,255,255,0.08);
  background:rgba(255,255,255,0.02);
  color:var(--muted);
  border-radius:8px;
  padding:4px 6px;
  font-size:11px;
}

.sidebar__body{
  flex:1;
  padding:10px 8px 14px;
  overflow-y:auto;
  scrollbar-width:thin;
  scrollbar-color:rgba(255,255,255,0.1) transparent;
}

.menu-item{
  display:flex;
  align-items:center;
  gap:8px;
  width:88%;
  margin:0 auto 6px auto;
  color:var(--muted);
  text-decoration:none;
  font-weight:600;
  padding:10px 12px;
  border-radius:10px;
  font-size:15px;
  border:1px solid transparent;
}

.menu-item:hover{
  background:rgba(255,255,255,0.03);
  border-color:rgba(255,255,255,0.05);
  color:var(--text);
  transform:translateX(3px);
}


/* ========================= ⬇️ ALTERAÇÃO CRUCIAL ⬇️ ========================= */

.sidebar--collapsed{
  width:var(--sidebar-w-collapsed);
}

/* esconde texto do logo, botão e dos ITENS DE MENU */
.sidebar--collapsed .sidebar__brand-text,
.sidebar--collapsed .sidebar__toggle,
.sidebar--collapsed .menu-item__text{
  display:none !important;
}

/* quando colapsado, só ícone centralizado */
.sidebar--collapsed .menu-item{
  width:42px;
  justify-content:center;
  gap:0;
  padding:8px 0;
}

/* ========================= ⬆️ ALTERAÇÃO CRUCIAL ⬆️ ========================= */


.main{
  flex:1;
  margin-left:calc(var(--sidebar-w) + 12px);
  margin-right:12px;
  padding:6px 0 12px 0!important;
  height:100vh!important;
  display:flex;
  flex-direction:column;
  align-items:flex-start;
  background:linear-gradient(135deg,#0e1117 0%,#1b1e24 100%);
  transition:margin-left .25s ease;
}

.main--collapsed{
  margin-left:calc(var(--sidebar-w-collapsed) + 12px);
  margin-right:12px;
  padding:8px 0 12px 0!important;
}

@media (max-width:1200px){
  .sidebar{position:relative;height:auto;}
  .main{margin-left:0!important;padding:8px;}
}
</style>
"""


# Patches efetivos
MARKETMAP_PATCH = r"""
<style>
.col--left #market-map-card{
  flex:0 0 auto!important;
  width:100%!important;
  max-width:100%!important;
  height:310px!important;
  aspect-ratio:1/1!important;
  box-sizing:border-box!important;
}
.col--left #market-map-card .card__body{ flex:1 1 auto!important; height:100%!important; padding:0!important; }
#market-map-body, #market-map-body > *{ width:100%!important; height:100%!important; }
</style>
"""

MARKET_MAP_CSS_FIX = r"""
<style>
#market-map-card .card__body,
#market-map-body,
#market-map-body .plot-container,
#market-map-body .svg-container,
#market-map-body .main-svg,
#market-map-body .bglayer rect,
#market-map-body .bglayer path,
#market-map-body path.bg,
#market-map-body rect.bg {
  background: transparent !important;
  fill: transparent !important;
  stroke: none !important;
}
</style>
"""

SELECT_CARBON_CSS = r"""
<style>
.center-sidecol .card { overflow: visible !important; }

.cp-select {
  min-width: 180px !important;
  max-width: 260px !important;
}

.cp-select .Select-control {
  background: transparent !important;
  border: none !important;
  border-bottom: 1px solid rgba(255,255,255,0.18) !important;
  border-radius: 0 !important;
  height: 34px !important;
  transition: all .2s ease;
  box-shadow: none !important;
}
.cp-select .Select-control:hover {
  border-bottom-color: var(--accent) !important;
}

/* ======== COR DO TEXTO — FINALMENTE FIXADO ======== */
.cp-select .Select--single > .Select-control .Select-value,
.cp-select .Select--single > .Select-control .Select-value-label {
  color: #ffffff !important;      /* texto branco visível */
  font-weight: 600 !important;
  opacity: 1 !important;
}

/* placeholder branco quando nada selecionado */
.cp-select .Select-placeholder {
  color: rgba(255,255,255,0.85) !important;
}

/* estados foco/aberto */
.cp-select .is-focused .Select-value-label,
.cp-select .is-open .Select-value-label,
.cp-select .is-focused .Select-placeholder,
.cp-select .is-open .Select-placeholder {
  color: #ffffff !important;
  opacity: 1 !important;
}

/* opções dentro do dropdown */
.cp-select .Select-menu-outer {
  background: rgba(15,17,20,0.98) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 10px !important;
  box-shadow: 0 8px 28px rgba(0,0,0,0.55) !important;
  z-index: 9999 !important;     /* garante que fica acima dos outros cards */
}
.cp-select .Select-option {
  color: #ffffff !important;     /* texto branco dentro das opções */
  font-size: 13px !important;
  padding: 7px 10px !important;
  transition: background .15s ease, color .15s ease;
}
.cp-select .Select-option:hover {
  background: rgba(89,240,200,0.15) !important;
  color: var(--accent) !important;
}

/* setinha */
.cp-select .Select-arrow {
  border-top-color: rgba(255,255,255,0.6) !important;
}
</style>
"""




NEWS_CSS = r"""
<style>
.news-item--text{
  padding: 10px 12px;
  border-radius: 10px;
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.06);
  max-width: 100%;
}
.news-item--text a{
  color: #E6EDF3;
  text-decoration: none;
  font-weight: 600;
  font-size: 14px;
  display: block;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
/* ---- NEWS FIX DEFINITIVO ---- */
#news-card{ 
  overflow: visible !important;
  min-height: 0 !important;
}
#news-card .card__body{
  display: flex !important;
  flex-direction: column !important;
  flex: 1 1 auto !important;
  min-height: 0 !important;
  padding: 0 !important;
}
/* o UL é quem rola; some a barra mas mantém a rolagem */
#news-list{
  flex: 1 1 auto !important;
  min-height: 0 !important;
  overflow-y: auto !important;
  padding: 10px 15px 28px !important;
  margin: 0 !important;
  scrollbar-width: none !important;
}
#news-list::-webkit-scrollbar{ display: none; }
/* garante espaço visual no fim */
#news-list li:last-child{ margin-bottom: 22px !important; }
</style>
"""

# KPI (tamanho final) + margem inferior e grid
KPI_SIZE_FIX = r"""
<style>
#kpi-area{
  grid-column: 1 / 3 !important;
  grid-row: 1 !important;
  position: relative;
  display: flex;
  align-items: center;
  width: 100%;
  min-height: 100px;
}
.kpi-viewport{ position: relative; width: 100%; overflow: hidden; }
.kpi-strip{
  display:flex; flex-direction:row; align-items:center!important; justify-content:flex-start;
  gap:12px!important; padding-right:6px; width:max-content; scroll-behavior:smooth;
  overflow-x:auto; overflow-y:hidden; -ms-overflow-style:none; scrollbar-width:none;
}
.kpi-strip::-webkit-scrollbar{ display:none; }
.kpi-card{
  min-width:260px!important; height:80px!important; padding:16px 20px!important;
  border-radius:12px; background:var(--panel); border:1px solid rgba(255,255,255,0.05);
  box-shadow:var(--shadow); display:flex; flex-direction:column; justify-content:center;
}
.kpi-top{ display:flex; justify-content:space-between; align-items:center; gap:8px; font-size:13px; line-height:1; margin-bottom:6px; }
.kpi-top span:first-child{ font-weight:600; letter-spacing:.01em; }
.kpi-top span:last-child{ opacity:.7; font-size:12px; }
.kpi-val{ font-size:20px; font-weight:800; line-height:1.1; margin-top:2px; }
.kpi-badge{ margin-top:8px; display:inline-block; padding:3px 8px; border-radius:999px; font-size:12px; font-weight:700; border:1px solid transparent; }
.kpi-badge.pos{ background:rgba(89,240,200,.12); color:#59f0c8; border-color:rgba(89,240,200,.35); }
.kpi-badge.neg{ background:rgba(239,68,68,.12); color:#ef4444; border-color:rgba(239,68,68,.35); }
.kpi-badge.neu{ background:rgba(148,163,184,.12); color:#94a3b8; border-color:rgba(148,163,184,.25); }
.kpi-nav{
  position:absolute; top:50%; transform:translateY(-50%); width:28px; height:28px; border-radius:999px;
  border:1px solid rgba(255,255,255,0.15); background:rgba(255,255,255,0.06); color:#e6edf3; display:flex; align-items:center; justify-content:center;
  font-size:14px; line-height:1; box-shadow:0 4px 16px rgba(0,0,0,0.35); opacity:0; pointer-events:none; transition:opacity .18s ease, transform .18s ease, background .18s ease, border-color .18s ease; user-select:none; z-index:2;
}
#kpi-area:hover .kpi-nav{ opacity:1; pointer-events:auto; }
.kpi-nav:hover{ background:rgba(255,255,255,0.12); border-color:rgba(89,240,200,0.35); }
.kpi-nav--left{ left:6px; } .kpi-nav--right{ right:6px; }
#kpi-area{ margin-bottom:0px!important; }
</style>
"""




KPI_BOTTOM_MARGIN_FIX = r"""
<style>
.kpi-strip{ margin-bottom:17px!important; padding-bottom:8px!important; }
.col--left{ grid-row:2!important; } .col--center{ grid-row:2!important; } .col--right{ grid-row:1 / 3!important; }
.grid{ grid-auto-rows:min-content; align-items:stretch!important; }
</style>
"""

# (REMOVIDO o FEAR_GREED_CSS antigo)

# ==== CENTER_LAYOUT_FIX (mantido) ====
CENTER_LAYOUT_FIX = r"""
<style>
.col--center > div:first-child{
  display:flex!important;
  align-items:stretch!important;
  gap:12px!important;
}
.col--center > div:first-child > .card{
  flex:6 1 0% !important;
  display:flex; flex-direction:column;
}
.center-sidecol{
  flex:1.6 1 0% !important;
  min-width:520px !important;
}
</style>
"""

CSS_CHART = r"""
<style>
:root{ --chart-h: 600px; }
#chart-card{ height: var(--chart-h) !important; min-height: var(--chart-h) !important; }
#chart-card .card__body{ display:flex!important; flex:1 1 auto!important; min-height:0!important; padding:0!important; }
#chart-container{ height:100%!important; min-height:0!important; width:100%!important; }
#tv-chart{ width:100%!important; height:100%!important; border:0!important; border-radius:14px!important; background:#0b0c0e!important; }
</style>
"""

TABS_CARBON_CSS = r"""
<style>
.dash-tabs{ background:transparent!important; border:0!important; padding:0 0 6px 0!important; display:flex!important; align-items:flex-end!important; }
.dash-tabs .tab{
  background:rgba(255,255,255,0.02)!important; color:#aab3c0!important; border:1px solid rgba(255,255,255,0.06)!important; border-bottom:2px solid rgba(255,255,255,0.06)!important;
  border-radius:12px 12px 0 0!important; padding:10px 16px!important; margin:0 8px 0 0!important; font-weight:600!important; font-size:13px!important; letter-spacing:.02em!important;
  transition:all .18s ease; box-shadow:0 6px 16px rgba(0,0,0,0.35)!important;
}
.dash-tabs .tab:hover{ color:#e6edf3!important; border-color:rgba(255,255,255,0.12)!important; }
.dash-tabs .tab--selected{
  background:linear-gradient(180deg,rgba(89,240,200,0.12),rgba(255,255,255,0.02))!important; color:#e6edf3!important; border-color:rgba(89,240,200,0.35)!important;
  box-shadow:inset 0 -2px 0 #59f0c8, 0 8px 20px rgba(0,0,0,0.45)!important;
}
.dash-tabs .tab--selected::after{ content:""; display:block; height:2px; width:100%; background:#59f0c8; border-radius:2px; position:relative; top:8px; }
</style>
"""

# ==== MOVE_3D_AND_FNG_CSS (sem overrides do FNG) ====
MOVE_3D_AND_FNG_CSS = r"""
<style>
.center-sidecol{ flex:1.4 1 0% !important; min-width:480px !important; }
.col--center > div:first-child > .card{ flex:6 1 0% !important; }
#surface-3d{ height:460px !important; }
</style>
"""
CLOCK_POPOVER_CSS = r"""
<style>
#clock-anchor{
  position:absolute; top:6px; right:380px;   /* ajuste conforme sua coluna direita */
  z-index:1200; pointer-events:none;
}
#clock-trigger{
  pointer-events:auto; display:flex; align-items:center; justify-content:center;
  width:28px; height:28px; border-radius:999px; cursor:pointer;
  border:1px solid rgba(255,255,255,.15); background:rgba(255,255,255,.06); color:#e6edf3;
  box-shadow:0 4px 16px rgba(0,0,0,.35); font-size:14px;
}
#clock-trigger:hover{ background:rgba(255,255,255,.12); border-color:rgba(89,240,200,.35); }

#clock-pop{
  position:absolute; top:42px; right:0;
  width:520px; height:520px; border-radius:16px; overflow:hidden;
  background:rgba(15,17,20,.98);
  border:1px solid rgba(255,255,255,.08);
  box-shadow:0 28px 80px rgba(0,0,0,.65);
  display:none; pointer-events:auto;
}
#clock-pop.show{ display:block; }

#clock-pop .pop-head{
  display:flex; align-items:center; justify-content:space-between;
  padding:10px 14px; border-bottom:1px solid rgba(255,255,255,.06);
}
#clock-pop .pop-title{ color:var(--accent); font-weight:700; letter-spacing:.02em; }
#clock-pop .pop-close{
  width:28px; height:28px; border-radius:999px; cursor:pointer;
  border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.06); color:#e6edf3;
}
#clock-pop .pop-close:hover{ background:rgba(255,255,255,.12); border-color:rgba(89,240,200,.35); }

#clock-pop .pop-body{ height:calc(100% - 50px); padding:8px; }
#market-clock-big{ width:100% !important; height:100% !important; }

/* mini clock sem eixos e com fundo transparente */
#market-clock .cartesianlayer, #market-clock .yaxislayer-above, #market-clock .xaxislayer-above{ display:none !important; }
#market-clock .bg, #market-clock .bglayer rect { fill:transparent !important; }
</style>
"""

# ====== FNG FINAL — RAIO MAIOR DO ARCO (substitui qualquer CSS antigo do FNG) ======
CSS_FNG_FINAL = r"""
<style>
:root{
  --FNG_CARD: 120px;     /* altura visível do card */
  --FNG_CANVAS: 200px;   /* altura real do canvas (controla o RAIO) */
  --FNG_SHIFT: 20px;   /* desloca o canvas para cima dentro do card */
}
#fear-greed-card{
  height: var(--FNG_CARD) !important;
  min-height: var(--FNG_CARD) !important;
  max-height: var(--FNG_CARD) !important;
  padding: 6px 8px !important;
}
#fear-greed-card .card__title{ margin:0 0 4px 0!important; font-size:12px!important; }
.fg-wrap{ position:relative; height:calc(var(--FNG_CARD) + 40px); overflow:hidden; margin-top: -20px;}
.fg-canvas{
  height: var(--FNG_CANVAS) !important;   /* maior = RAIO maior */
  width: 100% !important;
  transform: translateY(var(--FNG_SHIFT)) !important; /* centraliza visualmente */
}
</style>
"""

WORLD_MARKETS_CSS_FIX = r"""
<style>
#world-markets-card {
  flex: 1 1 auto !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: space-between !important;
  align-items: stretch !important;
  height: 100% !important;
  min-height: 260px !important;
  padding: 10px 14px 14px !important;
}
#world-markets-card .card__title {
  color: var(--accent) !important;
  font-size: 13px !important;
  font-weight: 700 !important;
  letter-spacing: 0.03em !important;
  text-transform: none !important;
  margin-bottom: 8px !important;
}
.wm-table {
  width: 100% !important;
  border-collapse: collapse !important;
  table-layout: fixed !important;
  font-size: 13px !important;
  color: var(--text) !important;
}
.wm-table th {
  text-align: left !important;
  color: var(--muted) !important;
  font-weight: 600 !important;
  padding-bottom: 6px !important;
  border-bottom: 1px solid rgba(255,255,255,0.06) !important;
}
.wm-table td {
  padding: 6px 0 !important;
  border-bottom: 1px solid rgba(255,255,255,0.03) !important;
}
.wm-table td:first-child {
  width: 60% !important;
  color: var(--text) !important;
  font-weight: 500 !important;
}
.wm-table td:nth-child(2),
.wm-table td:nth-child(3) {
  width: 20% !important;
  text-align: right !important;
  font-variant-numeric: tabular-nums !important;
}
.wm-pager {
  display: flex !important;
  justify-content: center !important;
  align-items: center !important;
  gap: 10px !important;
  margin-top: 10px !important;
}
.wm-btn {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  color: var(--text) !important;
  font-size: 12px !important;
  border-radius: 8px !important;
  padding: 4px 8px !important;
  cursor: pointer !important;
  transition: all .18s ease !important;
}
.wm-btn:hover {
  background: rgba(255,255,255,0.08) !important;
  border-color: rgba(89,240,200,0.35) !important;
  color: var(--accent) !important;
}
.wm-dot { width:7px; height:7px; border-radius:50%; background:rgba(255,255,255,0.12); transition:all .2s ease; }
.wm-dot--active { background:var(--accent); box-shadow:0 0 8px rgba(89,240,200,0.5); }
#world-markets { flex:1 1 auto !important; display:flex !important; flex-direction:column !important; justify-content:space-between !important; }
</style>
"""

# World Markets pode ser inserido separado


# ======= CORREÇÃO ÚNICA DO INDEX_STRING =======
extra_css_blocks = (
    CARBON_PRO_CSS
    + SIDEBAR_CSS
    + MARKETMAP_PATCH
    + MARKET_MAP_CSS_FIX
    + SELECT_CARBON_CSS
    + NEWS_CSS
    + KPI_SIZE_FIX
    + KPI_BOTTOM_MARGIN_FIX
    + TABS_CARBON_CSS
    + CENTER_LAYOUT_FIX
    + MOVE_3D_AND_FNG_CSS
    + CLOCK_POPOVER_CSS
    + CSS_CHART
    + CSS_FNG_FINAL
    + WORLD_MARKETS_CSS_FIX
)

app.index_string = app.index_string.replace(
    "</head>", extra_css_blocks + "</head>"
)

# Corrige template base para garantir renderização (sem tela preta)
app.index_string = app.index_string.replace(
    "<body>", "<body><div id='root'>"
).replace("</body>", "</div></body>")
# ===================== PARTE 6 — Sidebar e Layouts =====================

def sidebar_menu():
    def icon(txt): return html.Span(txt, className="menu-item__icon",
                                    style={"width":"14px","textAlign":"center","fontSize":"13px","opacity":".9"})
    return html.Div(
        id="sidebar", className="sidebar",
        children=[
            html.Div(
                className="sidebar__head",
                children=[
                    html.Div(id="brand-toggle", className="sidebar__brand",
                             children=[html.Img(src=app.get_asset_url("globalstat_miniatura.png")),
                                       html.Span("GlobalStat", className="sidebar__brand-text")]),
                    html.Button("⟨⟩", id="sidebar-toggle", n_clicks=0, className="sidebar__toggle", title="recolher/expandir"),
                ],
            ),
            html.Div(className="sidebar__body", children=[
                dcc.Link([icon("●"), html.Span("Correlação", className="menu-item__text")], href="/app/beta/", className="menu-item", refresh=False),
                dcc.Link([icon("↗"), html.Span("Previsão", className="menu-item__text")], href="/app/previsao/", className="menu-item", refresh=False),
                dcc.Link([icon("🌐"), html.Span("Macro visão", className="menu-item__text")], href="/app/macro/", className="menu-item", refresh=False),
                dcc.Link([icon("⤮"), html.Span("Correlação cruzada", className="menu-item__text")], href="/app/beta/", className="menu-item", refresh=False),
                dcc.Link([icon("▥"), html.Span("Análise exploratória", className="menu-item__text")], href="/app/eda/", className="menu-item", refresh=False),
            ]),
        ],
    )

def layout_fear_greed():
    return html.Div(
        className="card", id="fear-greed-card",
        children=[html.Div("Fear & Greed Index", className="card__title"),
                  html.Div(className="card__body",
                           children=html.Div(id="fear-greed", children="Carregando…",
                                             className="card-placeholder",
                                             style={"height":"170px","display":"flex","alignItems":"center","justifyContent":"center"}))]
    )

def layout_world_markets():
    return html.Div(
        className="card", id="world-markets-card",
        children=[
            html.Div("World Markets", className="card__title"),
            html.Div(id="world-markets", className="card__body", children=[
                html.Div(className="wm-viewport", id="wm-viewport"),
                html.Div(className="wm-pager", children=[
                    html.Button("◀", id="wm-prev", n_clicks=0, className="wm-btn"),
                    html.Div(id="wm-dots", style={"display":"flex","gap":"6px"}),
                    html.Button("▶", id="wm-next", n_clicks=0, className="wm-btn"),
                ]),
            ]),
        ],
    )

def get_home_layout():
    return html.Div(
        className="layout-shell",
        children=[
            # Stores
            dcc.Store(id="sidebar-collapsed", data=False),
            dcc.Store(id="selected-assets", data=CENTER_DEFAULT, storage_type="session"),
            dcc.Store(id="tv-symbol", data=TV_SYMBOLS.get(CENTER_DEFAULT[0], "SP:SPX"), storage_type="session"),
            dcc.Store(id="kpi-cache", data={}),
            dcc.Store(id="ta-config", data={"ma_fast": 9, "ma_slow": 21, "rsi": 14}),
            dcc.Store(id="filters-news", data={"q": "", "src": []}),
            dcc.Store(id="filters-eco", data={"countries": ["US", "BR", "GB", "EU"], "importance": ["high", "medium"]}),
            dcc.Store(id="alerts", data=[]),
            dcc.Store(id="wm-page", data=0),

            # Intervals
            dcc.Interval(id="tick-fast", interval=30_000, n_intervals=0),
            dcc.Interval(id="tick-slow", interval=5 * 60_000, n_intervals=0),
            dcc.Interval(id="tick", interval=60_000, n_intervals=0),
            dcc.Interval(id="news-tick", interval=3 * 60_000, n_intervals=0),
            dcc.Interval(id="wm-rot", interval=6000, n_intervals=0),

            # Sidebar
            sidebar_menu(),

            # Main
            html.Div(
                id="main",
                className="main",
                children=[
                    html.Div(
                        className="grid",
                        style={"padding": "0", "marginRight": "0", "marginLeft": "0"},
                        children=[
                            # KPI Strip (linha 1, colunas 1–2)
                            html.Div(
                                id="kpi-area",
                                children=[
                                    html.Button("◀", id="kpi-prev", n_clicks=0, className="kpi-nav kpi-nav--left", title="Anterior"),
                                    html.Div(
                                        className="kpi-viewport",
                                        children=html.Div(
                                            id="kpi-strip",
                                            className="kpi-strip",
                                            children=[
                                                html.Div(
                                                    "KPI Placeholder — índices de mercado",
                                                    className="card-placeholder",
                                                    style={"height": "72px", "display": "flex", "alignItems": "center", "justifyContent": "center"},
                                                )
                                            ],
                                        ),
                                    ),
                                    # === ÂNCORA + TRIGGER + POPOVER (abre sobre a área amarela) ===
                                    html.Div(
                                        id="clock-anchor",
                                        children=[
                                            html.Button("▾", id="clock-trigger", n_clicks=0, title="Relógio de Mercados"),
                                            html.Div(
                                                id="clock-pop",
                                                children=[
                                                    html.Div(
                                                        className="pop-head",
                                                        children=[
                                                            html.Span("Relógio de Mercados", className="pop-title"),
                                                            html.Button("✕", id="clock-close", n_clicks=0, className="pop-close", title="Fechar"),
                                                        ],
                                                    ),
                                                    html.Div(
                                                        className="pop-body",
                                                        children=dcc.Graph(id="market-clock-big", config={"displayModeBar": False}),
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),

                                    html.Button("▶", id="kpi-next", n_clicks=0, className="kpi-nav kpi-nav--right", title="Próximo"),
                                    dcc.Store(id="kpi-scroll", data=0),
                                ],
                            ),



                            # Coluna Esquerda
                            html.Div(
                                className="col col--left",
                                style={"display": "flex", "flexDirection": "column", "justifyContent": "flex-start", "gap": "10px",
                                       "height": "calc(100vh - 180px)", "minHeight": "180px"},
                                children=[
                                    html.Div(
                                        className="card",
                                        id="market-map-card",
                                        style={"width": "100%", "aspectRatio": "1 / 1", "height": "auto", "maxWidth": "100%", "alignSelf": "stretch",
                                               "display": "flex", "flexDirection": "column", "justifyContent": "flex-start", "boxSizing": "border-box",
                                               "overflow": "hidden", "flex": "0 0 auto"},
                                        children=[
                                            html.Div("Market Map", className="card__title"),
                                            html.Div(
                                                id="market-map-body",
                                                className="card__body",
                                                style={"flex": "1", "display": "flex", "alignItems": "stretch", "justifyContent": "center",
                                                       "padding": "0", "margin": "0", "height": "100%", "overflow": "hidden"},
                                                children=html.Div(
                                                    "Carregando mapa setorial...",
                                                    className="card-placeholder",
                                                    style={"width": "100%", "height": "100%", "display": "flex", "alignItems": "center", "justifyContent": "center"},
                                                ),
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="card",
                                        id="financial-calendar-card",
                                        style={"flex": "1 1 auto", "minHeight": "0"},
                                        children=[
                                            html.Div("Financial Calendar", className="card__title"),
                                            html.Div(
                                                className="card__body",
                                                style={"flex": "1", "display": "flex"},
                                                children=html.Iframe(
                                                    id="eco-calendar",
                                                    src=("https://s.tradingview.com/embed-widget/events/?locale=br#%7B"
                                                         "%22colorTheme%22%3A%22dark%22%2C%22isTransparent%22%3Atrue%2C"
                                                         "%22width%22%3A%22100%25%22%2C%22height%22%3A%22100%25%22%2C"
                                                         "%22importanceFilter%22%3A%22-1%2C0%2C1%22%2C%22displayMode%22%3A%22compact%22%2C"
                                                         "%22backgroundColor%22%3A%22rgba(2%2C8%2C23%2C1)%22%7D"),
                                                    style={"width": "100%", "height": "100%", "border": "0", "borderRadius": "12px"},
                                                ),
                                            ),
                                        ],
                                    ),
                                ],
                            ),

                            # Coluna Central
                            html.Div(
                                className="col col--center",
                                children=[
                                    html.Div(
                                        style={"display": "flex", "gap": "12px", "alignItems": "stretch", "justifyContent": "space-between"},
                                        children=[
                                            # CARD: Chart Principal
                                            html.Div(
                                                id="chart-card",
                                                className="card",
                                                style={"flex": "6 1 0%", "minWidth": "0"},
                                                children=[
                                                    html.Div("Chart Principal", className="card__title"),
                                                    html.Div(
                                                        className="card__body",
                                                        style={"flex": "1 1 auto", "minHeight": "0", "display": "flex", "padding": "0"},
                                                        children=[
                                                            html.Div(
                                                                id="chart-container",
                                                                style={
                                                                    "width": "100%",
                                                                    "height": "100%",
                                                                    "minHeight": "0",
                                                                    "border": "0",
                                                                    "borderRadius": "14px",
                                                                    "overflow": "hidden",
                                                                    "display": "flex",
                                                                    "alignItems": "stretch",
                                                                    "justifyContent": "center",
                                                                    "backgroundColor": "#0b0c0e",
                                                                },
                                                                children=[
                                                                    html.Iframe(
                                                                        id="tv-chart",
                                                                        src="about:blank",  # será definido no callback init_tv_chart
                                                                        style={
                                                                            "width": "100%",
                                                                            "height": "100%",
                                                                            "border": "0",
                                                                            "borderRadius": "14px",
                                                                            "backgroundColor": "transparent",
                                                                            "flex": "1 1 auto",
                                                                            "minHeight": "0",
                                                                        },
                                                                    )
                                                                ],
                                                            )
                                                        ],
                                                    ),
                                                ],
                                            ),

                                            # Coluna lateral (3D + filtros)
                                            html.Div(
                                                className="center-sidecol",
                                                style={"flex": "1.6 1 0%", "minWidth": "520px", "display": "flex", "flexDirection": "column"},
                                                children=[
                                                    html.Div(
                                                        className="card",
                                                        id="surface-card",
                                                        children=[
                                                            html.Div("3D Market Surface", className="card__title"),
                                                            html.Div(
                                                                className="card__body",
                                                                children=[
                                                                    dcc.Graph(
                                                                        id="surface-3d",
                                                                        style={"height": "80vh", "marginTop": "16px"},
                                                                    ),
                                                                    html.Div(
                                                                        style={"marginTop": "10px", "display": "flex", "justifyContent": "space-between", "gap": "12px", "flexWrap": "wrap"},
                                                                        children=[
                                                                            dcc.Dropdown(
                                                                                id="asset1",
                                                                                className="cp-select",
                                                                                options=[
                                                                                    {"label": "Bitcoin (BTC-USD)", "value": "BTC-USD"},
                                                                                    {"label": "Ethereum (ETH-USD)", "value": "ETH-USD"},
                                                                                    {"label": "Gold (GC=F)", "value": "GC=F"},
                                                                                    {"label": "NASDAQ (^NDX)", "value": "^NDX"},
                                                                                    {"label": "S&P 500 (^GSPC)", "value": "^GSPC"},
                                                                                ],
                                                                                value="BTC-USD",
                                                                                clearable=False,
                                                                                style={"width": "32%", "minWidth": "220px"},
                                                                            ),
                                                                            dcc.Dropdown(
                                                                                id="asset2",
                                                                                className="cp-select",
                                                                                options=[
                                                                                    {"label": "Bitcoin (BTC-USD)", "value": "BTC-USD"},
                                                                                    {"label": "Ethereum (ETH-USD)", "value": "ETH-USD"},
                                                                                    {"label": "Gold (GC=F)", "value": "GC=F"},
                                                                                    {"label": "NASDAQ (^NDX)", "value": "^NDX"},
                                                                                    {"label": "S&P 500 (^GSPC)", "value": "^GSPC"},
                                                                                ],
                                                                                value="ETH-USD",
                                                                                clearable=False,
                                                                                style={"width": "32%", "minWidth": "220px"},
                                                                            ),
                                                                            dcc.Dropdown(
                                                                                id="asset3",
                                                                                className="cp-select",
                                                                                options=[
                                                                                    {"label": "Bitcoin (BTC-USD)", "value": "BTC-USD"},
                                                                                    {"label": "Ethereum (ETH-USD)", "value": "ETH-USD"},
                                                                                    {"label": "Gold (GC=F)", "value": "GC=F"},
                                                                                    {"label": "NASDAQ (^NDX)", "value": "^NDX"},
                                                                                    {"label": "S&P 500 (^GSPC)", "value": "^GSPC"},
                                                                                ],
                                                                                value="GC=F",
                                                                                clearable=False,
                                                                                style={"width": "32%", "minWidth": "220px"},
                                                                            ),
                                                                        ],
                                                                        
                                                                    ),
                                                                    html.Button(
                                                                        "↻ Atualizar Superfície",
                                                                        id="update-surface",
                                                                        n_clicks=0,
                                                                        style={
                                                                            "marginTop": "10px",
                                                                            "alignSelf": "center",
                                                                            "background": "rgba(255,255,255,0.04)",
                                                                            "color": "#E6EDF3",
                                                                            "border": "1px solid rgba(255,255,255,0.1)",
                                                                            "borderRadius": "8px",
                                                                            "padding": "6px 14px",
                                                                            "fontWeight": "600",
                                                                            "fontSize": "13px",
                                                                            "cursor": "pointer",
                                                                        },
                                                                    )
                                                                ],
                                                            ),
                                                        ],
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),

                                    # Bottom row: News + (World Markets | Alerts | Watchlist)
                                    html.Div(
                                        id="bottom-row",
                                        style={
                                            "display": "flex",
                                            "gap": "12px",
                                            "alignItems": "stretch",
                                            "flex": "1 1 auto",
                                            "minHeight": "0",
                                            "height": "100%",
                                            "boxSizing": "border-box",
                                            "paddingBottom": "15px",
                                        },
                                        children=[
                                            # News
                                            html.Div(
                                                className="card",
                                                id="news-card",
                                                style={
                                                    "flex": "1.2 1 auto",
                                                    "minWidth": "360px",
                                                    "overflow": "hidden",
                                                    "display": "flex",
                                                    "flexDirection": "column",
                                                    "minHeight": "0",
                                                    "marginBottom": "10px",
                                                },
                                                children=[
                                                    html.Div("News List", className="card__title"),
                                                    html.Ul(
                                                        id="news-list",
                                                        className="card__body",
                                                        style={
                                                            "overflowY": "auto",
                                                            "paddingRight": "6px",
                                                            "listStyle": "none",
                                                            "display": "flex",
                                                            "flexDirection": "column",
                                                            "gap": "8px",
                                                            "flex": "1 1 auto",
                                                            "minHeight": "0",
                                                        },
                                                        children=[html.Li("Feed Placeholder", className="news-line")],
                                                    ),
                                                ],
                                            ),
                                    

                                            # Alerts
                                            html.Div(
                                                className="card",
                                                id="alerts-card",
                                                style={
                                                    "flex": "1.0 1 0%",
                                                    "minWidth": "200px",
                                                    "overflow": "hidden",
                                                    "display": "flex",
                                                    "flexDirection": "column",
                                                    "minHeight": "0",
                                                    "marginBottom": "10px",
                                                },
                                                children=[
                                                    html.Div("Alerts", className="card__title"),
                                                    html.Ul(
                                                        id="lst-alerts",
                                                        className="card__body",
                                                        style={"flex": "1 1 auto", "minHeight": "0", "overflowY": "auto"},
                                                        children=[html.Li("Sem alertas ativos.", className="card-placeholder")],
                                                    ),
                                                ],
                                            ),

                                            # Watchlist
                                            html.Div(
                                                className="card",
                                                id="watchlist-card",
                                                style={
                                                    "flex": "1.2 1 0%",
                                                    "minWidth": "240px",
                                                    "overflow": "hidden",
                                                    "display": "flex",
                                                    "flexDirection": "column",
                                                    "minHeight": "0",
                                                    "marginBottom": "10px",
                                                },
                                                children=[
                                                    html.Div("Watchlist", className="card__title"),
                                                    html.Div(
                                                        className="card__body",
                                                        style={"paddingBottom": "0", "overflowY": "auto", "flex": "1 1 auto", "minHeight": "0"},
                                                        children=html.Table(
                                                            id="tbl-watchlist",
                                                            children=html.Thead(
                                                                html.Tr([html.Th("Ticker"), html.Th("Último"), html.Th("Δ"), html.Th("%")])
                                                            ),
                                                        ),
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),

                            # Coluna Direita
                            html.Div(
                                className="col col--right", id="col-right",
                                style={"flex":"0 0 320px","maxWidth":"320px","marginRight":"0","paddingRight":"0"},
                                children=[
                                    layout_fear_greed(),                     # permanece no topo (100px)
                                    html.Div(                                # World Markets ocupa o resto
                                        className="card",
                                        style={"flex":"1 1 auto", "minHeight":"0", "display":"flex", "flexDirection":"column", "margin-bottom": "25px"},
                                        children=(
                                            layout_world_markets().children
                                            if hasattr(layout_world_markets(), "children")
                                            else layout_world_markets()
                                        ),
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )



# Root
app.layout = html.Div([dcc.Location(id="url", refresh=False), html.Div(id="page-content")], className="page")

# Validation layout (inclui todos os IDs usados em callbacks)
app.validation_layout = html.Div([
    app.layout,
    get_home_layout(),
    dcc.Store(id="sidebar-collapsed"),
    dcc.Store(id="selected-assets"),
    dcc.Store(id="tv-symbol"),
    dcc.Store(id="kpi-cache"),
    dcc.Store(id="ta-config"),
    dcc.Store(id="filters-news"),
    dcc.Store(id="filters-eco"),
    dcc.Store(id="alerts"),
    dcc.Store(id="wm-page"),
    dcc.Interval(id="tick-fast"),
    dcc.Interval(id="tick-slow"),
    dcc.Interval(id="news-tick"),
    dcc.Interval(id="wm-rot"),
])


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):

    # HOME primeiro — ISSO É O CRÍTICO
    if pathname in ("/", "/app", "/app/"):
        return get_home_layout()

    # BETA / ALPHA
    if pathname in ("/app/beta", "/app/beta/"):
        return html.Div([
            html.Iframe(
                src="/app/beta/",
                style={
                    "width": "100vw",
                    "height": "100vh",
                    "border": "none",
                    "margin": "0",
                    "padding": "0",
                    "display": "block",
                }
            )
        ], style={"margin": "0", "padding": "0", "width": "100vw", "height": "100vh"})

    # EDA
    if pathname in ("/app/eda", "/app/eda/"):
        return html.Div([
            html.Iframe(
                src="/app/eda/",
                style={
                    "width": "100vw",
                    "height": "100vh",
                    "border": "none",
                    "margin": "0",
                    "padding": "0",
                    "display": "block",
                }
            )
        ], style={"margin": "0", "padding": "0", "width": "100vw", "height": "100vh"})

    # PREVISÃO
    if pathname in ("/app/previsao", "/app/previsao/"):
        return html.Div([
            html.Iframe(
                src="/app/previsao/",
                style={
                    "width": "100vw",
                    "height": "100vh",
                    "border": "none",
                    "margin": "0",
                    "padding": "0",
                    "display": "block",
                }
            )
        ], style={"margin": "0", "padding": "0", "width": "100vw", "height": "100vh"})

    # MACRO
    if pathname in ("/app/macro", "/app/macro/"):
        return html.Div([
            html.Iframe(
                src="/app/macro/",
                style={
                    "width": "100vw",
                    "height": "100vh",
                    "border": "none",
                    "margin": "0",
                    "padding": "0",
                    "display": "block",
                }
            )
        ], style={"margin": "0", "padding": "0", "width": "100vw", "height": "100vh"})

    # fallback
    return get_home_layout()



@app.callback(
    [Output("sidebar", "className"), Output("main", "className"), Output("sidebar-collapsed", "data")],
    [Input("sidebar-toggle", "n_clicks"), Input("brand-toggle", "n_clicks")],
    State("sidebar-collapsed", "data"),
)
def toggle_sidebar(n_btn, n_brand, collapsed):
    ctx = dash.callback_context
    if not ctx.triggered: raise dash.exceptions.PreventUpdate
    new_state = not bool(collapsed)
    return ("sidebar sidebar--collapsed" if new_state else "sidebar",
            "main main--collapsed" if new_state else "main",
            new_state)

# ===================== PARTE 9 =====================

@app.callback(Output("tbl-watchlist", "children"),
              Input("tick-fast", "n_intervals"),
              State("selected-assets", "data"))
def cb_watchlist(_, assets):
    try:
        assets = assets or CENTER_DEFAULT
        quotes = get_snapshot_quotes(assets)
        rows = []
        for sym in assets:
            px, chg, pct = quotes.get(sym, (None, None, None))
            rows.append(html.Tr(children=[
                html.Td(sym), html.Td(fmt_num(px)), html.Td(fmt_num(chg)),
                html.Td(f"{fmt_num(pct)} %" if pct is not None else "—", className=f"badge {badge_color(pct)}"),
            ]))
        return html.Table(className="table",
                          children=[html.Thead(html.Tr([html.Th("Ticker"), html.Th("Último"), html.Th("Δ"), html.Th("%")])),
                                    html.Tbody(rows)])
    except Exception as e:
        return html.Div(f"Erro ao atualizar watchlist: {e}", className="card-placeholder")

@app.callback(Output("kpi-strip", "children"), Input("tick-fast", "n_intervals"))
def cb_kpi_strip(_):
    data = get_snapshot_quotes([s for s, _ in KPI_SYMBOLS])
    cards = []
    for sym, label in KPI_SYMBOLS:
        px, chg, pct = data.get(sym, (None, None, None))
        cor = "pos" if (pct or 0) > 0 else "neg" if (pct or 0) < 0 else "neu"
        cards.append(html.Div(className="kpi-card", children=[
            html.Div(className="kpi-top",
                     children=[html.Span(label), html.Span(sym)]),
            html.Div(className="kpi-val", children=fmt_num(px)),
            html.Span(f"{fmt_num(pct)} %" if pct is not None else "— %",
                      className=f"kpi-badge {cor}", style={"alignSelf":"flex-start"}),
        ]))
    return cards

# Rolagem suave dos KPIs pelas setas (sem exibir scrollbar)
app.clientside_callback(
    """
    function(nPrev, nNext, stamp){
        const el = document.getElementById('kpi-strip');
        if(!el){ return window.dash_clientside.no_update; }

        // tamanho de passo ≈ largura de 1 card + gap
        const card = el.querySelector('.kpi-card');
        const step = card ? (card.offsetWidth + 22) : 260;

        // quem disparou?
        var trig = null;
        try{
          const ctx = window.dash_clientside && window.dash_clientside.callback_context;
          if(ctx && ctx.triggered && ctx.triggered.length){
            trig = ctx.triggered[0].prop_id.split('.')[0];
          }
        }catch(e){}

        if(trig === 'kpi-prev'){
            el.scrollBy({ left: -step, behavior: 'smooth' });
        }else if(trig === 'kpi-next'){
            el.scrollBy({ left:  step, behavior: 'smooth' });
        }
        return Date.now();
    }
    """,
    Output("kpi-scroll", "data"),
    [Input("kpi-prev", "n_clicks"), Input("kpi-next", "n_clicks")],
    State("kpi-scroll", "data"),
)


@app.callback(Output("fear-greed", "children"), Input("tick-slow", "n_intervals"))
def cb_fng(_):
    try:
        val, cls, dt = fetch_crypto_fng()
        if val is None:
            return html.Div("Sem dados do FNG", className="card-placeholder",
                            style={"height":"86px","display":"flex","alignItems":"center","justifyContent":"center"})

        import numpy as np
        import plotly.graph_objects as go

        # ---- semicirculo com gradiente NEG -> MID -> POS
        COLOR_NEG = "#EF4444"; COLOR_MID = "#FBBF24"; COLOR_POS = "#59F0C8"; TXT = "#E6EDF3"
        N = 24
        stops = np.linspace(0, 100, N + 1)

        def _hex2rgb(h):
            h = h.lstrip("#")
            return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
        def lerp(c1, c2, t):
            r1, g1, b1 = _hex2rgb(c1); r2, g2, b2 = _hex2rgb(c2)
            r = r1 + (r2 - r1) * t; g = g1 + (g2 - g1) * t; b = b1 + (b2 - b1) * t
            return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

        colors = []
        for s0, s1 in zip(stops[:-1], stops[1:]):
            mid = 0.5 * (s0 + s1)
            colors.append(lerp(COLOR_NEG, COLOR_MID, mid/50.0) if mid <= 50
                          else lerp(COLOR_MID, COLOR_POS, (mid-50)/50.0))

        # canvas alto para RAIO grande; manter em sincronia com --FNG_CANVAS
        FIG_H = 360

        arc = go.Pie(
            values=[1]*N + [N],
            hole=0.75,                  # espessura normal (não engrossa)
            rotation=270,
            direction="clockwise",
            sort=False,
            textinfo="none",
            hoverinfo="skip",
            marker=dict(colors=colors + ["rgba(0,0,0,0)"],
                        line=dict(color="rgba(0,0,0,0)", width=0)),
            domain={"x":[0.0, 1.0], "y":[0.0, 1.0]},
            showlegend=False,
        )
        fig = go.Figure([arc])

        # ponteiro — centro e raio compatíveis com o canvas alto
        ang = np.deg2rad(180 * (1 - (val / 100.0)))
        cx, cy, r = 0.5, 0.60, 0.25   # r maior = ponteiro acompanha o arco grande
        x = cx + r*np.cos(ang); y = cy + r*np.sin(ang)
        fig.add_shape(type="line", x0=cx, y0=cy, x1=x, y1=y, line=dict(color=TXT, width=3), layer="above")
        fig.add_shape(type="circle", x0=cx-0.02, y0=cy-0.02, x1=cx+0.02, y1=cy+0.02,
                      fillcolor=TXT, line=dict(color=TXT, width=2), layer="above")

        fig.update_layout(
            height=FIG_H,
            margin=dict(t=0, b=0, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )

        return html.Div(className="fg-wrap", children=[
            dcc.Graph(figure=fig, config={"displayModeBar": False}, className="fg-canvas")
        ])

    except Exception as e:
        return html.Div(f"Erro ao carregar Fear & Greed: {e}",
                        className="card-placeholder",
                        style={"height":"86px","display":"flex","alignItems":"center","justifyContent":"center"})

from datetime import datetime, timezone, timedelta
import pytz
import numpy as np
import plotly.graph_objects as go

def _hhmm_to_utc_range(tzname, hhmm_open, hhmm_close, span_date=None):
    """
    Converte janelas [HH:MM, HH:MM] na TZ da bolsa para faixas em UTC (em horas decimais).
    Aceita virada de dia (ex.: 23:00 → 05:00).
    Retorna lista de (start_hour_utc, end_hour_utc).
    """
    if span_date is None:
        span_date = datetime.utcnow().date()
    tz = pytz.timezone(tzname)

    def to_dt(hhmm, day):
        h, m = map(int, hhmm.split(":"))
        return tz.localize(datetime(day.year, day.month, day.day, h, m, 0)).astimezone(timezone.utc)

    d0 = span_date
    d1 = span_date + timedelta(days=1)

    o_dt = to_dt(hhmm_open, d0)
    c_dt = to_dt(hhmm_close, d0)

    if c_dt <= o_dt:  # vira o dia
        c_dt = to_dt(hhmm_close, d1)

    st = o_dt.hour + o_dt.minute/60.0
    en = c_dt.hour + c_dt.minute/60.0
    if en <= 24:
        return [(st, en)]
    # pega parte até 24h e resto no dia seguinte (0–…)
    return [(st, 24.0), (0.0, en - 24.0)]

def _ring_for_session(name, tzname, windows, r0, r1, now_utc):
    """
    Gera um 'barpolar' para uma bolsa. 'windows' é lista de (HH:MM, HH:MM) locais.
    r0/r1 definem o anel (raio interno/externo). Colore verde se aberto, cinza se fechado.
    """
    # Discretiza em “meia hora”
    centers = np.arange(0, 24, 0.5)  # horas decimais
    width = 0.5 * 360/24             # 30min em graus

    # mapa de aberto/fechado por slot
    open_mask = np.zeros_like(centers, dtype=bool)
    for hhmm_o, hhmm_c in windows:
        for st, en in _hhmm_to_utc_range(tzname, hhmm_o, hhmm_c, now_utc.date()):
            mask = (centers >= st) & (centers < en) if st < en else ((centers >= st) | (centers < en))
            open_mask |= mask

    # cores
    col_open = "rgba(40, 205, 140, 0.85)"
    col_closed = "rgba(200, 205, 215, 0.28)"

    colors = [col_open if ok else col_closed for ok in open_mask]
    r = np.where(open_mask, r1, r0)  # barras vão do r0 ao r1

    trace = go.Barpolar(
        r = r - r0,
        base = r0,
        theta = centers * (360/24.0),   # 0..360
        width = [width]*len(centers),
        marker_color = colors,
        marker_line_color = "rgba(0,0,0,0)",
        opacity = 1.0,
        name = name,
        hoverinfo = "skip",
        showlegend=False,
    )
    return trace

def build_market_clock_figure(scale: float = 1.0, height: int = 120):
    """
    Desenha o relógio com:
     - Ticks de hora
     - Dois relógios digitais no topo (UTC-3 e UTC)
     - Anéis para: ASX, JPX, HKEX (outer) e B3, LSE, NYSE, NASDAQ, CME (inner)
     - Ponteiro para UTC (branco) e local (verde suave)
    """
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    tz_local = pytz.timezone("America/Sao_Paulo")
    now_loc = now_utc.astimezone(tz_local)
    SCALE = float(scale) if scale else 1.0

    # Definição dos anéis (r0→r1) — 1.0 é o raio total
    rings = [
        ("HKEX","Asia/Hong_Kong",[("09:30","12:00"),("13:00","16:00")], 0.78*SCALE, 0.92*SCALE),
        ("ASX","Australia/Sydney",[("10:00","16:00")],                    0.84*SCALE, 0.98*SCALE),
        ("JPX","Asia/Tokyo",[("09:00","15:00")],                          0.72*SCALE, 0.86*SCALE),
    
        ("CME Chicago","America/Chicago",[("08:30","15:00")],             0.38*SCALE, 0.54*SCALE),
        ("NYSE New York","America/New_York",[("09:30","16:00")],          0.32*SCALE, 0.48*SCALE),
        ("NASDAQ New York","America/New_York",[("09:30","16:00")],        0.26*SCALE, 0.42*SCALE),
        ("LSE London","Europe/London",[("08:00","16:30")],                0.20*SCALE, 0.36*SCALE),
        ("B3 São Paulo","America/Sao_Paulo",[("10:00","17:30")],          0.14*SCALE, 0.30*SCALE),
    ]


    fig = go.Figure()

    # Anéis
    for name, tzname, windows, r0, r1 in rings:
        fig.add_trace(_ring_for_session(name, tzname, windows, r0, r1, now_utc))

    # Ticks de hora (24h)
    hours = np.arange(0,24)
    fig.add_trace(go.Scatterpolar(
        r = [1.02]*len(hours),
        theta = hours * (360/24.0),
        mode="text",
        text=[f"{h:02d}" for h in hours],
        textfont=dict(size=9, color="rgba(230,237,243,0.7)"),
        hoverinfo="skip",
        showlegend=False,
    ))
    # círculos de guia
    for rr in [0.30, 0.55, 0.70, 0.90, 1.00]:
        fig.add_shape(type="circle", xref="x", yref="y",
                      x0=-rr, y0=-rr, x1=rr, y1=rr,
                      line=dict(color="rgba(255,255,255,0.08)", width=1))

    # Ponteiros (UTC em branco; local em verde)
    def _angle_from_time(dt):
        return (dt.hour + dt.minute/60 + dt.second/3600) * (360/24.0)  # graus

    ang_utc = _angle_from_time(now_utc)
    ang_loc = _angle_from_time(now_loc)

    def _add_hand(angle_deg, length, color, width):
        th = np.deg2rad(angle_deg)
        fig.add_shape(type="line",
            x0=0, y0=0, x1=length*np.cos(th), y1=length*np.sin(th),
            line=dict(color=color, width=width))

    _add_hand(ang_utc, 0.95*SCALE, "rgba(230,237,243,0.95)", 3.5)
    _add_hand(ang_loc, 0.70*SCALE, "rgba(40,205,140,0.95)", 3.0)

    # Miolo
    fig.add_shape(type="circle", x0=-0.03, y0=-0.03, x1=0.03, y1=0.03,
                  fillcolor="rgba(230,237,243,1)", line=dict(color="rgba(0,0,0,0)", width=0))

    # “digitais” no topo (texto)
    utc_str  = now_loc.strftime("%H:%M:%S") + " (UTC-3)"
    z_str    = now_utc.strftime("%H:%M:%S") + " (UTC)"
    fig.add_annotation(x=-0.80, y=1.12, text=utc_str, showarrow=False,
                       font=dict(size=12, color="#E6EDF3"))
    fig.add_annotation(x=0.55,  y=1.12, text=z_str, showarrow=False,
                       font=dict(size=12, color="#E6EDF3"))

    # Layout polar
    fig.update_polars(
        radialaxis=dict(visible=False, range=[0,1.0]),
        angularaxis=dict(visible=False, direction="clockwise", rotation=90),
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=6, r=6, t=10, b=6),
        autosize=True,
        height=int(height),                # <<< usa o parâmetro
        polar=dict(
            bgcolor="#0b0c0e",
            radialaxis=dict(visible=False, showgrid=False, showticklabels=False, ticks=""),
            angularaxis=dict(visible=False, showgrid=False, showticklabels=False, ticks="", rotation=90, direction="clockwise"),
        ),
    )
    return fig

WM_ROWS = 6

@app.callback(Output("world-markets", "children"),
              Input("tick-slow", "n_intervals"))
def cb_world_markets_full(_):
    try:
        # achata lista (classe, nome, símbolo)
        triples = []
        for klass, mapping in GLOBAL_MARKETS_BY_CLASS.items():
            for name, sym in mapping.items():
                triples.append((klass, name, sym))

        # cotações em lote
        symbols = [sym for _, _, sym in triples]
        quotes = get_snapshot_quotes(symbols)

        # monta tabela com cabeçalhos por CLASSE
        rows, last_class = [], None
        for klass, name, sym in triples:
            if klass != last_class:
                rows.append(html.Tr([html.Td(klass, colSpan=3, className="wm-region")]))
                last_class = klass

            px, chg, pct = quotes.get(sym, (None, None, None))
            rows.append(
                html.Tr(className="wm-row", children=[
                    html.Td(name, className="wm-name"),
                    html.Td(fmt_num(px), className="wm-last"),
                    html.Td(
                        html.Span(
                            f"{fmt_num(pct)} %" if pct is not None else "—",
                            className=(
                                "kpi-badge "
                                + ("pos" if (pct or 0) > 0 else "neg" if (pct or 0) < 0 else "neu")
                            ),
                        ),
                        className="wm-pct",
                        style={"textAlign": "right"},
                    ),
                ])
            )

        table = html.Table(
            className="wm-table wm-table--grouped",
            children=[
                html.Thead(html.Tr([html.Th("Market"), html.Th("Last"), html.Th("%")])),
                html.Tbody(rows),
            ],
        )
        return html.Div(className="wm-wrap", children=table)

    except Exception as e:
        return html.Div(f"Erro ao atualizar mercados: {e}", className="card-placeholder")


# ===================== PARTE 10 =====================

@app.callback(Output("market-map-body", "children"), Input("tick-slow", "n_intervals"))
def cb_market_map(_):
    try:
        data = get_forex_snapshot(list(FOREX_PAIRS.values()))
        df = pd.DataFrame({"Pair": list(FOREX_PAIRS.keys()), "Change": [data.get(sym) for sym in FOREX_PAIRS.values()]}).dropna()
        if df.empty: raise RuntimeError("Sem dados válidos retornados.")

        treemap = go.Treemap(
            labels=df["Pair"], parents=[""] * len(df),
            values=[abs(v) for v in df["Change"]],
            text=[f"{v:+.2f}%" for v in df["Change"]], textinfo="label+text",
            textfont=dict(size=13, color="white"),
            marker=dict(colors=df["Change"],
                        colorscale=[[0.0,"#8B0000"],[0.3,"#F87171"],[0.5,"#1E293B"],[0.7,"#22C55E"],[1.0,"#14532D"]],
                        cmin=-2, cmax=2, line=dict(width=1.5, color="rgba(0,0,0,0.3)")),
            hovertemplate="<b>%{label}</b><br>Δ: %{text}<extra></extra>",
            tiling=dict(pad=0), root_color="rgba(0,0,0,0)", branchvalues="total", pathbar=dict(visible=False),
        )
        fig = go.Figure(treemap)
        fig.update_layout(height=250, margin=dict(l=0,r=0,t=25,b=0),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", template="carbon_pro",
                          uniformtext=dict(minsize=10, mode='hide'))
        return dcc.Graph(figure=fig, config={"displayModeBar": False},
                         style={"height":"250px","width":"100%","border":"none","borderRadius":"12px","margin":"0","padding":"0","background":"transparent"})
    except Exception as e:
        return html.Div(f"Erro ao gerar mapa de moedas: {str(e)}", className="card-placeholder", style={"height":"250px"})

@app.callback(
    Output("surface-3d", "figure"),
    Input("update-surface", "n_clicks"),
    [State("asset1","value"), State("asset2","value"), State("asset3","value")]
)
def cb_surface3d(n_clicks, a1, a2, a3):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not a1 or not a2 or not a3 or len({a1, a2, a3}) < 3:
        raise dash.exceptions.PreventUpdate

    label_map = {"BTC-USD":"Bitcoin","ETH-USD":"Ethereum","GC=F":"Gold","^NDX":"NASDAQ","^GSPC":"S&P 500"}
    l1, l2, l3 = label_map.get(a1, a1), label_map.get(a2, a2), label_map.get(a3, a3)
    try:
        return build_surface_3d(a1, a2, a3, l1, l2, l3)
    except Exception as e:
        # devolve placeholder com erro visível (sem derrubar o app)
        f = go.Figure()
        f.update_layout(height=460, paper_bgcolor="rgba(0,0,0,0)")
        f.add_annotation(text=f"Superfície indisponível: {e}", showarrow=False, font=dict(color="#E6EDF3"))
        return f



# ---- figura GRANDE do popover (escala maior) ----
@app.callback(Output("market-clock-big", "figure"),
              Input("tick", "n_intervals"))
def cb_market_clock_big(_):
    try:
        # use sua mesma função, mas com escala/altura maiores
        fig = build_market_clock_figure(scale=1.1, height=490)  # <<< ajuste rápido do tamanho
        return fig
    except Exception as e:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)")
        fig.add_annotation(text=f"Erro no relógio: {e}", showarrow=False, font=dict(color="#E6EDF3"))
        return fig


@app.callback(Output("news-list", "children"), Input("news-tick", "n_intervals"))
def cb_news(_):
    try:
        items = fetch_news(limit=3)  # mostra só 3 notícias
        if not items:
            return [html.Li("Sem notícias no momento.", className="card-placeholder")]
        return [
            html.Li(
                className="news-item--text",
                children=html.A(
                    e["title"],
                    href=e["link"], target="_blank", rel="noopener noreferrer"
                ),
            )
            for e in items
        ]
    except Exception as e:
        return [html.Li(f"Erro ao carregar notícias: {e}", className="card-placeholder")]

    
import json
from urllib.parse import quote

@app.callback(Output("tv-chart", "src"), [Input("tv-chart", "id")])
def init_tv_chart(_):
    symbol = "BMFBOVESPA:IBOV"
    cfg = {
        "symbol": symbol,
        "interval": "1D",
        "theme": "dark",
        "locale": "br",
        "allow_symbol_change": True,
        "hide_top_toolbar": False,
        "hide_side_toolbar": False,
        "isTransparent": True,
        "backgroundColor": "rgba(11,12,14,1)",
        "withdateranges": False,
        "save_image": False,
    }
    payload = quote(json.dumps(cfg, separators=(",", ":")))
    return f"https://s.tradingview.com/embed-widget/advanced-chart/?locale=br#{payload}"



import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone

ACCENT = "#59F0C8"
MUTED  = "rgba(255,255,255,0.35)"
BASEBG = "rgba(0,0,0,1)"

# ---------------- Execução ----------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    port = int(os.environ.get("PORT", 8052))
    logging.info(f"🚀 Iniciando Carbon Pro PRO na porta {port}...")
    app.run(host="0.0.0.0", port=port, debug=False, dev_tools_ui=False, dev_tools_props_check=False, use_reloader=False)
