# ===============================================================
# GLOBALSTAT — Módulo de Previsão (com engine de download do Beta/Alpha)
# ===============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import time
import requests
from pathlib import Path
from datetime import timedelta, datetime

import numpy as np
import pandas as pd
import yfinance as yf

from dash import Dash, dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===============================================================
# CONFIG
# ===============================================================
H = 5
TEST_RATIO = 0.2
VAL_RATIO = 0.2
TARGET_COL = "Adj Close"

COLORS = {
    "bg": "rgba(0,0,0,0)",
    "panel": "rgba(255,255,255,0.03)",
    "text": "#e6e6e6",
    "accent": "#59f0c8",
    "train": "#6ecb63",
    "test": "#f28482",
    "test_pred": "#ffd166",
    "future_pred": "#cdb4db",
    "grid": "rgba(255,255,255,0.08)"
}

# ===============================================================
# ENGINE DE DOWNLOAD — MESMO DO BETA/ALPHA
# ===============================================================

_CHART_CACHE = {}
_CHART_TTL = 300.0  # 5 minutos


def _cache_get(key):
    item = _CHART_CACHE.get(key)
    if not item:
        return None
    ts, obj = item
    return obj if (time.time() - ts) < _CHART_TTL else None


def _cache_set(key, obj):
    _CHART_CACHE[key] = (time.time(), obj)


def _yahoo_chart_period(symbol: str, period: str, interval: str):
    """Download robusto via Yahoo Chart API (funciona no Railway)."""
    key = ("period", symbol, period, interval)
    hit = _cache_get(key)
    if hit is not None:
        return hit

    try:
        r = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"range": period, "interval": interval,
                    "events": "div,splits", "includeAdjustedClose": "true"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        r.raise_for_status()
        js = r.json().get("chart", {}).get("result", [])

        if not js:
            return pd.Series(dtype=float)

        r0 = js[0]
        ts = r0.get("timestamp", [])
        ind = r0.get("indicators", {})

        quote = (ind.get("quote") or [{}])[0]
        closes = quote.get("close")

        if closes is None:
            adj = (ind.get("adjclose") or [{}])[0].get("adjclose")
            closes = adj

        if not ts or not closes:
            return pd.Series(dtype=float)

        idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert("UTC").tz_localize(None)
        s = pd.Series(closes, index=idx, name="Close").astype(float).dropna()

        _cache_set(key, s)
        return s

    except Exception:
        return pd.Series(dtype=float)


def _yahoo_chart_range(symbol: str, start, end, interval="1d"):
    key = ("range", symbol, str(start), str(end), interval)
    hit = _cache_get(key)
    if hit is not None:
        return hit

    try:
        p1 = int(time.mktime(datetime(start.year, start.month, start.day).timetuple()))
        p2 = int(time.mktime(datetime(end.year, end.month, end.day, 23, 59).timetuple()))

        r = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"period1": p1, "period2": p2, "interval": interval,
                    "events": "div,splits", "includeAdjustedClose": "true"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        r.raise_for_status()

        js = r.json().get("chart", {}).get("result", [])
        if not js:
            return pd.DataFrame()

        r0 = js[0]
        ts = r0.get("timestamp", [])
        ind = r0.get("indicators", {})

        if not ts or not ind:
            return pd.DataFrame()

        idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert("UTC").tz_localize(None)
        quote = (ind.get("quote") or [{}])[0]

        data = {
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": quote.get("close"),
            "Volume": quote.get("volume")
        }

        adj = (ind.get("adjclose") or [{}])[0].get("adjclose")
        if adj is not None:
            data["Adj Close"] = adj

        df = pd.DataFrame(data, index=idx).astype(float).dropna(how="all")

        _cache_set(key, df)
        return df

    except Exception:
        return pd.DataFrame()


def baixar_serie(ticker: str, period: str = "1y", interval: str = "1d"):
    """
    Função principal — usa Chart API → fallback range → fallback yfinance.
    """
    # 1) Chart API (principal)
    s = _yahoo_chart_period(ticker, period, interval)
    if not s.empty:
        return s

    # 2) Fallback: range completo
    end = datetime.utcnow().date()
    start = end - timedelta(days=365)
    df = _yahoo_chart_range(ticker, start, end, interval)
    if isinstance(df, pd.DataFrame) and not df.empty:
        col = "Adj Close" if "Adj Close" in df else "Close"
        return df[col].dropna()

    # 3) Fallback do fallback: yfinance
    df2 = yf.download(ticker, period=period, interval=interval,
                      progress=False, auto_adjust=False, threads=False)

    if df2.empty:
        return pd.Series(dtype=float)

    if isinstance(df2.columns, pd.MultiIndex):
        if "Adj Close" in df2.columns.get_level_values(0):
            df2 = df2.xs("Adj Close", axis=1, level=0)
        else:
            df2 = df2.xs("Close", axis=1, level=0)
        return pd.to_numeric(df2.iloc[:, 0], errors="coerce").dropna()
    else:
        col = "Adj Close" if "Adj Close" in df2.columns else "Close"
        return pd.to_numeric(df2[col], errors="coerce").dropna()


# ===============================================================
# LISTA COMPLETA DE ATIVOS (MESMA DO SEU CÓDIGO)
# ===============================================================

B3_TICKERS = [
    {"label": "Bitcoin (BTC)", "value": "BTC-USD"},
    {"label": "Ethereum (ETH)", "value": "ETH-USD"},
    {"label": "BNB", "value": "BNB-USD"},
    {"label": "Cardano (ADA)", "value": "ADA-USD"},
    {"label": "Solana (SOL)", "value": "SOL-USD"},
    {"label": "XRP", "value": "XRP-USD"},
    {"label": "Dogecoin (DOGE)", "value": "DOGE-USD"},
    {"label": "Gold", "value": "GC=F"},
    {"label": "Silver", "value": "SI=F"},
    {"label": "Platinum", "value": "PL=F"},
    {"label": "Copper", "value": "HG=F"},
    {"label": "WTI Crude Oil", "value": "CL=F"},
    {"label": "Brent Crude", "value": "BZ=F"},
    {"label": "Natural Gas", "value": "NG=F"},
    {"label": "Corn", "value": "ZC=F"},
    {"label": "Soybeans", "value": "ZS=F"},
    {"label": "Wheat", "value": "ZW=F"},
    {"label": "Coffee", "value": "KC=F"},
    {"label": "Cotton", "value": "CT=F"},
    {"label": "Sugar", "value": "SB=F"},
    {"label": "US Dollar Index (DXY)", "value": "DX-Y.NYB"},
    {"label": "EUR/USD", "value": "EURUSD=X"},
    {"label": "GBP/USD", "value": "GBPUSD=X"},
    {"label": "USD/JPY", "value": "USDJPY=X"},
    {"label": "USD/BRL", "value": "BRL=X"},
    {"label": "USD/CHF", "value": "CHFUSD=X"},
    {"label": "USD/CAD", "value": "CAD=X"},
    {"label": "USD/MXN", "value": "MXN=X"},
    {"label": "USD/CNY", "value": "CNY=X"},
    {"label": "USD/ZAR", "value": "ZAR=X"},
    {"label": "S&P 500", "value": "^GSPC"},
    {"label": "NASDAQ 100", "value": "^NDX"},
    {"label": "Dow Jones", "value": "^DJI"},
    {"label": "Russell 2000", "value": "^RUT"},
    {"label": "VIX", "value": "^VIX"},
    {"label": "Ibovespa", "value": "^BVSP"},
    {"label": "Euro Stoxx 50", "value": "^STOXX50E"},
    {"label": "DAX", "value": "^GDAXI"},
    {"label": "CAC 40", "value": "^FCHI"},
    {"label": "FTSE 100", "value": "^FTSE"},
    {"label": "Nikkei 225", "value": "^N225"},
    {"label": "Hang Seng", "value": "^HSI"},
    {"label": "Shanghai Composite", "value": "000001.SS"},
    {"label": "KOSPI", "value": "^KS11"},
    {"label": "Nifty 50", "value": "^NSEI"},
    {"label": "ASX 200", "value": "^AXJO"},
    {"label": "TSX", "value": "^GSPTSE"},
    {"label": "US 2Y Treasury Yield", "value": "^IRX"},
    {"label": "US 10Y Treasury Yield", "value": "^TNX"},
    {"label": "US 30Y Treasury Yield", "value": "^TYX"},
    {"label": "T-Note Futures", "value": "ZN=F"},
    {"label": "T-Bond Futures", "value": "ZB=F"},
    {"label": "SPY", "value": "SPY"},
    {"label": "QQQ", "value": "QQQ"},
    {"label": "IWM", "value": "IWM"},
    {"label": "EEM", "value": "EEM"},
    {"label": "EWZ", "value": "EWZ"},
    {"label": "FXI", "value": "FXI"},
    {"label": "VGK", "value": "VGK"},
    {"label": "GLD", "value": "GLD"},
    {"label": "SLV", "value": "SLV"},
    {"label": "DBC", "value": "DBC"},
    {"label": "USO", "value": "USO"},
    {"label": "TLT", "value": "TLT"},
    {"label": "HYG", "value": "HYG"},

    {"label": "Petrobras PN", "value": "PETR4.SA"},
    {"label": "Petrobras ON", "value": "PETR3.SA"},
    {"label": "Vale", "value": "VALE3.SA"},
    {"label": "Itaú Unibanco", "value": "ITUB4.SA"},
    {"label": "Bradesco PN", "value": "BBDC4.SA"},
    {"label": "Bradesco ON", "value": "BBDC3.SA"},
    {"label": "Ambev", "value": "ABEV3.SA"},
    {"label": "Banco do Brasil", "value": "BBAS3.SA"},
    {"label": "Gerdau PN", "value": "GGBR4.SA"},
    {"label": "Suzano", "value": "SUZB3.SA"},
    {"label": "WEG", "value": "WEGE3.SA"},
    {"label": "Localiza", "value": "RENT3.SA"},
    {"label": "Raízen", "value": "RAIZ4.SA"},
    {"label": "CSN", "value": "CSNA3.SA"},
    {"label": "Usiminas PNA", "value": "USIM5.SA"},
    {"label": "Eletrobras ON", "value": "ELET3.SA"},
    {"label": "Magazine Luiza", "value": "MGLU3.SA"},
    {"label": "Natura", "value": "NTCO3.SA"},
    {"label": "Pão de Açúcar", "value": "PCAR3.SA"},
    {"label": "Embraer", "value": "EMBR3.SA"},
    {"label": "Apple", "value": "AAPL"},
    {"label": "Microsoft", "value": "MSFT"},
    {"label": "Amazon", "value": "AMZN"},
    {"label": "Alphabet A", "value": "GOOGL"},
    {"label": "Alphabet C", "value": "GOOG"},
    {"label": "NVIDIA", "value": "NVDA"},
    {"label": "Tesla", "value": "TSLA"},
    {"label": "Meta", "value": "META"},
    {"label": "Berkshire Hathaway", "value": "BRK-B"},
    {"label": "JPMorgan", "value": "JPM"},
    {"label": "Goldman Sachs", "value": "GS"},
    {"label": "Visa", "value": "V"},
    {"label": "Mastercard", "value": "MA"},
    {"label": "Coca-Cola", "value": "KO"},
    {"label": "Pfizer", "value": "PFE"},
    {"label": "Exxon Mobil", "value": "XOM"},
    {"label": "Chevron", "value": "CVX"},
    {"label": "Walmart", "value": "WMT"},
    {"label": "Costco", "value": "COST"}
]

# ===============================================================
# FUNÇÕES UTILITÁRIAS
# ===============================================================
def get_data(ticker: str) -> pd.DataFrame:
    """
    Usa o mesmo engine robusto do Beta/Alpha (baixar_serie),
    garantindo 1 ano de dados diários.
    """
    s = baixar_serie(ticker, period="1y", interval="1d")

    if s is None or len(s) == 0:
        raise ValueError(f"Nenhum dado retornado para {ticker}.")

    # garante série limpa, index datetime e ordenado
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index().drop_duplicates()

    if len(s) < 60:
        raise ValueError(f"Série de {ticker} muito curta ({len(s)} pontos).")

    return pd.DataFrame({TARGET_COL: s})


def is_business_calendar(index: pd.DatetimeIndex) -> bool:
    # Heurística simples: se não há sábados/domingos na série, consideramos "business days"
    return (~pd.Series(index.dayofweek).isin([5, 6])).all()


def next_index_range(last_date: pd.Timestamp, periods: int, use_business_days: bool) -> pd.DatetimeIndex:
    if use_business_days:
        return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=periods)
    else:
        return pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq="D")


def split_series(y: pd.Series):
    y = y.dropna().astype(float)
    n_test = max(5, int(len(y) * TEST_RATIO))
    return y.iloc[:-n_test], y.iloc[-n_test:]


def make_features(y: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": y})
    for lag in [1, 5, 10, 20]:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["ma_5"] = df["y"].rolling(5).mean()
    df["ma_21"] = df["y"].rolling(21).mean()
    df["ret_1"] = df["y"].pct_change()
    return df.replace([np.inf, -np.inf], np.nan).dropna().astype(float)


def evaluate(y_true, y_pred):
    y_true, y_pred = y_true.align(y_pred, join="inner")
    y_true = y_true.dropna()
    y_pred = y_pred.dropna()
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Sem dados válidos para calcular métricas.")
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }

# ===============================================================
# MODELOS (ARIMA / SARIMA)
# ===============================================================
def fit_predict_arima(y_train, y_test, use_business_days: bool):
    y_train, y_test = y_train.dropna(), y_test.dropna()
    y_train_arr, y_test_arr = np.asarray(y_train), np.asarray(y_test)

    res = ARIMA(y_train_arr, order=(1, 1, 1), dates=None, freq=None).fit()
    pred_vals = res.forecast(steps=len(y_test_arr))
    y_pred_test = pd.Series(pred_vals, index=y_test.index)

    y_all = np.concatenate([y_train_arr, y_test_arr])
    res_full = ARIMA(y_all, order=(1, 1, 1), dates=None, freq=None).fit()
    future_vals = res_full.forecast(steps=H)
    last_date = pd.to_datetime(y_test.index[-1])
    future_idx = next_index_range(last_date, H, use_business_days)
    return y_pred_test, pd.Series(future_vals, index=future_idx)


def fit_predict_sarima(y_train, y_test, use_business_days: bool):
    m = 5 if use_business_days else 7
    y_train, y_test = y_train.dropna(), y_test.dropna()
    y_train_arr, y_test_arr = np.asarray(y_train), np.asarray(y_test)

    res = SARIMAX(
        y_train_arr,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, m),
        dates=None,
        freq=None
    ).fit(disp=False)
    pred_vals = res.forecast(steps=len(y_test_arr))
    y_pred_test = pd.Series(pred_vals, index=y_test.index)

    y_all = np.concatenate([y_train_arr, y_test_arr])
    res_full = SARIMAX(
        y_all,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, m),
        dates=None,
        freq=None
    ).fit(disp=False)
    future_vals = res_full.forecast(steps=H)
    last_date = pd.to_datetime(y_test.index[-1])
    future_idx = next_index_range(last_date, H, use_business_days)
    return y_pred_test, pd.Series(future_vals, index=future_idx)

# ===============================================================
# MODELOS MACHINE LEARNING (RF / XGBOOST)
# ===============================================================
def _temporal_train_val_split(df_feat: pd.DataFrame, cutoff: pd.Timestamp):
    """
    Recebe df_feat com index temporal e coluna 'y'.
    Divide em:
      - train_df: até cutoff (INCLUSIVO)
      - test_df : após cutoff
    Dentro de train_df, separa validação temporal ao final.
    """
    train_df = df_feat[df_feat.index <= cutoff].copy()
    test_df = df_feat[df_feat.index > cutoff].copy()

    if len(train_df) < 40 or len(test_df) < 1:
        raise ValueError("Janelas muito curtas após engenharia de features.")

    n_val = max(10, int(len(train_df) * VAL_RATIO))
    val_df = train_df.iloc[-n_val:].copy()
    tr_df = train_df.iloc[:-n_val].copy()

    if len(tr_df) < 20:
        n_val = min(n_val, max(5, len(train_df) // 4))
        val_df = train_df.iloc[-n_val:].copy()
        tr_df = train_df.iloc[:-n_val].copy()

    X_tr, y_tr = tr_df.drop(columns=["y"]), tr_df["y"]
    X_val, y_val = val_df.drop(columns=["y"]), val_df["y"]
    X_te, y_te = test_df.drop(columns=["y"]), test_df["y"]
    return X_tr, y_tr, X_val, y_val, X_te, y_te


def _ml_recursive_forecast(model, y_hist: pd.Series, use_business_days: bool) -> pd.Series:
    preds, last = [], y_hist.copy()
    for _ in range(H):
        feat = make_features(last)
        x_last = feat.drop(columns=["y"]).iloc[-1:].values
        nxt = float(model.predict(x_last)[0])
        if use_business_days:
            nxt_idx = pd.bdate_range(last.index[-1] + pd.Timedelta(days=1), periods=1)[0]
            while nxt_idx in last.index:
                nxt_idx = pd.bdate_range(nxt_idx + pd.Timedelta(days=1), periods=1)[0]
        else:
            nxt_idx = last.index[-1] + pd.Timedelta(days=1)
            while nxt_idx in last.index:
                nxt_idx += pd.Timedelta(days=1)
        last.loc[nxt_idx] = nxt
        preds.append((nxt_idx, nxt))
    return pd.Series([v for _, v in preds], index=[i for i, _ in preds])


def _xgb_predict_best_booster(bst: xgb.Booster, X: np.ndarray) -> np.ndarray:
    d = xgb.DMatrix(X)
    ntree = getattr(bst, "best_ntree_limit", 0)
    if isinstance(ntree, int) and ntree > 0:
        try:
            return bst.predict(d, ntree_limit=ntree)
        except TypeError:
            pass
    return bst.predict(d)


def fit_predict_rf(y_train: pd.Series, y_test: pd.Series, use_business_days: bool):
    df_feat = make_features(pd.concat([y_train, y_test]))
    cutoff = y_train.index[-1]
    X_tr, y_tr, X_val, y_val, X_te, y_te = _temporal_train_val_split(df_feat, cutoff)

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=1,
        min_samples_split=2,
        bootstrap=True
    )
    model.fit(X_tr, y_tr)
    y_pred_test = pd.Series(model.predict(X_te), index=y_te.index)

    y_all = pd.concat([y_train, y_test])
    y_forecast = _ml_recursive_forecast(model, y_all, use_business_days)
    return y_pred_test, y_forecast


def fit_predict_xgb(y_train: pd.Series, y_test: pd.Series, use_business_days: bool):
    df_feat = make_features(pd.concat([y_train, y_test]))
    cutoff = y_train.index[-1]
    X_tr, y_tr, X_val, y_val, X_te, y_te = _temporal_train_val_split(df_feat, cutoff)

    dtr = xgb.DMatrix(X_tr.values, label=y_tr.values)
    dval = xgb.DMatrix(X_val.values, label=y_val.values)
    dte = xgb.DMatrix(X_te.values)

    params = {
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.0,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "seed": 42
    }

    bst = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=5000,
        evals=[(dtr, "train"), (dval, "validation")],
        early_stopping_rounds=150,
        verbose_eval=False
    )

    y_pred_test = pd.Series(_xgb_predict_best_booster(bst, X_te.values), index=y_te.index)

    class BoosterWrapper:
        def __init__(self, booster):
            self.booster = booster
        def predict(self, X):
            return _xgb_predict_best_booster(self.booster, X)

    y_all = pd.concat([y_train, y_test])
    y_forecast = _ml_recursive_forecast(BoosterWrapper(bst), y_all, use_business_days)
    return y_pred_test, y_forecast

# ===============================================================
# EXECUÇÃO PRINCIPAL DOS MODELOS
# ===============================================================
def run_model(model_name: str, y: pd.Series):
    use_bdays = is_business_calendar(y.index)
    y_train, y_test = split_series(y)

    if model_name == "ARIMA":
        y_pred_test, y_forecast = fit_predict_arima(y_train, y_test, use_bdays)
    elif model_name == "SARIMA":
        y_pred_test, y_forecast = fit_predict_sarima(y_train, y_test, use_bdays)
    elif model_name == "XGBoost":
        y_pred_test, y_forecast = fit_predict_xgb(y_train, y_test, use_bdays)
    elif model_name == "Random Forest":
        y_pred_test, y_forecast = fit_predict_rf(y_train, y_test, use_bdays)
    else:
        raise ValueError("Modelo inválido.")

    m = evaluate(y_test, y_pred_test)
    return y_train, y_test, y_pred_test, y_forecast, m

# ===============================================================
# VISUAL (FIGURA + MÉTRICAS)
# ===============================================================
def make_figure(y_train, y_test, y_pred_test, y_forecast, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_train.index, y=y_train,
        name="Treino (Real)",
        line=dict(color=COLORS["train"])
    ))
    fig.add_trace(go.Scatter(
        x=y_test.index, y=y_test,
        name="Teste (Real)",
        line=dict(color=COLORS["test"])
    ))
    fig.add_trace(go.Scatter(
        x=y_pred_test.index, y=y_pred_test,
        name="Teste (Previsto)",
        line=dict(color=COLORS["test_pred"], dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=y_forecast.index, y=y_forecast,
        name=f"Futuro +{H} (Previsto)",
        line=dict(color=COLORS["future_pred"], width=3)
    ))

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"]),
        legend=dict(bgcolor=COLORS["panel"]),
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"]),
        yaxis=dict(showgrid=True, gridcolor=COLORS["grid"])
    )
    return fig


def metrics_panel(m):
    fmt = lambda v: f"{v:,.4f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return html.Div([
        html.H4("Métricas (teste)", style={"color": COLORS["accent"]}),
        html.Div(f"MSE:  {fmt(m['MSE'])}"),
        html.Div(f"RMSE: {fmt(m['RMSE'])}"),
        html.Div(f"MAE:  {fmt(m['MAE'])}"),
        html.Div(f"R²:   {fmt(m['R2'])}"),
        html.Hr(style={"borderColor": COLORS["grid"]}),
        html.P(f"Horizonte de previsão: {H} passos", style={"opacity": 0.8})
    ], style={
        "backgroundColor": COLORS["panel"],
        "padding": "16px",
        "borderRadius": "12px",
        "border": f"1px solid {COLORS['grid']}",
        "color": COLORS["text"]
    })
# ===============================================================
# MONTAGEM NO PAINEL PRINCIPAL (sub-app em /app/previsao/)
# ===============================================================
def mount_previsao(shared_server):
    """
    Monta o app Dash de Previsão sob /app/previsao/ no Flask `shared_server`.
    Usa o mesmo engine robusto de download (baixar_serie) do módulo Beta/Alpha.
    """
    PREFIX = "/app/previsao/"

    app = Dash(
        name="previsao_app",
        server=shared_server,
        requests_pathname_prefix=PREFIX,
        routes_pathname_prefix=PREFIX,
        suppress_callback_exceptions=True,
        title="GlobalStat — Previsão",
        serve_locally=True,
    )

    # ===================== CSS Retro/Carbon Pro para Dropdown =====================
    retro_css = """
    <style>
    /* Dropdown base */
    .retro-dropdown .Select-control{
        background: transparent !important;
        border: none !important;
        border-bottom: 1px solid rgba(255,255,255,0.18) !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        height: 36px !important;
    }

    .retro-dropdown .Select-control:hover{
        border-bottom-color: #59f0c8 !important;
    }

    .retro-dropdown .Select-placeholder,
    .retro-dropdown .Select-value-label {
        color: #e8edf2 !important;
        font-weight: 600 !important;
    }

    /* Fundo das opções */
    .retro-dropdown .Select-menu-outer{
        background: rgba(15,17,20,0.98) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
    }

    .retro-dropdown .Select-option{
        color: #e8edf2 !important;
        background: transparent !important;
        padding: 8px 12px !important;
        font-size: 13px !important;
    }

    .retro-dropdown .Select-option:hover{
        background: rgba(89,240,200,0.15) !important;
        color: #59f0c8 !important;
    }
    </style>
    """

    # injeta o CSS no index
    app.index_string = app.index_string.replace(
        "</head>", retro_css + "</head>"
    )

    # ------------------------ Layout ------------------------
    app.layout = html.Div(
        className="app",
        style={
            "height": "100vh",
            "width": "100vw",
            "overflow": "hidden",
            "display": "flex",
            "flexDirection": "column",
            "backgroundColor": "var(--bg, #0b0c0e)",
            "padding": "24px",
            "boxSizing": "border-box",
        },
        children=[
            # Header com botão voltar
            html.Div(
                className="beta-header",
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "gap": "14px",
                    "marginBottom": "16px",
                },
                children=[
                    html.A(
                        html.Button("← Voltar", className="btn-tab back-btn"),
                        href="/app/",
                    ),
                    html.H2(
                        "GlobalStat — Previsão (1 ano | horizonte 5 pregões)",
                        style={
                            "textAlign": "center",
                            "color": COLORS["accent"],
                            "fontWeight": "600",
                            "fontSize": "24px",
                            "margin": "0",
                            "flex": "1",
                        },
                    ),
                    html.Div(),  # espaçador
                ],
            ),

            # Controles
            html.Div(
                className="controls-grid",
                style={"marginBottom": "20px"},
                children=[
                    html.Div(
                        className="field",
                        children=[
                            html.Label("Ativo"),
                            dcc.Dropdown(
                                id="ticker-dd",
                                options=B3_TICKERS,
                                value="PETR4.SA",
                                clearable=False,
                                className="retro-dropdown",
                            ),
                        ],
                    ),
                    html.Div(
                        className="field",
                        children=[
                            html.Label("Modelo"),
                            dcc.Dropdown(
                                id="model-dd",
                                options=[
                                    "ARIMA",
                                    "SARIMA",
                                    "XGBoost",
                                    "Random Forest",
                                ],
                                value="ARIMA",
                                clearable=False,
                                className="retro-dropdown",
                            ),
                        ],
                    ),
                    html.Div(
                        className="field",
                        children=[
                            html.Button(
                                "Atualizar previsão",
                                id="run-btn",
                                n_clicks=0,
                                style={
                                    "marginTop": "6px",
                                    "background": COLORS["accent"],
                                    "color": "#000",
                                    "border": "none",
                                    "padding": "8px 16px",
                                    "borderRadius": "10px",
                                    "fontWeight": "600",
                                    "cursor": "pointer",
                                },
                            ),
                        ],
                    ),
                ],
            ),

            html.Div(
                id="error-msg",
                style={
                    "color": "#ff6b6b",
                    "marginTop": "4px",
                    "marginBottom": "8px",
                    "fontSize": "13px",
                },
            ),

            # Main grid: gráfico (esquerda) + métricas (direita)
            html.Div(
                className="main-grid",
                style={
                    "flex": "1",
                    "display": "grid",
                    "gridTemplateColumns": "1fr 300px",
                    "gap": "20px",
                    "height": "100%",
                    "overflow": "hidden",
                },
                children=[
                    html.Div(
                        className="graph-wrapper card",
                        children=[
                            dcc.Graph(
                                id="main-graph",
                                style={
                                    "height": "100%",
                                    "width": "100%",
                                    "flex": "1",
                                },
                            )
                        ],
                    ),
                    html.Div(
                        id="metrics-box",
                        className="aside card",
                        style={
                            "height": "100%",
                            "overflowY": "auto",
                            "padding": "16px",
                        },
                    ),
                ],
            ),
        ],
    )
    # ------------------------ Callbacks ------------------------
    @app.callback(
        [
            Output("main-graph", "figure"),
            Output("metrics-box", "children"),
            Output("error-msg", "children"),
        ],
        [
            Input("run-btn", "n_clicks"),
            Input("ticker-dd", "value"),
            Input("model-dd", "value"),
        ],
        prevent_initial_call=False,  # roda na abertura da aba
    )
    def update_forecast(n_clicks, ticker, model_name):
        if not ticker or not model_name:
            return (
                go.Figure(),
                html.Div("Selecione um ativo e um modelo.", style={"fontSize": "13px"}),
                "",
            )

        try:
            df = get_data(ticker)
            y = df[TARGET_COL]
            y_train, y_test, y_pred_test, y_forecast, m = run_model(model_name, y)

            fig = make_figure(
                y_train,
                y_test,
                y_pred_test,
                y_forecast,
                f"{ticker} — {model_name}",
            )
            box = metrics_panel(m)
            return fig, box, ""
        except Exception as e:
            # Em caso de erro, mostra mensagem e mantém gráfico/box vazio
            empty_fig = go.Figure()
            empty_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=COLORS["bg"],
                plot_bgcolor=COLORS["bg"],
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
            )
            return empty_fig, html.Div(), f"Erro: {str(e)}"

    print("[INFO] Previsão em /app/previsao/ — usando engine robusto de download (baixar_serie)")
    return app


# ===============================================================
# EXECUÇÃO LOCAL (opcional para teste isolado)
# ===============================================================
if __name__ == "__main__":
    from flask import Flask

    dev_server = Flask(__name__)
    mount_previsao(dev_server)
    port = int(os.environ.get("PORT", "8077"))
    dev_server.run(host="0.0.0.0", port=port, debug=True)
