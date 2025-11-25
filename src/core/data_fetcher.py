"""Sistema unificado de download de dados do Yahoo Finance."""

import time
import datetime as dt
from typing import Optional, Dict, Tuple, List
import requests

# Imports condicionais (pode falhar se pandas/yfinance não estiverem instalados)
try:
    import pandas as pd
    import yfinance as yf
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False
    pd = None
    yf = None

from .config import Config
from .cache import get_cache


class YahooDataFetcher:
    """Classe para download de dados do Yahoo Finance com cache e fallback."""
    
    def __init__(self, use_cache: bool = True, use_persistent_cache: bool = False):
        """
        Inicializa o fetcher.
        
        Args:
            use_cache: Se True, usa cache
            use_persistent_cache: Se True, usa cache persistente (SQLite)
        """
        self.use_cache = use_cache
        self.cache = get_cache(use_persistent=use_persistent_cache) if use_cache else None
        self.headers = Config.get_yahoo_headers()
        self.url_template = Config.YCHART_URL
        self.v7_url = Config.YCHART_V7_URL
    
    def _cache_key(self, symbol: str, start: Optional[dt.date], end: Optional[dt.date], 
                   interval: str, period: Optional[str] = None) -> str:
        """Gera chave de cache."""
        if period:
            return f"period|{symbol}|{period}|{interval}"
        if start and end:
            return f"range|{symbol}|{start.isoformat()}|{end.isoformat()}|{interval}"
        return f"symbol|{symbol}|{interval}"
    
    def _yahoo_chart_range(self, symbol: str, start: dt.date, end: dt.date, 
                           interval: str = "1d", max_tries: int = None) -> pd.DataFrame:
        """
        Baixa dados OHLC via Yahoo Chart API por intervalo de datas.
        
        Args:
            symbol: Símbolo do ativo
            start: Data inicial
            end: Data final
            interval: Intervalo (1d, 1h, etc.)
            max_tries: Número máximo de tentativas
            
        Returns:
            DataFrame com OHLC ou DataFrame vazio em caso de erro
        """
        max_tries = max_tries or Config.DEFAULT_MAX_TRIES
        
        # Verifica cache
        if self.use_cache and self.cache:
            cache_key = self._cache_key(symbol, start, end, interval)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached.copy() if isinstance(cached, pd.DataFrame) else cached
        
        try:
            p1 = int(time.mktime(dt.datetime(start.year, start.month, start.day, 0, 0).timetuple()))
            p2 = int(time.mktime(dt.datetime(end.year, end.month, end.day, 23, 59).timetuple()))
        except Exception:
            # Fallback simples
            p1 = int(time.time()) - 86400 * 365 * 2
            p2 = int(time.time())
        
        params = {
            "period1": p1,
            "period2": p2,
            "interval": interval,
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }
        
        delay = Config.DEFAULT_DELAY
        for attempt in range(1, max_tries + 1):
            try:
                r = requests.get(
                    self.url_template.format(symbol=symbol),
                    params=params,
                    headers=self.headers,
                    timeout=Config.DEFAULT_TIMEOUT,
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
                if quote.get("open") is not None:
                    data["Open"] = quote["open"]
                if quote.get("high") is not None:
                    data["High"] = quote["high"]
                if quote.get("low") is not None:
                    data["Low"] = quote["low"]
                if quote.get("close") is not None:
                    data["Close"] = quote["close"]
                if quote.get("volume") is not None:
                    data["Volume"] = quote["volume"]
                
                adjc = (ind.get("adjclose") or [{}])
                adjc = adjc[0].get("adjclose") if adjc else None
                if adjc is not None:
                    data["Adj Close"] = adjc
                
                if not data:
                    raise ValueError("no_price_fields")
                
                df = pd.DataFrame(data, index=idx).astype(float)
                df = df.dropna(how="all")
                
                # Armazena no cache
                if self.use_cache and self.cache and not df.empty:
                    cache_key = self._cache_key(symbol, start, end, interval)
                    self.cache.set(cache_key, df.copy())
                
                return df
                
            except Exception:
                if attempt >= max_tries:
                    return pd.DataFrame()
                time.sleep(delay)
                delay *= 1.8  # backoff exponencial
        
        return pd.DataFrame()
    
    def _yahoo_chart_period(self, symbol: str, period: str, interval: str) -> pd.Series:
        """
        Baixa série de preços por período (1y, 6mo, etc.).
        
        Args:
            symbol: Símbolo do ativo
            period: Período (1y, 6mo, 3mo, 1mo, max, etc.)
            interval: Intervalo (1d, 1wk, 1mo)
            
        Returns:
            Series com preços de fechamento
        """
        # Verifica cache
        if self.use_cache and self.cache:
            cache_key = self._cache_key(symbol, None, None, interval, period=period)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached.copy() if isinstance(cached, pd.Series) else cached
        
        try:
            r = requests.get(
                self.url_template.format(symbol=symbol),
                params={
                    "range": period,
                    "interval": interval,
                    "events": "div,splits",
                    "includeAdjustedClose": "true"
                },
                headers=self.headers,
                timeout=Config.DEFAULT_TIMEOUT
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
            
            # Armazena no cache
            if self.use_cache and self.cache and not s.empty:
                cache_key = self._cache_key(symbol, None, None, interval, period=period)
                self.cache.set(cache_key, s.copy())
            
            return s
            
        except Exception:
            return pd.Series(dtype=float)
    
    def download_range(self, symbol: str, start: dt.date, end: dt.date, 
                      interval: str = "1d") -> pd.DataFrame:
        """
        Download de dados OHLC por intervalo de datas com fallback.
        
        Args:
            symbol: Símbolo do ativo
            start: Data inicial
            end: Data final
            interval: Intervalo
            
        Returns:
            DataFrame com OHLC
        """
        # 1) Tentativa com Chart API
        df = self._yahoo_chart_range(symbol, start, end, interval)
        if df is not None and not df.empty:
            return df
        
        # 2) Fallback yfinance com datas explícitas
        try:
            df = yf.download(
                symbol, start=start, end=end,
                auto_adjust=False, progress=False, threads=False
            )
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        
        # 3) Fallback heurístico de período
        try:
            delta = (end - start).days if isinstance(end, dt.date) and isinstance(start, dt.date) else 730
            if delta > 1825:
                period = "5y"
            elif delta > 1095:
                period = "3y"
            elif delta > 730:
                period = "2y"
            elif delta > 365:
                period = "1y"
            elif delta > 182:
                period = "6mo"
            elif delta > 90:
                period = "3mo"
            else:
                period = "1mo"
            
            df = yf.download(
                symbol, period=period,
                auto_adjust=False, progress=False, threads=False
            )
        except Exception:
            df = pd.DataFrame()
        
        return df if df is not None else pd.DataFrame()
    
    def download_period(self, symbol: str, period: str = "1y", 
                       interval: str = "1d") -> pd.Series:
        """
        Download de série de preços por período com fallback.
        
        Args:
            symbol: Símbolo do ativo
            period: Período (1y, 6mo, etc.)
            interval: Intervalo
            
        Returns:
            Series com preços
        """
        # 1) Tentativa Chart API
        s = self._yahoo_chart_period(symbol, period, interval)
        if not s.empty:
            return s
        
        # 2) Fallback yfinance
        try:
            df = yf.download(
                symbol, period=period, interval=interval,
                progress=False, auto_adjust=False, threads=False
            )
            if df.empty:
                return pd.Series(dtype=float)
            
            if isinstance(df.columns, pd.MultiIndex):
                if "Adj Close" in df.columns.get_level_values(0):
                    df = df.xs("Adj Close", axis=1, level=0)
                else:
                    df = df.xs("Close", axis=1, level=0)
                return pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
            else:
                col = "Adj Close" if "Adj Close" in df.columns else "Close"
                return pd.to_numeric(df[col], errors="coerce").dropna()
        except Exception:
            return pd.Series(dtype=float)
    
    def get_quote_data(self, symbols: List[str], retries: int = None) -> Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]]:
        """
        Obtém cotações rápidas via Yahoo v7 API.
        
        Args:
            symbols: Lista de símbolos
            retries: Número de tentativas
            
        Returns:
            Dict com (preço, mudança, mudança_pct) para cada símbolo
        """
        retries = retries or Config.DEFAULT_RETRIES
        out = {s: (None, None, None) for s in symbols}
        
        for k in range(retries):
            try:
                r = requests.get(
                    self.v7_url,
                    params={"symbols": ",".join(symbols)},
                    headers=self.headers,
                    timeout=Config.DEFAULT_TIMEOUT
                )
                if r.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"busy {r.status_code}")
                r.raise_for_status()
                
                js = r.json().get("quoteResponse", {}).get("result", [])
                for row in js:
                    sym = row.get("symbol")
                    px = row.get("regularMarketPrice")
                    chg = row.get("regularMarketChange")
                    pct = row.get("regularMarketChangePercent")
                    if sym is not None:
                        out[sym] = (
                            float(px) if px is not None else None,
                            float(chg) if chg is not None else None,
                            float(pct) if pct is not None else None
                        )
                if any(v != (None, None, None) for v in out.values()):
                    return out
            except Exception:
                if k < retries - 1:
                    time.sleep(0.35 * (2 ** k))
        
        return out


# Instância global (singleton)
_fetcher_instance: Optional[YahooDataFetcher] = None

def get_fetcher(use_cache: bool = True, use_persistent_cache: bool = False) -> YahooDataFetcher:
    """
    Retorna instância global do fetcher (singleton).
    
    Args:
        use_cache: Se True, usa cache
        use_persistent_cache: Se True, usa cache persistente
        
    Returns:
        Instância do YahooDataFetcher
    """
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = YahooDataFetcher(
            use_cache=use_cache,
            use_persistent_cache=use_persistent_cache
        )
    return _fetcher_instance

