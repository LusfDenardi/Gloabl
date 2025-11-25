"""Configurações globais do GlobalStat."""

import os
from typing import Dict, Any

class Config:
    """Configurações centralizadas do sistema."""
    
    # Yahoo Finance API
    YCHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    YCHART_V7_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
    YCHART_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
        "Connection": "close",
        "Referer": "https://finance.yahoo.com/",
    }
    
    # Cache
    CACHE_TTL = float(os.environ.get("CACHE_TTL", "300.0"))  # 5 minutos
    PRICE_CACHE_TTL = int(os.environ.get("PRICE_TTL", "300"))  # 5 minutos
    
    # Retry e timeout
    DEFAULT_RETRIES = 3
    DEFAULT_MAX_TRIES = 4
    DEFAULT_TIMEOUT = 8
    DEFAULT_DELAY = 0.6
    
    # MetaTrader5 (opcional)
    USE_MT5 = False
    
    # Intervalos e períodos
    DEFAULT_INTERVAL_MS = 300000  # 5 minutos
    DEFAULT_N_BARRAS = 365
    
    # Paths
    ASSETS_DIR = "assets"
    CSS_DIR = "assets/css"
    IMAGES_DIR = "assets/images"
    
    @classmethod
    def get_cache_ttl(cls) -> float:
        """Retorna TTL do cache em segundos."""
        return cls.CACHE_TTL
    
    @classmethod
    def get_yahoo_headers(cls) -> Dict[str, str]:
        """Retorna headers padrão para requisições Yahoo."""
        return cls.YCHART_HEADERS.copy()

