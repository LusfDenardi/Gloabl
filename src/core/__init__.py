"""Módulo core com funcionalidades compartilhadas."""

from .config import Config

# Importações condicionais para evitar erros durante desenvolvimento
try:
    from .cache import CacheManager, get_cache
except (ImportError, Exception) as e:
    CacheManager = None
    get_cache = None
    _cache_error = str(e)

try:
    from .data_fetcher import YahooDataFetcher, get_fetcher
except (ImportError, Exception) as e:
    YahooDataFetcher = None
    get_fetcher = None
    _fetcher_error = str(e)

try:
    from .auth import AuthManager, get_auth_manager
except (ImportError, Exception) as e:
    AuthManager = None
    get_auth_manager = None
    _auth_error = str(e)

__all__ = ["Config", "CacheManager", "YahooDataFetcher", "get_cache", "get_fetcher",
           "AuthManager", "get_auth_manager"]