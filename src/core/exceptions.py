"""Exceções customizadas para GlobalStat."""


class GlobalStatException(Exception):
    """Exceção base para GlobalStat."""
    pass


class DataFetchError(GlobalStatException):
    """Erro ao buscar dados."""
    pass


class CacheError(GlobalStatException):
    """Erro no sistema de cache."""
    pass


class ValidationError(GlobalStatException):
    """Erro de validação de dados."""
    pass

