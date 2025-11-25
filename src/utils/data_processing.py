"""Utilitários para processamento de dados financeiros."""

import pandas as pd
import numpy as np
from typing import Union, Optional


def ensure_series_close(df: Optional[pd.DataFrame]) -> pd.Series:
    """
    Garante que retorna uma Series com preços de fechamento.
    
    Args:
        df: DataFrame ou None
        
    Returns:
        Series com preços de fechamento
    """
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return pd.Series(dtype="float64", name="Close")
    
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            s = df["Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
        elif "Adj Close" in df.columns.get_level_values(0):
            s = df["Adj Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
        else:
            s = df.iloc[:, 0]
    else:
        if "Close" in df.columns:
            s = df["Close"]
        elif "Adj Close" in df.columns:
            s = df["Adj Close"]
        else:
            s = df.iloc[:, 0]
    
    s = pd.to_numeric(s, errors="coerce").dropna()
    try:
        s.index = pd.to_datetime(df.index)
    except Exception:
        pass
    return s


def normalize_dataframe(df: pd.DataFrame, base_value: Optional[float] = None) -> pd.DataFrame:
    """
    Normaliza DataFrame para base 100.
    
    Args:
        df: DataFrame a normalizar
        base_value: Valor base (usa primeiro valor se None)
        
    Returns:
        DataFrame normalizado
    """
    df = df.copy()
    for col in df.columns:
        if base_value is None:
            first_val = df[col].iloc[0]
        else:
            first_val = base_value
        if first_val and first_val != 0:
            df[col] = 100 + 100 * np.log(df[col] / first_val)
    return df


def remove_duplicates_index(series: pd.Series, keep: str = 'last') -> pd.Series:
    """
    Remove duplicatas do índice mantendo apenas uma ocorrência.
    
    Args:
        series: Series com índice possivelmente duplicado
        keep: Qual manter ('first' ou 'last')
        
    Returns:
        Series sem duplicatas no índice
    """
    return series[~series.index.duplicated(keep=keep)]


def align_dataframes(*dfs: pd.DataFrame, how: str = 'outer') -> list:
    """
    Alinha múltiplos DataFrames pelo índice.
    
    Args:
        *dfs: DataFrames a alinhar
        how: Tipo de join ('outer', 'inner', 'left', 'right')
        
    Returns:
        Lista de DataFrames alinhados
    """
    if not dfs:
        return []
    
    # Encontra índice comum
    indices = [df.index for df in dfs]
    if how == 'outer':
        common_idx = indices[0].union_many(indices[1:]) if len(indices) > 1 else indices[0]
    elif how == 'inner':
        common_idx = indices[0].intersection_many(indices[1:]) if len(indices) > 1 else indices[0]
    else:
        common_idx = indices[0]
    
    # Alinha cada DataFrame
    aligned = []
    for df in dfs:
        aligned_df = df.reindex(common_idx)
        aligned.append(aligned_df)
    
    return aligned


def calculate_returns(series: pd.Series, method: str = 'log') -> pd.Series:
    """
    Calcula retornos de uma série de preços.
    
    Args:
        series: Series de preços
        method: Método ('log' para logarítmico, 'pct' para percentual)
        
    Returns:
        Series de retornos
    """
    if method == 'log':
        return np.log(series / series.shift(1)).dropna()
    else:
        return series.pct_change().dropna()

