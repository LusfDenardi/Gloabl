"""Sistema de cache centralizado para GlobalStat."""

import time
import os
from typing import Any, Optional, Dict, Tuple
from pathlib import Path
import sqlite3
import pickle
import json

from .config import Config


class CacheManager:
    """Gerenciador de cache unificado com suporte a memória e persistência."""
    
    def __init__(self, use_persistent: bool = False, db_path: Optional[str] = None):
        """
        Inicializa o gerenciador de cache.
        
        Args:
            use_persistent: Se True, usa cache persistente (SQLite)
            db_path: Caminho para o banco SQLite (padrão: cache.db no diretório atual)
        """
        self._memory_cache: Dict[str, Tuple[float, Any]] = {}
        self._ttl = Config.CACHE_TTL
        self._use_persistent = use_persistent
        self._db_path = db_path or "cache.db"
        self._conn: Optional[sqlite3.Connection] = None
        
        if use_persistent:
            self._init_db()
    
    def _init_db(self):
        """Inicializa banco de dados SQLite para cache persistente."""
        try:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    timestamp REAL,
                    ttl REAL
                )
            """)
            self._conn.commit()
        except Exception as e:
            print(f"[Cache] Erro ao inicializar DB: {e}")
            self._use_persistent = False
    
    def _clean_expired_memory(self):
        """Remove entradas expiradas do cache em memória."""
        now = time.time()
        expired = [k for k, (ts, _) in self._memory_cache.items() if now - ts > self._ttl]
        for k in expired:
            self._memory_cache.pop(k, None)
    
    def _clean_expired_db(self):
        """Remove entradas expiradas do banco de dados."""
        if not self._conn:
            return
        try:
            now = time.time()
            self._conn.execute("DELETE FROM cache WHERE (timestamp + ttl) < ?", (now,))
            self._conn.commit()
        except Exception as e:
            print(f"[Cache] Erro ao limpar DB: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Recupera valor do cache.
        
        Args:
            key: Chave do cache
            
        Returns:
            Valor armazenado ou None se não encontrado/expirado
        """
        # Tenta memória primeiro
        item = self._memory_cache.get(key)
        if item:
            ts, value = item
            if time.time() - ts < self._ttl:
                return value
            else:
                self._memory_cache.pop(key, None)
        
        # Tenta banco de dados
        if self._use_persistent and self._conn:
            try:
                cursor = self._conn.execute(
                    "SELECT value, timestamp, ttl FROM cache WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                if row:
                    value_blob, ts, ttl = row
                    if time.time() - ts < ttl:
                        try:
                            return pickle.loads(value_blob)
                        except Exception:
                            return json.loads(value_blob.decode('utf-8'))
                    else:
                        # Remove expirado
                        self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                        self._conn.commit()
            except Exception as e:
                print(f"[Cache] Erro ao ler do DB: {e}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """
        Armazena valor no cache.
        
        Args:
            key: Chave do cache
            value: Valor a armazenar
            ttl: Time-to-live em segundos (usa padrão se None)
        """
        ttl = ttl or self._ttl
        ts = time.time()
        
        # Armazena em memória
        self._memory_cache[key] = (ts, value)
        
        # Armazena no banco se habilitado
        if self._use_persistent and self._conn:
            try:
                # Tenta serializar com pickle primeiro (mais eficiente)
                try:
                    value_blob = pickle.dumps(value)
                except Exception:
                    # Fallback para JSON
                    value_blob = json.dumps(value).encode('utf-8')
                
                self._conn.execute(
                    "INSERT OR REPLACE INTO cache (key, value, timestamp, ttl) VALUES (?, ?, ?, ?)",
                    (key, value_blob, ts, ttl)
                )
                self._conn.commit()
            except Exception as e:
                print(f"[Cache] Erro ao escrever no DB: {e}")
        
        # Limpa expirados periodicamente
        if len(self._memory_cache) > 1000:
            self._clean_expired_memory()
        if self._use_persistent:
            self._clean_expired_db()
    
    def clear(self, pattern: Optional[str] = None):
        """
        Limpa cache.
        
        Args:
            pattern: Se fornecido, limpa apenas chaves que começam com pattern
        """
        if pattern:
            # Limpa memória
            keys_to_remove = [k for k in self._memory_cache.keys() if k.startswith(pattern)]
            for k in keys_to_remove:
                self._memory_cache.pop(k, None)
            
            # Limpa DB
            if self._use_persistent and self._conn:
                try:
                    self._conn.execute("DELETE FROM cache WHERE key LIKE ?", (f"{pattern}%",))
                    self._conn.commit()
                except Exception as e:
                    print(f"[Cache] Erro ao limpar DB: {e}")
        else:
            self._memory_cache.clear()
            if self._use_persistent and self._conn:
                try:
                    self._conn.execute("DELETE FROM cache")
                    self._conn.commit()
                except Exception as e:
                    print(f"[Cache] Erro ao limpar DB: {e}")
    
    def close(self):
        """Fecha conexão com banco de dados."""
        if self._conn:
            self._conn.close()
            self._conn = None


# Instância global do cache (singleton)
_cache_instance: Optional[CacheManager] = None

def get_cache(use_persistent: bool = False) -> CacheManager:
    """
    Retorna instância global do cache (singleton).
    
    Args:
        use_persistent: Se True, usa cache persistente
        
    Returns:
        Instância do CacheManager
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheManager(use_persistent=use_persistent)
    return _cache_instance

