"""Módulo Macro - Refatorado usando módulos core."""

# Importa o código original mantendo compatibilidade
# TODO: Refatoração completa para usar core/data_fetcher e core/cache
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path para importar macro_app original
_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_root))

# Importa função mount_macro do arquivo original (temporário durante migração)
from macro_app import mount_macro

__all__ = ["mount_macro"]

