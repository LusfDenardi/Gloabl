"""Módulo Forecast - Refatorado usando módulos core."""

import sys
from pathlib import Path

_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_root))

from previsao import mount_previsao

__all__ = ["mount_previsao"]

