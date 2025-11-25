"""Módulo EDA - Refatorado usando módulos core."""

import sys
from pathlib import Path

_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_root))

from decomposicao import mount_eda

__all__ = ["mount_eda"]

