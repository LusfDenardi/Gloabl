"""Módulo Beta/Alpha - Refatorado usando módulos core."""

import sys
from pathlib import Path

_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_root))

from analise_beta_alpha import mount_beta

__all__ = ["mount_beta"]

