"""Internal distributed building blocks used by :class:`bae.optim.Schur`."""

from .context import (
    DistributedIndexContext,
    DistributedIndexMode,
)

__all__ = [
    "DistributedIndexContext",
    "DistributedIndexMode",
]
