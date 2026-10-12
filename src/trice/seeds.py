"""Deterministic seeds and permutations.

Every source of randomness in TRICE goes through :func:`derive_seed`, so a seed is a pure function
of *what it is for* (a purpose string plus identifiers) rather than of call order or global state.
"""

import hashlib
from collections.abc import Sequence

import numpy as np


def derive_seed(*parts: object) -> int:
    """Return a 32-bit seed derived from ``sha256("|".join(parts))``."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def rng(*parts: object) -> np.random.Generator:
    """A fresh PCG64 generator seeded by :func:`derive_seed`."""
    return np.random.Generator(np.random.PCG64(derive_seed(*parts)))


def shuffled[T](items: Sequence[T], *parts: object, key=None) -> list[T]:
    """Deterministically shuffle ``items``.

    Items are first sorted (by ``key`` if given) so the output depends only on the *set* of items
    and the seed parts, never on the order they arrived in (e.g. retrieval rank).
    """
    ordered = sorted(items, key=key)
    perm = rng(*parts).permutation(len(ordered))
    return [ordered[i] for i in perm]


def sample[T](items: Sequence[T], k: int, *parts: object, key=None) -> list[T]:
    """Deterministic sample of ``k`` items without replacement (order-independent input)."""
    return shuffled(items, *parts, key=key)[:k]
