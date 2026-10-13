"""Train-only decisions: the number of demonstrations k* and the prompt format.

k*: LOO micro-F1 over the k grid, isotonic regression (monotone, parameter-free), then the Kneedle
knee (S=1, concave, increasing) on a linear k axis, because prompt cost grows linearly in k.
Without a knee, the smallest k reaching the isotonic maximum.
"""

import numpy as np
from kneed import KneeLocator
from pydantic import BaseModel
from sklearn.isotonic import IsotonicRegression

from trice.schema import FormatName


class KSelection(BaseModel):
    k_star: int
    rule: str
    ks: list[int]
    f1: list[float]
    f1_isotonic: list[float]


def select_k(f1_by_k: dict[int, float]) -> KSelection:
    ks = sorted(f1_by_k)
    f1 = np.array([f1_by_k[k] for k in ks])
    iso = IsotonicRegression(increasing=True).fit_transform(np.array(ks, float), f1)
    knee = KneeLocator(ks, iso, S=1.0, curve="concave", direction="increasing").knee if np.ptp(iso) > 0 else None
    if knee is None:
        k_star, rule = ks[int(np.argmax(iso >= iso.max() - 1e-12))], "isotonic-max"
    else:
        k_star, rule = int(knee), "kneedle"
    return KSelection(k_star=k_star, rule=rule, ks=ks, f1=f1.tolist(), f1_isotonic=iso.tolist())


def select_format(mean_f1: dict[FormatName, float]) -> FormatName:
    """Highest mean LOO F1; ties go to the earliest format (P0 < P1 < P2 < P3)."""
    return max(sorted(mean_f1), key=lambda fmt: mean_f1[fmt])
