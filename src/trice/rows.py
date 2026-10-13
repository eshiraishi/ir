"""Rows of the pre-registered tables and the points of their figures."""

from pydantic import BaseModel


class CellRow(BaseModel):
    model: str
    dataset: str
    condition: str
    k: int
    f1: float
    invalid: float  # share of proposed mentions that could not be aligned to the query


class PairedRow(BaseModel):
    model: str
    dataset: str
    versus: str
    delta: float
    p_value: float
    ci_low: float
    ci_high: float
    se: float
    mcnemar_p: float
    p_holm: float


class ModelDeltaRow(BaseModel):
    model: str
    mean_delta: float
    wilcoxon_p: float


class RankRow(BaseModel):
    study: str
    treatment: str
    mean_rank: float
    statistic: float  # Friedman chi-square of the whole study
    p_value: float


class RegretRow(BaseModel):
    model: str
    dataset: str
    k_star: int
    k_oracle: int
    regret: float


class ScalingRow(BaseModel):
    ladder: str
    metric: str
    p_increasing: float
    p_decreasing: float
    p_two_sided: float


class CurvePoint(BaseModel):
    dataset: str
    k: int
    f1: float
    curve: str


class ScalingPoint(BaseModel):
    ladder: str
    params_b: float
    dataset: str
    f1: float


class BaselineRow(BaseModel):
    system: str
    dataset: str
    f1: float
