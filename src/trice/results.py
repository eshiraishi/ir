"""Records the runner persists or returns: the frozen format, k* choices, and S7/S8 checks."""

from pydantic import BaseModel, RootModel

from trice.core.selection import KSelection
from trice.schema import FormatName


class FormatScore(BaseModel):
    format: FormatName
    mean_f1: float


class FormatChoice(BaseModel):
    winner: FormatName
    scores: list[FormatScore]


class KChoice(BaseModel):
    model: str
    dataset: str
    selection: KSelection
    k_one_se: int  # robustness check: the smallest k within one standard error of the best


class KTable(RootModel[list[KChoice]]):
    def find(self, model: str, dataset: str) -> KChoice:
        for choice in self.root:
            if choice.model == model and choice.dataset == dataset:
                return choice
        raise KeyError(f"no k* chosen yet for {model}/{dataset} (run `trice select-k`)")


class SmokeCheck(BaseModel):
    format: FormatName
    answered: bool
    parses: bool
    no_thinking: bool
    pinned: bool


class AuditRow(BaseModel):
    key: str
    output: str
    tags: list[str]
    gold: list[str]


class AuditReport(BaseModel):
    repetitions: int
    exact_agreement: float
    f1: list[float]


class ProbeScore(BaseModel):
    dataset: str
    share: float  # of the next lines of the raw test file reproduced verbatim


class Mention(BaseModel):
    text: str
    label: str


class ExternalPrediction(BaseModel):
    """One line of a baseline's output (written by envs/baselines)."""

    id: str
    entities: list[Mention]
