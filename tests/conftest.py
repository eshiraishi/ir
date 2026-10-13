from collections.abc import Iterator

import pytest

from trice.compose import App, compose
from trice.config import Config
from trice.schema import Example, LabelMap


@pytest.fixture(scope="session")
def app(tmp_path_factory: pytest.TempPathFactory) -> Iterator[App]:
    base = Config()
    layout = base.layout.model_copy(update={"cache": str(tmp_path_factory.mktemp("cache"))})
    with compose(base.model_copy(update={"layout": layout})) as composed:
        yield composed


@pytest.fixture
def labels() -> LabelMap:
    return LabelMap({"PER": "person", "LOC": "location", "programlang": "programming language"})


def example(i: int, tokens: list[str], tags: list[str] | None = None) -> Example:
    return Example(id=f"d/train/{i:06d}", tokens=tokens, tags=tags or ["O"] * len(tokens))
