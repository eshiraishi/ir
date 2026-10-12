import pytest

from trice.container import Container
from trice.core.prompting import Labels
from trice.schema import Example


@pytest.fixture(scope="session")
def container() -> Container:
    return Container()


@pytest.fixture
def labels() -> Labels:
    return Labels(raw_to_verbal={"PER": "person", "LOC": "location", "programlang": "programming language"})


def example(i: int, tokens: list[str], tags: list[str] | None = None) -> Example:
    return Example(id=f"d/train/{i:06d}", tokens=tokens, tags=tags or ["O"] * len(tokens))
