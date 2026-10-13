"""Tables (Markdown + LaTeX through templates/) and figures (seaborn, exported to TikZ)."""

import json
import math
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import matplot2tikz
import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
from jinja2 import Environment
from matplotlib.axes import Axes
from pydantic import BaseModel, ConfigDict

from trice.evaluation.inference import Friedman
from trice.schema import write_text


def format_cell(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def escape_latex(value: object) -> str:
    return "".join("\\" + c if c in "&%$#_{}" else c for c in str(value))


class Reporter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    env: Environment
    out: Path

    def write_table(self, name: str, rows: list[dict], caption: str) -> None:
        columns = list(rows[0]) if rows else []
        for kind in ("md", "tex"):
            text = self.env.get_template(f"table.{kind}.j2").render(
                name=name, caption=caption, columns=columns, rows=rows
            )
            write_text(self.out / f"{name}.{kind}", text)
        write_text(self.out / f"{name}.json", json.dumps(rows, indent=2) + "\n")

    @contextmanager
    def figure(self, name: str, panels: int = 1, size: tuple[float, float] = (4.0, 3.0)) -> Generator[list[Axes]]:
        """Panels in a grid of at most four columns; PDF for viewing and TikZ/PGFPlots (.tex) for the paper."""
        columns = min(4, panels)
        rows = math.ceil(panels / columns)
        figure = plt.figure(figsize=(size[0] * columns, size[1] * rows))
        try:
            yield [figure.add_subplot(rows, columns, i + 1) for i in range(panels)]
            self.out.mkdir(parents=True, exist_ok=True)
            figure.savefig(self.out / f"{name}.pdf", bbox_inches="tight")
            matplot2tikz.save(self.out / f"{name}.tex", figure=figure)
        finally:
            plt.close(figure)

    def plot_lines(self, name: str, data: pd.DataFrame, x: str, y: str, hue: str, col: str | None = None) -> None:
        facets = sorted(data[col].unique()) if col else [None]
        with self.figure(name, panels=len(facets)) as axes:
            for i, (ax, facet) in enumerate(zip(axes, facets, strict=True)):
                panel = data if col is None else data[data[col] == facet]
                # Point estimates only: errorbar=None also keeps seaborn's empty CI bands out of the TikZ export.
                sns.lineplot(data=panel, x=x, y=y, hue=hue, marker="o", errorbar=None, ax=ax, legend=i == 0)
                ax.set_title(str(facet) if col else "")

    def plot_critical_difference(self, name: str, friedman: Friedman) -> None:
        with self.figure(name, size=(6.0, 2.0)) as (ax,):
            sp.critical_difference_diagram(pd.Series(friedman.mean_ranks), pd.DataFrame(friedman.nemenyi), ax=ax)
