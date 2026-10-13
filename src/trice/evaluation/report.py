"""Tables (Markdown + LaTeX through templates/) and figures (seaborn, exported to TikZ)."""

import json
import math
from collections.abc import Generator, Sequence
from contextlib import contextmanager

import matplot2tikz
import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
from jinja2 import Environment
from matplotlib.axes import Axes
from pydantic import BaseModel, ConfigDict

from trice.evaluation.inference import Friedman
from trice.schema import ensure_parent, write_text


class Reporter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    env: Environment
    tables: str  # path template with {name} and {suffix}

    def write_table(self, name: str, rows: Sequence[BaseModel], caption: str) -> None:
        records = [r.model_dump(mode="json") for r in rows]
        columns = list(records[0]) if len(records) > 0 else []
        markdown = self.env.get_template("table.md.j2").render(
            name=name, caption=caption, columns=columns, rows=records
        )
        write_text(self.tables.format(name=name, suffix="md"), markdown)
        latex = self.env.get_template("table.tex.j2").render(name=name, caption=caption, columns=columns, rows=records)
        write_text(self.tables.format(name=name, suffix="tex"), latex)
        write_text(self.tables.format(name=name, suffix="json"), json.dumps(records, indent=2) + "\n")

    @contextmanager
    def figure(self, name: str, panels: int = 1, size: tuple[float, float] = (4.0, 3.0)) -> Generator[list[Axes]]:
        """Panels in a grid of at most four columns; PDF for viewing and TikZ/PGFPlots (.tex) for the paper."""
        columns = min(4, panels)
        rows = math.ceil(panels / columns)
        figure = plt.figure(figsize=(size[0] * columns, size[1] * rows))
        try:
            yield [figure.add_subplot(rows, columns, i + 1) for i in range(panels)]
            figure.savefig(ensure_parent(self.tables.format(name=name, suffix="pdf")), bbox_inches="tight")
            matplot2tikz.save(ensure_parent(self.tables.format(name=name, suffix="tex")), figure=figure)
        finally:
            plt.close(figure)

    def plot_lines(self, name: str, points: Sequence[BaseModel], x: str, y: str, hue: str, col: str) -> None:
        data = pd.DataFrame([p.model_dump() for p in points])
        facets = sorted(data[col].unique())
        with self.figure(name, panels=len(facets)) as axes:
            for i, (ax, facet) in enumerate(zip(axes, facets, strict=True)):
                panel = data[data[col] == facet]
                # Point estimates only: errorbar=None also keeps seaborn's empty CI bands out of the TikZ export.
                sns.lineplot(data=panel, x=x, y=y, hue=hue, marker="o", errorbar=None, ax=ax, legend=i == 0)
                ax.set_title(str(facet))

    def plot_critical_difference(self, name: str, friedman: Friedman) -> None:
        with self.figure(name, size=(6.0, 2.0)) as (ax,):
            sp.critical_difference_diagram(friedman.mean_ranks, friedman.nemenyi, ax=ax)
