"""Tables (Markdown + LaTeX through templates/) and figures (seaborn, exported to TikZ)."""

import json
from pathlib import Path

import matplot2tikz
import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
from jinja2 import Environment
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict

from trice.evaluation.stats import Friedman


class Reporter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    env: Environment
    out: Path

    def table(self, name: str, rows: list[dict], caption: str) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        columns = list(rows[0]) if rows else []
        for kind in ("md", "tex"):
            text = self.env.get_template(f"table.{kind}.j2").render(
                name=name, caption=caption, columns=columns, rows=rows
            )
            (self.out / f"{name}.{kind}").write_text(text)
        (self.out / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n")

    def save(self, name: str, figure: Figure) -> None:
        """PDF for viewing, TikZ/PGFPlots (.tex) for the paper."""
        self.out.mkdir(parents=True, exist_ok=True)
        figure.savefig(self.out / f"{name}.pdf", bbox_inches="tight")
        matplot2tikz.save(self.out / f"{name}.tex", figure=figure)
        plt.close(figure)

    def lines(self, name: str, data: pd.DataFrame, x: str, y: str, hue: str, col: str | None = None) -> None:
        # Point estimates only: errorbar=None also keeps seaborn's empty CI bands out of the TikZ export.
        grid = sns.relplot(
            data=data, x=x, y=y, hue=hue, col=col, kind="line", marker="o", errorbar=None, col_wrap=4 if col else None
        )
        self.save(name, grid.figure)

    def critical_difference(self, name: str, friedman: Friedman) -> None:
        figure = plt.figure(figsize=(6, 2))
        sp.critical_difference_diagram(pd.Series(friedman.mean_ranks), pd.DataFrame(friedman.nemenyi), ax=figure.gca())
        self.save(name, figure)
