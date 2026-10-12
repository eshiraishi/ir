"""Pinned raw files -> validated HF ``DatasetDict`` per dataset (validation splits are never read)."""

import hashlib
import re
import subprocess
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from datasets import Dataset, DatasetDict
from pydantic import BaseModel

from trice.schema import Example, derive_seed
from trice.settings import DatasetSpec, FileSpec, SourceSpec, Split


def read_columns(content: str, order: str) -> Iterator[tuple[list[str], list[str]]]:
    """``token<TAB>tag`` (or ``tag<TAB>token``) sentences separated by whitespace-only lines
    (WNUT-17 uses a lone tab); skips -DOCSTART-."""
    for block in re.split(r"\n\s*\n", content.replace("\r", "")):
        rows = [line.strip().split("\t") for line in block.splitlines() if line.strip()]
        rows = [r if order == "token_tag" else r[::-1] for r in rows]
        rows = [r for r in rows if r[0] != "-DOCSTART-"]
        if rows:
            yield [r[0] for r in rows], [r[1] for r in rows]


class Corpus(BaseModel):
    root: Path
    sources: dict[str, SourceSpec]
    specs: dict[str, DatasetSpec]

    def fetch(self, name: str) -> dict[Split, Path]:
        spec = self.specs[name]
        source = self.sources[spec.source]
        files: dict[Split, FileSpec] = {"train": spec.train, "test": spec.test}
        dest = self.root / "raw" / spec.source
        dest.mkdir(parents=True, exist_ok=True)
        paths = {split: dest / f.path for split, f in files.items()}
        if source.commit and not all(p.exists() for p in paths.values()):
            git = ["git", "-C", str(dest)]
            if not (dest / ".git").exists():
                subprocess.run([*git, "init", "-q"], check=True)
                subprocess.run([*git, "remote", "add", "origin", source.url], check=True)
            subprocess.run(
                [*git, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", source.commit], check=True
            )
            subprocess.run([*git, "checkout", "-q", "FETCH_HEAD", "--", *(f.path for f in files.values())], check=True)
        for split, f in files.items():
            if not source.commit and not paths[split].exists():
                urllib.request.urlretrieve(f"{source.url}/{f.path}", paths[split])
            digest = hashlib.sha256(paths[split].read_bytes()).hexdigest()
            if f.sha256 not in (None, digest):
                raise ValueError(f"{name}/{split}: sha256 {digest} != pinned {f.sha256}")
        return paths

    def build(self, name: str) -> DatasetDict:
        spec = self.specs[name]
        splits: dict[str, Dataset] = {}
        for split, path in self.fetch(name).items():
            rows = [
                Example(id=f"{name}/{split}/{i:06d}", tokens=tokens, tags=tags)
                for i, (tokens, tags) in enumerate(read_columns(path.read_text(encoding="utf-8"), spec.columns))
            ]
            unknown = {label for row in rows for label, _, _ in row.spans} - spec.labels.keys()
            if unknown:
                raise ValueError(f"{name}/{split}: labels missing from trice.toml: {sorted(unknown)}")
            expected = getattr(spec, split).size
            if len(rows) != expected:
                raise ValueError(f"{name}/{split}: {len(rows)} sentences, expected {expected}")
            splits[split] = Dataset.from_list([row.model_dump() for row in rows])
        data = DatasetDict(**splits)
        data.save_to_disk(self.root / "canonical" / name)
        return data

    def load(self, name: str, split: Split) -> list[Example]:
        """``conll2003@50`` keeps the first 50 training sentences of one seeded shuffle, so the
        S6 pools are nested; the test split is never subsampled."""
        base, _, size = name.partition("@")
        data = DatasetDict.load_from_disk(self.root / "canonical" / base)[split]
        if split == "train" and size:
            data = data.shuffle(seed=derive_seed("pool", base)).select(range(int(size)))
        return [Example.model_validate(row) for row in data]

    def labels(self, name: str) -> dict[str, str]:
        return self.specs[name.partition("@")[0]].labels

    def pool_size(self, name: str) -> int:
        base, _, size = name.partition("@")
        return int(size) if size else self.specs[base].train.size
