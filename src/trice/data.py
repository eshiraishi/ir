"""Pinned raw files -> validated HF ``DatasetDict`` per dataset (validation splits are never read)."""

import hashlib
import re
import subprocess
import urllib.request
from pathlib import Path

from datasets import Dataset, DatasetDict
from pydantic import BaseModel

from trice.config import DatasetSpec, SourceSpec
from trice.schema import Example, LabelMap, Split, derive_seed, ensure_parent


class Corpus(BaseModel):
    raw: str
    canonical: str
    sources: dict[str, SourceSpec]
    specs: dict[str, DatasetSpec]

    def fetch(self, name: str, split: Split) -> Path:
        spec = self.specs[name]
        source = self.sources[spec.source]
        file = spec.select(split)
        path = Path(ensure_parent(self.raw.format(source=spec.source, path=file.path)))
        checkout = self.raw.format(source=spec.source, path="")
        if source.commit is not None and not path.exists():
            if not Path(checkout, ".git").exists():
                subprocess.run(["git", "-C", checkout, "init", "-q"], check=True)
                subprocess.run(["git", "-C", checkout, "remote", "add", "origin", source.url], check=True)
            subprocess.run(
                ["git", "-C", checkout, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", source.commit],
                check=True,
            )
            subprocess.run(
                ["git", "-C", checkout, "checkout", "-q", "FETCH_HEAD", "--", spec.train.path, spec.test.path],
                check=True,
            )
        if source.commit is None and not path.exists():
            urllib.request.urlretrieve(f"{source.url}/{file.path}", path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != file.sha256:
            raise ValueError(f"{name}/{split}: sha256 {digest} != pinned {file.sha256}")
        return path

    def build_split(self, name: str, split: Split) -> Dataset:
        """token<TAB>tag (or tag<TAB>token) sentences separated by whitespace-only lines (WNUT-17 uses
        a lone tab); -DOCSTART- is skipped."""
        spec = self.specs[name]
        rows = []
        for block in re.split(r"\n\s*\n", self.fetch(name, split).read_text(encoding="utf-8").replace("\r", "")):
            cells = [line.strip().split("\t") for line in block.splitlines() if line.strip()]
            if spec.columns == "tag_token":
                cells = [cell[::-1] for cell in cells]
            cells = [cell for cell in cells if cell[0] != "-DOCSTART-"]
            if len(cells) > 0:
                tokens = [cell[0] for cell in cells]
                tags = [cell[1] for cell in cells]
                rows.append(Example(id=f"{name}/{split}/{len(rows):06d}", tokens=tokens, tags=tags))
        unknown = {label for row in rows for label, _, _ in row.spans} - spec.labels.root.keys()
        if len(unknown) > 0:
            raise ValueError(f"{name}/{split}: labels missing from trice.toml: {sorted(unknown)}")
        if len(rows) != spec.select(split).size:
            raise ValueError(f"{name}/{split}: {len(rows)} sentences, expected {spec.select(split).size}")
        return Dataset.from_list([row.model_dump() for row in rows])

    def build(self, name: str) -> DatasetDict:
        data = DatasetDict({"train": self.build_split(name, "train"), "test": self.build_split(name, "test")})
        data.save_to_disk(self.canonical.format(dataset=name))
        return data

    def load(self, name: str, split: Split) -> list[Example]:
        """``conll2003@50`` keeps the first 50 training sentences of one seeded shuffle, so the
        S6 pools are nested; the test split is never subsampled."""
        base, _, size = name.partition("@")
        data = DatasetDict.load_from_disk(self.canonical.format(dataset=base))[split]
        if split == "train" and size != "":
            data = data.shuffle(seed=derive_seed("pool", base)).select(range(int(size)))
        return [Example.model_validate(row) for row in data]

    def load_labels(self, name: str) -> LabelMap:
        return self.specs[name.partition("@")[0]].labels

    def count_pool(self, name: str) -> int:
        base, _, size = name.partition("@")
        return self.specs[base].train.size if size == "" else int(size)
