"""Pinned raw files -> validated HF ``DatasetDict`` per dataset (validation splits are never read)."""

import hashlib
import re
import subprocess
import urllib.request
from pathlib import Path

from datasets import Dataset, DatasetDict
from pydantic import BaseModel

from trice.config import DatasetSpec, SourceSpec
from trice.schema import Example, LabelMap, PathTemplate, Split, derive_seed


class Corpus(BaseModel):
    raw: PathTemplate
    canonical: PathTemplate
    sources: dict[str, SourceSpec]
    specs: dict[str, DatasetSpec]

    def fetch(self, name: str, split: Split) -> Path:
        spec, file = self.specs[name], self.specs[name].select(split)
        source = self.sources[spec.source]
        path, checkout = (
            self.raw.prepare(source=spec.source, path=file.path),
            self.raw.fill(source=spec.source, path=""),
        )
        if source.commit and not path.exists():
            git = ["git", "-C", str(checkout)]
            if not (checkout / ".git").exists():
                subprocess.run([*git, "init", "-q"], check=True)
                subprocess.run([*git, "remote", "add", "origin", source.url], check=True)
            subprocess.run(
                [*git, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", source.commit], check=True
            )
            subprocess.run([*git, "checkout", "-q", "FETCH_HEAD", "--", spec.train.path, spec.test.path], check=True)
        if not source.commit and not path.exists():
            urllib.request.urlretrieve(f"{source.url}/{file.path}", path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if file.sha256 not in (None, digest):
            raise ValueError(f"{name}/{split}: sha256 {digest} != pinned {file.sha256}")
        return path

    def build(self, name: str) -> DatasetDict:
        spec = self.specs[name]
        splits: dict[str, Dataset] = {}
        for split in ("train", "test"):
            # token<TAB>tag (or tag<TAB>token) sentences separated by whitespace-only lines
            # (WNUT-17 uses a lone tab); -DOCSTART- is skipped.
            rows = []
            for block in re.split(r"\n\s*\n", self.fetch(name, split).read_text(encoding="utf-8").replace("\r", "")):
                cells = [line.strip().split("\t") for line in block.splitlines() if line.strip()]
                cells = [c if spec.columns == "token_tag" else c[::-1] for c in cells]
                cells = [c for c in cells if c[0] != "-DOCSTART-"]
                if cells:
                    rows.append(
                        Example(
                            id=f"{name}/{split}/{len(rows):06d}",
                            tokens=[c[0] for c in cells],
                            tags=[c[1] for c in cells],
                        )
                    )
            unknown = {label for row in rows for label, _, _ in row.spans} - spec.labels.root.keys()
            if unknown:
                raise ValueError(f"{name}/{split}: labels missing from trice.toml: {sorted(unknown)}")
            expected = spec.select(split).size
            if len(rows) != expected:
                raise ValueError(f"{name}/{split}: {len(rows)} sentences, expected {expected}")
            splits[split] = Dataset.from_list([row.model_dump() for row in rows])
        data = DatasetDict(**splits)
        data.save_to_disk(self.canonical.fill(dataset=name))
        return data

    def load(self, name: str, split: Split) -> list[Example]:
        """``conll2003@50`` keeps the first 50 training sentences of one seeded shuffle, so the
        S6 pools are nested; the test split is never subsampled."""
        base, _, size = name.partition("@")
        data = DatasetDict.load_from_disk(self.canonical.fill(dataset=base))[split]
        if split == "train" and size:
            data = data.shuffle(seed=derive_seed("pool", base)).select(range(int(size)))
        return [Example.model_validate(row) for row in data]

    def load_labels(self, name: str) -> LabelMap:
        return self.specs[name.partition("@")[0]].labels

    def count_pool(self, name: str) -> int:
        base, _, size = name.partition("@")
        return int(size) if size else self.specs[base].train.size
