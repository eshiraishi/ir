"""Pinned raw files -> validated HF ``DatasetDict`` per dataset (validation splits are never read)."""

import hashlib
import re
import subprocess
import urllib.request
from pathlib import Path

from datasets import Dataset, DatasetDict
from pydantic import BaseModel

from trice.config import DatasetSpec, SourceSpec
from trice.schema import Example, LabelMap, Split, derive_seed


class Corpus(BaseModel):
    raw: Path
    canonical: Path
    sources: dict[str, SourceSpec]
    specs: dict[str, DatasetSpec]

    def fetch(self, name: str) -> dict[Split, Path]:
        spec = self.specs[name]
        source = self.sources[spec.source]
        files = spec.files
        dest = self.raw / spec.source
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
            # token<TAB>tag (or tag<TAB>token) sentences separated by whitespace-only lines
            # (WNUT-17 uses a lone tab); -DOCSTART- is skipped.
            rows = []
            for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").replace("\r", "")):
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
            expected = spec.files[split].size
            if len(rows) != expected:
                raise ValueError(f"{name}/{split}: {len(rows)} sentences, expected {expected}")
            splits[split] = Dataset.from_list([row.model_dump() for row in rows])
        data = DatasetDict(**splits)
        data.save_to_disk(self.canonical / name)
        return data

    def load(self, name: str, split: Split) -> list[Example]:
        """``conll2003@50`` keeps the first 50 training sentences of one seeded shuffle, so the
        S6 pools are nested; the test split is never subsampled."""
        base, _, size = name.partition("@")
        data = DatasetDict.load_from_disk(self.canonical / base)[split]
        if split == "train" and size:
            data = data.shuffle(seed=derive_seed("pool", base)).select(range(int(size)))
        return [Example.model_validate(row) for row in data]

    def load_labels(self, name: str) -> LabelMap:
        return self.specs[name.partition("@")[0]].labels

    def count_pool(self, name: str) -> int:
        base, _, size = name.partition("@")
        return int(size) if size else self.specs[base].train.size
