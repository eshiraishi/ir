"""Fetch raw dataset files from pinned sources and verify their SHA-256."""

import logging
import subprocess
from pathlib import Path

import httpx

from trice.config import DatasetSpec, FileSpec, SourceSpec, datasets_config, repo_root
from trice.data.schema import sha256_file

log = logging.getLogger(__name__)


class HashMismatchError(RuntimeError):
    pass


def raw_dir() -> Path:
    return repo_root() / "data" / "raw"


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def fetch_git(source: str, spec: SourceSpec, paths: list[str]) -> Path:
    """Materialize ``paths`` of ``spec.url`` at exactly ``spec.commit`` under data/raw/<source>."""
    if not (spec.url and spec.commit):
        raise ValueError(f"git source {source} needs url and commit")
    dest = raw_dir() / f"{source}@{spec.commit[:12]}"
    if all((dest / p).exists() for p in paths):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        _git(dest, "init", "-q")
        _git(dest, "remote", "add", "origin", spec.url)
    _git(dest, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", spec.commit)
    _git(dest, "-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD", "--", *paths)
    return dest


def fetch_http(source: str, url: str) -> Path:
    dest = raw_dir() / source / url.rsplit("/", 1)[-1]
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
        tmp.rename(dest)
    return dest


def verify(path: Path, spec: FileSpec, label: str) -> str:
    """Check ``path`` against its pinned hash (unpinned files are reported, never trusted)."""
    digest = sha256_file(path)
    if spec.sha256 is None:
        log.warning(
            "%s is not pinned yet; record sha256: %s in configs/datasets.yaml", label, digest
        )
    elif digest != spec.sha256:
        raise HashMismatchError(f"{label}: expected {spec.sha256}, got {digest}")
    return digest


def fetch_dataset(ds: DatasetSpec) -> dict[str, tuple[Path, str]]:
    """Return ``{split: (local_path, sha256)}`` for the dataset's train and test files."""
    cfg = datasets_config()
    source = cfg.sources[ds.source]
    out: dict[str, tuple[Path, str]] = {}
    if source.kind == "git":
        paths = [f.path for f in ds.files.values() if f.path]
        root = fetch_git(ds.source, source, paths)
        for split, f in ds.files.items():
            if not f.path:
                raise ValueError(f"{ds.name}/{split}: git file needs a path")
            p = root / f.path
            out[split] = (p, verify(p, f, f"{ds.name}/{split}"))
    else:
        for split, f in ds.files.items():
            if not f.url:
                raise ValueError(f"{ds.name}/{split}: http file needs a url")
            p = fetch_http(ds.source, f.url)
            out[split] = (p, verify(p, f, f"{ds.name}/{split}"))
    return out
