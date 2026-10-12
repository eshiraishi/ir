"""S7 determinism audit and S8 contamination probe."""

import json
from collections import defaultdict

import numpy as np

from trice.config import datasets_config
from trice.data.build import label_map
from trice.data.fetch import fetch_dataset
from trice.eval.metrics import counts_matrix, f1_of_sums, gold_pairs
from trice.llm.client import ChatClient, EndpointPin, build_body
from trice.prompt.formats import Prompt
from trice.prompt.labels import LabelSet
from trice.prompt.parse import parse
from trice.runner import prompts_for
from trice.seeds import sample
from trice.studies import Run, expand, results_dir

AUDIT_N = 300


def audit_sample(model: str, n: int = AUDIT_N) -> list[tuple[Run, str]]:
    """A fixed, seeded set of (run, query_id) drawn from the model's main TRICE runs."""
    runs = [r for r in expand("S1", models=[model]) if r.condition == "small/full"]
    pairs = [(r, q.id) for r in runs for q, _ in prompts_for(r)]
    return sample(pairs, n, "audit", model, key=lambda x: (x[0].slug, x[1]))


def run_audit(model: str, client: ChatClient, pin: EndpointPin, tag: str, n: int = AUDIT_N) -> str:
    """Send the audit prompts live (cache bypassed) and store the outputs under ``tag``."""
    if client.mode != "bypass":
        raise ValueError("the audit client must use mode='bypass'")
    rows = []
    by_run: dict[Run, dict] = {}
    for run, qid in audit_sample(model, n):
        if run not in by_run:
            by_run[run] = {q.id: (q, p) for q, p in prompts_for(run)}
        q, prompt = by_run[run][qid]
        c = client.complete(build_body(pin, prompt), audit_tag=tag)
        labels = LabelSet(label_map(run.dataset))
        ents = parse(run.fmt, c.text, q.text, labels).entities
        rows.append(
            {
                "run": run.slug,
                "query_id": qid,
                "output": c.text,
                "entities": [list(e) for e in ents],
                "gold": [list(g) for g in gold_pairs(q, labels)],
                "provider": c.provider,
            }
        )
    path = results_dir() / "audit" / model / f"{tag}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return str(path)


def audit_summary(model: str) -> dict:
    """Exact-output agreement across audit repetitions and the run-to-run SD of micro-F1."""
    tags = sorted((results_dir() / "audit" / model).glob("*.jsonl"))
    reps = {p.stem: [json.loads(line) for line in p.read_text().splitlines()] for p in tags}
    if len(reps) < 2:
        return {"repetitions": len(reps)}
    outputs = defaultdict(set)
    f1s = []
    for rows in reps.values():
        for r in rows:
            outputs[(r["run"], r["query_id"])].add(r["output"])
        c = counts_matrix(
            [[tuple(e) for e in r["entities"]] for r in rows],
            [[tuple(g) for g in r["gold"]] for r in rows],
        )
        f1s.append(float(f1_of_sums(c.sum(0))))
    agree = float(np.mean([len(v) == 1 for v in outputs.values()]))
    return {
        "repetitions": len(reps),
        "exact_agreement": agree,
        "f1": f1s,
        "f1_sd": float(np.std(f1s, ddof=1)),
    }


def contamination_probe(model: str, client: ChatClient, pin: EndpointPin, lines: int = 40) -> dict:
    """S8: give the opening ``lines`` of each raw test file and measure how many of the next
    ``lines`` lines the model reproduces exactly (memorization signal; v1 future work)."""
    out = {}
    for name, ds in datasets_config().datasets.items():
        path, _ = fetch_dataset(ds)["test"]
        raw = path.read_text(encoding="utf-8").splitlines()
        head, tail = raw[:lines], raw[lines : 2 * lines]
        prompt = Prompt(
            messages=[
                {
                    "role": "user",
                    "content": "Here is the beginning of a file from a public NLP dataset. "
                    "Continue it exactly, "
                    "line by line, with no commentary.\n\n" + "\n".join(head),
                }
            ]
        )
        text = client.complete(build_body(pin, prompt)).text
        got = text.strip("\n").splitlines()[: len(tail)]
        matches = sum(a.strip() == b.strip() for a, b in zip(got, tail, strict=False))
        out[name] = {
            "lines": len(tail),
            "exact_lines": matches,
            "rate": matches / len(tail) if tail else 0.0,
        }
    path = results_dir() / "contamination" / f"{model}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    return out
