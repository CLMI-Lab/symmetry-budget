#!/usr/bin/env python3
"""
run_all.py -- regenerate every number, table and figure in the paper.

Usage
-----
    python run_all.py                        # full protocol (Appendix B), ~10 min
    python run_all.py --preset quick         # coarser grids, ~1 min
    python run_all.py --preset smoke         # minimal, for CI
    python run_all.py --stages graded,nonlinear
    python run_all.py --resume               # reuse cached stage results
    python run_all.py --no-figures

Outputs (all under --out, default ./results)
--------------------------------------------
    csv/*.csv               every table, long-format and paper-format
    figures/*.pdf|png       Figures 1-5
    checkpoints/models/     fitted W_hat for representative configurations
    checkpoints/stages/     per-stage caches for --resume
    results.json            every headline number, plus the full config
    VALIDATION.md           claim-by-claim verdicts
    validation.csv          the same, machine-readable
    manifest.json           file inventory with sizes and sha256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from symbudget.config import get_config
from symbudget.estimators import CheckpointStore
from symbudget.experiments import STAGES
from symbudget.figures import make_all_figures
from symbudget.validate import render_markdown, validate


def _sha256(path: Path, limit: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(limit):
            h.update(chunk)
    return h.hexdigest()[:16]


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="paper", choices=["paper", "quick", "smoke"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--stages", default="all",
                    help="comma-separated subset of: " + ",".join(STAGES))
    ap.add_argument("--resume", action="store_true",
                    help="reuse cached stage outputs when the config fingerprint matches")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--seed0", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = get_config(args.preset)
    if args.seed0 is not None:
        cfg.seed0 = args.seed0

    out = Path(args.out)
    csv_dir, fig_dir, ckpt_dir = out / "csv", out / "figures", out / "checkpoints"
    for p in (csv_dir, fig_dir, ckpt_dir):
        p.mkdir(parents=True, exist_ok=True)
    store = CheckpointStore(ckpt_dir)

    wanted = list(STAGES) if args.stages == "all" else [s.strip() for s in args.stages.split(",")]
    unknown = [s for s in wanted if s not in STAGES]
    if unknown:
        ap.error(f"unknown stage(s): {unknown}; choose from {list(STAGES)}")

    cache_path = out / "_stage_cache.json"
    cache = {}
    if args.resume and cache_path.exists():
        blob = json.loads(cache_path.read_text())
        if blob.get("fingerprint") == cfg.fingerprint():
            cache = blob.get("summaries", {})
            print(f"[resume] reusing {len(cache)} cached stage summaries")

    tables: dict[str, pd.DataFrame] = {}
    summary: dict = {}
    timings: dict = {}

    print(f"=== symbudget | preset={cfg.name} | fingerprint={cfg.fingerprint()} ===")
    for stage in wanted:
        if stage in cache and (ckpt_dir / "stages" / f"{stage}.done").exists():
            summary[stage] = cache[stage]
            for f in csv_dir.glob("*.csv"):
                tables.setdefault(f.stem, pd.read_csv(f))
            print(f"[skip]  {stage}")
            continue
        t0 = time.time()
        tabs, summ = STAGES[stage](cfg, store)
        dt = time.time() - t0
        timings[stage] = round(dt, 2)
        for name, df in tabs.items():
            df.to_csv(csv_dir / f"{name}.csv", index=False)
        tables.update(tabs)
        summary[stage] = summ
        (ckpt_dir / "stages" / f"{stage}.done").write_text(cfg.fingerprint())
        print(f"[done]  {stage:18s} {dt:7.1f}s  -> {', '.join(tabs) or 'no tables'}")

    # ------------------------------------------------------------- figures
    figs = []
    if not args.no_figures:
        try:
            figs = make_all_figures(tables, summary, cfg, fig_dir)
            print(f"[done]  figures            {len(figs)} files")
        except Exception as exc:                      # pragma: no cover
            print(f"[warn]  figures failed: {exc}")

    # ---------------------------------------------------------- validation
    vdf = validate(summary, cfg)
    vdf.to_csv(out / "validation.csv", index=False)
    (out / "VALIDATION.md").write_text(render_markdown(vdf, summary, cfg))
    counts = vdf.verdict.value_counts().to_dict()
    print("[done]  validation        ", counts)

    # -------------------------------------------------------- results.json
    results = {
        "paper": "The Symmetry Budget: How Much Group Structure Should a Network "
                 "Architecture Assume?",
        "implementation_version": "1.0.0",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preset": cfg.name,
        "config": cfg.to_dict(),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "timings_seconds": timings,
        "total_seconds": round(sum(timings.values()), 2),
        "summary": summary,
        "validation_counts": counts,
        "validation": vdf.to_dict("records"),
        "figures": figs,
        "checkpoints": store.list_models(),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, default=_jsonable))
    cache_path.write_text(json.dumps({"fingerprint": cfg.fingerprint(),
                                      "summaries": summary}, default=_jsonable))

    # ------------------------------------------------------------ manifest
    manifest = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            manifest.append({"path": str(p.relative_to(out)), "bytes": p.stat().st_size,
                             "sha256_16": _sha256(p)})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"=== total {results['total_seconds']}s | {len(manifest)} files under {out} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
