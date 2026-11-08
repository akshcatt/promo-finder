#!/usr/bin/env python3
"""Find promos in a folder of timestamped microdrama screenplays.

  python3 run.py "<show folder>"                                  # rules only, offline, free
  GEMINI_API_KEY=... python3 run.py "<show folder>" --provider gemini
  python3 run.py "<show folder>" --provider custom --base-url https://x/v1 --api-key-env KEY --model m
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

from promo_finder.assemble import LOGLINES, posting_order, rank, stitched_candidates
from promo_finder.heuristics import CONFIG, nms, windows_for_episode
from promo_finder.judge import judge_episode
from promo_finder.llm import LLM, PRESETS, LLMError
from promo_finder.parse import load_show
from promo_finder.report import write_all


def load_dotenv(path: Path = Path(__file__).resolve().parent / ".env") -> None:
    """Project-local keys: KEY=value lines in .env (gitignored). Real env vars win."""
    if not path.exists():
        return
    import os
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("show_dir")
    ap.add_argument("--out", default=None, help="output folder (default outputs/<show>/<mode>)")
    ap.add_argument("--provider", default="none", choices=["none", "custom", *PRESETS])
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--min-interval", type=float, default=4.0, help="seconds between LLM calls (free-tier limits)")
    ap.add_argument("--reasoning-effort", default="auto", help="low|medium|high|none; auto = low for gemini")
    ap.add_argument("--cache-dir", default=".cache", help="where LLM answers are cached")
    ap.add_argument("--episodes", default=None, help="e.g. 1-10 to run a subset")
    ap.add_argument("--per-episode", type=int, default=CONFIG["keep_per_episode"])
    ap.add_argument("--post", type=int, default=10, help="size of the post-first batch")
    ap.add_argument("--no-stitch", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    show_dir = Path(a.show_dir)
    eps = load_show(show_dir)
    if a.episodes:
        lo, _, hi = a.episodes.partition("-")
        eps = [e for e in eps if int(lo) <= e.num <= int(hi or lo)]
    if not eps:
        print(f"no screenplays found in {show_dir}", file=sys.stderr)
        return 1
    LOGLINES.update({e.num: e.logline for e in eps})
    log = [w for e in eps for w in (f"E{e.num:02d}: {x}" for x in e.warnings)]

    llm = None
    if a.provider != "none":
        try:
            llm = LLM(a.provider, a.model, a.base_url, a.api_key_env, min_interval=a.min_interval, cache_dir=a.cache_dir,
                      reasoning_effort=None if a.reasoning_effort == "none" else a.reasoning_effort)
        except LLMError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    mode = f"llm ({llm.name})" if llm else "rules-only"

    singles, scored = [], 0
    for ep in eps:
        ws = windows_for_episode(ep)
        scored += len(ws)
        short = nms(ws, CONFIG["nms_overlap"], a.per_episode)
        if llm:
            short, jlog = judge_episode(llm, ep, short)
            log += jlog
            short = nms(short, CONFIG["nms_overlap"], a.per_episode)
        singles += short
        print(f"E{ep.num:02d}: {len(ws):4d} windows -> {len(short)} kept"
              + (f"  best {short[0].score:.3f}" if short else ""), file=sys.stderr)

    stitched = [] if a.no_stitch else stitched_candidates(eps)
    promos = rank(singles, stitched)
    post = posting_order(promos, a.post, max(e.num for e in eps))

    meta = {"run_at": dt.datetime.now().isoformat(timespec="seconds"), "mode": mode, "episodes": len(eps),
            "candidates_scored": scored, "seconds": round(time.time() - t0, 1), "config": CONFIG,
            "usage": llm.usage if llm else None, "log": log}
    out = Path(a.out) if a.out else Path("outputs") / show_dir.name.replace(" ", "_")[:40] / (
        "rules" if not llm else f"llm_{llm.provider}")
    write_all(out, show_dir.name, promos, post, meta)
    print(f"\n{len(promos)} promos ({len(stitched)} stitched) -> {out}/report.html  [{mode}, {meta['seconds']}s]")
    for i, p in enumerate(post, 1):
        segs = " + ".join(f"E{s['episode']:02d} {s['start']}-{s['end']}" for s in p["segments"])
        print(f"{i:2d}. {segs:38s} {p['total_s']:4.0f}s  {p['archetype']:20s} {p['hook'][:60]}")
    if llm:
        print(f"LLM usage: {llm.usage}")
        fails = [l for l in log if "LLM failed" in l or "no valid model items" in l]
        if fails:
            print(f"\nWARNING: {len(fails)}/{len(eps)} episodes fell back to rules-only scores. First error:\n  {fails[0][:300]}",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
