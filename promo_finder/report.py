"""Write the outputs an editor actually uses: ranked markdown/HTML, CSV for sheets, full JSON."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path


def seg_str(p: dict) -> str:
    return " + ".join(f"E{s['episode']:02d} {s['start']}–{s['end']}" for s in p["segments"])


def write_all(out: Path, show: str, promos: list[dict], post: list[dict], meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "promos.json").write_text(json.dumps({"show": show, "meta": meta, "post_first": post,
                                                 "all_promos": promos}, indent=2, ensure_ascii=False))
    with open(out / "promos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quality_rank", "post_order", "kind", "segments", "total_s", "score", "hook_s", "cliff_s",
                    "intensity_s", "standalone_s", "archetype", "hook", "cliffhanger", "why", "flags", "judged_by"])
        order = {(p["kind"], seg_str(p)): i for i, p in enumerate(post, 1)}
        for p in promos:
            s = p["scores"]
            w.writerow([p["quality_rank"], order.get((p["kind"], seg_str(p)), ""), p["kind"], seg_str(p),
                        p["total_s"], p["score"], s.get("hook"), s.get("cliff"), s.get("intensity"),
                        s.get("standalone"), p["archetype"], p["hook"], p["cliffhanger"], p["why"],
                        " | ".join(p["flags"]), p["judged_by"]])
    _md(out / "report.md", show, promos, post, meta)
    _html(out / "report.html", show, promos, post, meta)


def _card_md(i, p) -> str:
    s = p["scores"]
    lines = [f"### {i}. {seg_str(p)}  ({p['total_s']:.1f}s, {p['kind']}, {p['archetype']})",
             f"- **Hook:** {p['hook']}", f"- **Cliffhanger:** {p['cliffhanger']}", f"- **Why it works:** {p['why']}",
             f"- Score {p['score']:.3f} — hook {s['hook']:.2f} · cliff {s['cliff']:.2f} · intensity {s['intensity']:.2f}"
             f" · standalone {s['standalone']:.2f} · judged by {p['judged_by']}"]
    if p["flags"]:
        lines.append(f"- ⚠ {'; '.join(p['flags'])}")
    return "\n".join(lines)


def _md(path, show, promos, post, meta):
    parts = [f"# Promo picks — {show}", "",
             f"Run: {meta['run_at']} · mode `{meta['mode']}` · {meta['episodes']} episodes · "
             f"{meta['candidates_scored']} windows scored · {len(promos)} promos kept · {meta['seconds']}s",
             "", "## Post these first", ""]
    parts += [_card_md(i, p) for i, p in enumerate(post, 1)]
    parts += ["", "## All promos by quality", "",
              "| # | Segments | Len | Score | Kind | Archetype | Hook | Flags |", "|---|---|---|---|---|---|---|---|"]
    for p in promos:
        hook = p["hook"].replace("|", "/")[:70]
        parts.append(f"| {p['quality_rank']} | {seg_str(p)} | {p['total_s']:.0f}s | {p['score']:.3f} | {p['kind']} | "
                     f"{p['archetype']} | {hook} | {len(p['flags'])} |")
    if meta.get("log"):
        parts += ["", "## Run log", ""] + [f"- {l}" for l in meta["log"]]
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _html(path, show, promos, post, meta):
    e = html.escape

    def card(i, p, open_=False):
        s = p["scores"]
        bars = "".join(f"<span class=b><i style='width:{int(v*100)}%'></i>{k}</span>" for k, v in s.items())
        flags = "".join(f"<li>{e(f)}</li>" for f in p["flags"])
        tr = "".join(f"<div>{e(t)}</div>" for t in p["transcript"])
        return (f"<article><header><b>{i}</b><code>{e(seg_str(p))}</code><span>{p['total_s']:.0f}s · {p['kind']} · "
                f"{e(p['archetype'])} · {p['score']:.3f}</span></header>"
                f"<p><em>Hook</em> {e(p['hook'])}</p><p><em>Cliff</em> {e(p['cliffhanger'])}</p>"
                f"<p><em>Why</em> {e(p['why'])}</p><div class=bars>{bars}</div>"
                f"{f'<ul class=flags>{flags}</ul>' if flags else ''}"
                f"<details{' open' if open_ else ''}><summary>Transcript of the clip</summary><div class=tr>{tr}</div></details></article>")

    body = "".join(card(i, p) for i, p in enumerate(post, 1))
    rest = "".join(card(p["quality_rank"], p) for p in promos)
    path.write_text(f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>Promo picks — {e(show)}</title><style>
body{{font:14px/1.45 system-ui;max-width:900px;margin:auto;padding:16px;background:#faf9f7;color:#222}}
article{{background:#fff;border:1px solid #e5e2dc;border-radius:10px;padding:12px 14px;margin:10px 0}}
header{{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}}header b{{font-size:18px}}header span{{color:#777}}
em{{font-style:normal;font-weight:600;display:inline-block;width:52px;color:#8a4b2a}}p{{margin:4px 0}}
.bars{{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0}}.b{{position:relative;width:120px;background:#eee;border-radius:4px;font-size:11px;padding:1px 6px;overflow:hidden}}
.b i{{position:absolute;left:0;top:0;bottom:0;background:#e9b98f;z-index:0}}.b{{z-index:1}}
.flags{{color:#a33;margin:4px 0;padding-left:18px}}.tr{{font:12px ui-monospace,monospace;color:#444;max-height:260px;overflow:auto}}
</style><h1>Promo picks — {e(show)}</h1><p>{e(meta['run_at'])} · mode <b>{e(meta['mode'])}</b> · {meta['episodes']} episodes ·
{meta['candidates_scored']} windows scored · {meta['seconds']}s</p><h2>Post these first</h2>{body}
<h2>All {len(promos)} promos by quality</h2>{rest}""", encoding="utf-8")
