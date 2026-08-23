#!/usr/bin/env python3
"""Aggregate validation.json across all generated problems into a summary table.

Reads the AUTHORITATIVE validation.json each problem carries (written by the harness),
NOT any agent self-report. Run the harness over problems/*/ first for ground truth.
"""
import json, sys
from pathlib import Path
from collections import Counter, defaultdict

SYNTH = Path(__file__).resolve().parent.parent
PROB = SYNTH / "problems"


def main():
    rows = []
    for d in sorted(PROB.glob("*/")):
        vj = d / "validation.json"
        meta = d / "meta.json"
        if not vj.exists():
            rows.append({"id": d.name, "verdict": "NO_VALIDATION", "tier": "?", "family": "?"})
            continue
        v = json.loads(vj.read_text())
        m = json.loads(meta.read_text()) if meta.exists() else {}
        met = v.get("metrics", {})
        rows.append({
            "id": d.name,
            "tier": m.get("tier", "?"),
            "family": m.get("family", "?"),
            "title": m.get("title", ""),
            "verdict": v.get("verdict", "?"),
            "trivial": met.get("trivial_mean"),
            "greedy": met.get("greedy_mean"),
            "strong": met.get("strong_mean"),
            "gap": met.get("strong_minus_trivial"),
            "div": met.get("exec_divergence"),
            "errors": v.get("errors", []),
        })

    npass = sum(1 for r in rows if r["verdict"] == "PASS")
    by_tier = defaultdict(lambda: [0, 0])
    for r in rows:
        by_tier[r["tier"]][1] += 1
        if r["verdict"] == "PASS":
            by_tier[r["tier"]][0] += 1

    summary = {
        "total": len(rows),
        "passed": npass,
        "pass_rate": round(npass / max(1, len(rows)), 3),
        "by_tier": {t: {"pass": p, "total": n} for t, (p, n) in sorted(by_tier.items())},
        "verdicts": dict(Counter(r["verdict"] for r in rows)),
        "rows": rows,
    }
    (SYNTH / "reports" / "summary.json").write_text(json.dumps(summary, indent=2))

    # markdown
    lines = [f"# Generation summary\n",
             f"**{npass}/{len(rows)} PASS** ({summary['pass_rate']*100:.0f}%)\n",
             "| tier | pass/total |", "|---|---|"]
    for t, (p, n) in sorted(by_tier.items()):
        lines.append(f"| {t} | {p}/{n} |")
    lines += ["", "| id | tier | family | verdict | trivial | greedy | strong | gap | div |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["tier"], x["id"])):
        lines.append(f"| {r['id']} | {r['tier']} | {r['family']} | {r['verdict']} | "
                     f"{r.get('trivial')} | {r.get('greedy')} | {r.get('strong')} | "
                     f"{r.get('gap')} | {r.get('div')} |")
    (SYNTH / "reports" / "summary.md").write_text("\n".join(lines))
    print(json.dumps({k: summary[k] for k in ("total", "passed", "pass_rate", "by_tier", "verdicts")}, indent=2))


if __name__ == "__main__":
    main()
