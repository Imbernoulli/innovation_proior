#!/usr/bin/env python3
"""Build the two SFT arms for the Qwen3.5-4B agentic ablation (2026-08-22).

Question being settled: the r3-era verdict dropped the 473 agentic rows as
"实测偏负" (empty-think dilution -> think collapse), but every August recipe
(innnew/ctl/gated_v2/timeonly) silently trains them again. No controlled A/B
exists on the latest corpus. These two arms differ ONLY in the agentic rows.

Arms (identical maintain, identical hyperparams):
  withag = timeonly(innovation_sft HEAD rebuild, 2622 rows incl. 473 agentic)
  noag   = same minus rows with a non-empty `tools` field (the r3 filter)
Shared:  maintain = wave2 x8 replay (the c2_maint8x dose that held FCS at base
         in the 9B exploration; wave3 deliberately NOT used here so the arms
         stay comparable to the exploration line).

The timeonly transform mirrors the user's 2026-08-18 ruling exactly as
implemented in training/FrontierSmith/scripts/build_training_final_innovation.py:
strip the persona sentence and the delivery clause; keep task-specific setup
(agentic tool workflow, v4 C++ contract). The year-None fix is upstream now
(trajectories.json registration), so this script asserts no None years remain.
"""
import gzip, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, "sft", "innovation_sft.jsonl")
WAVE2 = os.path.join(REPO, "sft", "innovation_wave2_sft.jsonl.gz")
OUTDIR = os.path.dirname(os.path.abspath(__file__))
MAINT_K = 8

PERSONA = "You are a good researcher."
DELIVERY = ("When you write code, deliver a single, self-contained, runnable implementation that "
            "respects any stated input/output contract; if an idea is not converging within the "
            "budget, fall back to the simplest correct approach and ship that.")


def timeonly(s):
    import re
    s = (s or "").replace(PERSONA, "").replace(DELIVERY, "")
    s = re.sub(r"[ \t]+", " ", s).replace(" .", ".").strip()
    return s


def main():
    withag, noag, stripped = [], [], 0
    for line in open(SRC):
        r = json.loads(line)
        s = r.get("system") or ""
        assert "year None" not in s, "year-None row survived the trajectories.json fix"
        t = timeonly(s)
        if t != s.strip():
            stripped += 1
        r["system"] = t
        withag.append(r)
        if not (r.get("tools") or "").strip():
            noag.append(r)

    n_agentic = len(withag) - len(noag)
    assert n_agentic == 473, f"expected 473 agentic rows, got {n_agentic}"

    maint = [json.loads(l) for l in gzip.open(WAVE2, "rt")]

    for name, rows in (("innovation_timeonly_withag", withag),
                       ("innovation_timeonly_noag", noag)):
        p = os.path.join(OUTDIR, name + ".jsonl")
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")},
                                   ensure_ascii=False) + "\n")
        print(f"{p}: {len(rows)} rows")

    p = os.path.join(OUTDIR, f"maintain_wave2_x{MAINT_K}.jsonl")
    with open(p, "w") as f:
        for _ in range(MAINT_K):
            for r in maint:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{p}: {len(maint)}x{MAINT_K} = {len(maint)*MAINT_K} rows")
    print(f"systems stripped to timeonly: {stripped}/{len(withag)}")


if __name__ == "__main__":
    main()
