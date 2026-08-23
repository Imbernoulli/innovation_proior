#!/usr/bin/env python3
"""Build the two SFT arms for the Qwen3.5-4B agentic ablation (2026-08-22).

Question being settled: the r3-era verdict dropped the 473 agentic rows as
"实测偏负" (empty-think dilution -> think collapse), but every August recipe
(innnew/ctl/gated_v2/timeonly) silently trains them again. No controlled A/B
exists on the latest corpus. These two arms differ ONLY in the agentic rows.

Arms (identical maintain, identical hyperparams):
  withag = timeonly(innovation_sft HEAD rebuild, 2622 rows incl. 473 agentic)
  noag   = same minus rows with a non-empty `tools` field (the r3 filter)
Shared:  maintain = wave2 (hard-only, 750) + wave3 (5,291) single pass, NO
         replay (user ruling 2026-08-22: replay was a workaround for scarce
         maintain data; the distillation campaign is complete and volume now
         does the job). wave3 ids exclude wave2 by construction (zero overlap).

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
WAVE3 = os.path.join(REPO, "sft", "innovation_wave3_sft.jsonl.gz")
OUTDIR = os.path.dirname(os.path.abspath(__file__))
# One uniform schema across all three files (the loss:None landmine class):
# exactly these keys on every row, absent ones filled with "". wave3's
# pass_rate/samples_used are analysis metadata, not training fields, and
# enable_thinking is uniformly absent (all rows are thinking-mode).
KEEP_KEYS = ("conversations", "system", "tools")

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

    def keep(r):
        return {k: r.get(k, "") for k in KEEP_KEYS}

    maint = [json.loads(l) for l in gzip.open(WAVE2, "rt")]
    n_w2 = len(maint)
    maint += [json.loads(l) for l in gzip.open(WAVE3, "rt")]

    for name, rows in (("innovation_timeonly_withag", withag),
                       ("innovation_timeonly_noag", noag)):
        p = os.path.join(OUTDIR, name + ".jsonl")
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(keep(r), ensure_ascii=False) + "\n")
        print(f"{p}: {len(rows)} rows")

    p = os.path.join(OUTDIR, "maintain_w2w3.jsonl")
    with open(p, "w") as f:
        for r in maint:
            f.write(json.dumps(keep(r), ensure_ascii=False) + "\n")
    print(f"{p}: wave2 {n_w2} + wave3 {len(maint)-n_w2} = {len(maint)} rows (single pass, no replay)")
    print(f"systems stripped to timeonly: {stripped}/{len(withag)}")


if __name__ == "__main__":
    main()
