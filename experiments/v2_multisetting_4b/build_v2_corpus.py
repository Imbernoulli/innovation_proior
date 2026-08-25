#!/usr/bin/env python3
"""Training corpus for the v2 multi-setting campaign (2026-08-24).

innovation = build_sft.py HEAD rebuild (INCLUDES agentic v2: 752 folded rows,
164 tasks, rewrite-contract, zero empty think) + timeonly SP (08-18 ruling)
+ <image>/<video>/<audio> neutralized (Qwen3.5 VL processor landmine).
maintain = FULL wave2+wave3 single pass (user ruling: 全量 maintenance),
reused from experiments/agentic_ablation_4b/maintain_w2w3.jsonl.
"""
import json, os, re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, "sft", "innovation_sft.jsonl")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "innovation_v2_timeonly.jsonl")
PERSONA = "You are a good researcher."
DELIVERY = ("When you write code, deliver a single, self-contained, runnable implementation that "
            "respects any stated input/output contract; if an idea is not converging within the "
            "budget, fall back to the simplest correct approach and ship that.")
KEEP = ("conversations", "system", "tools")
n = ag = 0
with open(OUT, "w") as f:
    for line in open(SRC):
        r = json.loads(line)
        s = (r.get("system") or "").replace(PERSONA, "").replace(DELIVERY, "")
        s = re.sub(r"[ \t]+", " ", s).replace(" .", ".").strip()
        assert "year None" not in s
        out = {k: r.get(k, "") for k in KEEP}
        out["system"] = s
        fixed = []
        for t in out["conversations"]:
            t = dict(t); v = t.get("value")
            if isinstance(v, str):
                for tok in ("image", "video", "audio"):
                    t["value"] = v = v.replace(f"<{tok}>", f"⟨{tok}⟩")
            fixed.append(t)
        out["conversations"] = fixed
        if (out.get("tools") or "").strip(): ag += 1
        f.write(json.dumps(out, ensure_ascii=False) + "\n"); n += 1
print(f"{OUT}: {n} rows ({ag} with tools)")
