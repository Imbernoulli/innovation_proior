#!/usr/bin/env python3
"""Holdout NLL split by source corpus, for a mixed-corpus teacher.

The trainer reports one pooled holdout NLL. When the mix contains a half the
teacher itself generated (maintain q38 came out of Qwen3.8-27B) and a half written
by people (innovation), the pooled number is uninformative: the self-generated
half sits near the floor from step 0 and flattens the average, so a small pooled
drop is consistent BOTH with "learned the human half well" and with "learned
nothing". Only the split tells them apart, and only the human half matters here.

    python3 eval_by_source.py --state .cache/tinker/qwen38_run.json
"""
import argparse, collections, json, sys

import numpy as np
import tinker
from tinker import types as tt


def datum(row):
    ids, w = row["ids"], row["w"]
    return tt.Datum(
        model_input=tt.ModelInput.from_ints(ids[:-1]),
        loss_fn_inputs={
            "target_tokens": tt.TensorData.from_numpy(np.array(ids[1:], dtype=np.int64)),
            "weights": tt.TensorData.from_numpy(np.array(w[1:], dtype=np.float32)),
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=".cache/tinker/qwen38_mix.holdout.jsonl")
    ap.add_argument("--state", default=".cache/tinker/qwen38_run.json")
    ap.add_argument("--base-model", default=None, help="score the UNTRAINED base instead")
    ap.add_argument("--budget", type=int, default=65536)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.holdout) if l.strip()]
    sc = tinker.ServiceClient()
    if a.base_model:
        tc = sc.create_lora_training_client(base_model=a.base_model, rank=32)
        label = f"base {a.base_model}"
    else:
        st = json.load(open(a.state))
        tc = sc.create_training_client_from_state(st["model_path"])
        label = f"trained ({st.get('steps')} steps)"

    order = sorted(range(len(rows)), key=lambda i: rows[i]["n"])
    groups, cur, cmax = [], [], 0
    for i in order:
        n = rows[i]["n"]
        if cur and max(cmax, n) * (len(cur) + 1) > a.budget:
            groups.append(cur); cur, cmax = [], 0
        cur.append(i); cmax = max(cmax, n)
    if cur:
        groups.append(cur)

    acc = collections.defaultdict(lambda: [0.0, 0.0])
    for g in groups:
        out = tc.forward([datum(rows[i]) for i in g], "cross_entropy").result()
        for i, o in zip(g, out.loss_fn_outputs):
            lp = o["logprobs"]
            lp = lp.to_numpy() if hasattr(lp, "to_numpy") else np.array(lp)
            w = np.array(rows[i]["w"][1:], dtype=np.float64)
            m = min(len(lp), len(w))
            s = acc[rows[i]["src"]]
            s[0] += float(-(lp[:m] * w[:m]).sum()); s[1] += float(w[:m].sum())

    print(f"=== holdout NLL by source — {label}")
    tot = [0.0, 0.0]
    for src, (nll, ntok) in sorted(acc.items()):
        n = sum(1 for r in rows if r["src"] == src)
        print(f"  {src:26s} n={n:4d}  {ntok/1e6:5.2f}M trained tok  nll {nll/ntok:.4f}")
        tot[0] += nll; tot[1] += ntok
    print(f"  {'pooled':26s} n={len(rows):4d}  {tot[1]/1e6:5.2f}M trained tok  nll {tot[0]/tot[1]:.4f}")


if __name__ == "__main__":
    main()
