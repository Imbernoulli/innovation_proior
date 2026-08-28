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

# Scored through the SAMPLING client, not a training client: what the run persists
# is a sampler_weights path, and create_training_client_from_state wants a state
# written by save_state, which this run never called. compute_logprobs takes the
# full sequence and returns a per-token logprob, so the per-turn weight mask can
# be applied on our side exactly as the trainer does.


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
        cl = sc.create_sampling_client(base_model=a.base_model)
        label = f"base {a.base_model}"
    else:
        st = json.load(open(a.state))
        cl = sc.create_sampling_client(model_path=st["model_path"])
        label = f"trained ({st.get('steps')} steps)"

    from concurrent.futures import ThreadPoolExecutor
    acc = collections.defaultdict(lambda: [0.0, 0.0])
    lock = __import__("threading").Lock()

    def score(row):
        lp = cl.compute_logprobs(tt.ModelInput.from_ints(row["ids"])).result()
        lp = np.array([0.0 if x is None else x for x in lp], dtype=np.float64)
        w = np.array(row["w"], dtype=np.float64)
        # compute_logprobs gives the logprob of each token given its prefix, so
        # position i is the prediction of ids[i]; the mask lines up directly.
        m = min(len(lp), len(w))
        with lock:
            s = acc[row["src"]]
            s[0] += float(-(lp[:m] * w[:m]).sum()); s[1] += float(w[:m].sum())

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(score, rows))

    print(f"=== holdout NLL by source — {label}")
    tot = [0.0, 0.0]
    for src, (nll, ntok) in sorted(acc.items()):
        n = sum(1 for r in rows if r["src"] == src)
        print(f"  {src:26s} n={n:4d}  {ntok/1e6:5.2f}M trained tok  nll {nll/ntok:.4f}")
        tot[0] += nll; tot[1] += ntok
    print(f"  {'pooled':26s} n={len(rows):4d}  {tot[1]/1e6:5.2f}M trained tok  nll {tot[0]/tot[1]:.4f}")


if __name__ == "__main__":
    main()
