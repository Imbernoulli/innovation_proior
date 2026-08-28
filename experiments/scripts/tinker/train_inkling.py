"""LoRA-train Inkling-Small on the innovation corpus via the Tinker API.

Input is the token/weight stream produced by build_data.py (Inkling wire format,
per-turn loss mask preserved). Objective is plain shifted cross-entropy with the
mask as per-token weights, so masked context turns contribute exactly zero.
"""
import argparse, json, math, os, random, sys, time

import numpy as np
import tinker
from tinker import types as tt


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def make_datum(row):
    ids, w = row["ids"], row["w"]
    inp = ids[:-1]
    tgt = ids[1:]
    wt = w[1:]                      # weight belongs to the token being predicted
    return tt.Datum(
        model_input=tt.ModelInput.from_ints(inp),
        loss_fn_inputs={
            "target_tokens": tt.TensorData.from_numpy(np.array(tgt, dtype=np.int64)),
            "weights": tt.TensorData.from_numpy(np.array(wt, dtype=np.float32)),
        },
    )


def batches(rows, token_budget, seed):
    """Length-bucketed batches so one long row does not blow the budget."""
    rng = random.Random(seed)
    order = sorted(range(len(rows)), key=lambda i: rows[i]["n"])
    groups, cur, cur_max = [], [], 0
    for i in order:
        n = rows[i]["n"]
        if cur and (max(cur_max, n) * (len(cur) + 1) > token_budget):
            groups.append(cur); cur, cur_max = [], 0
        cur.append(i); cur_max = max(cur_max, n)
    if cur:
        groups.append(cur)
    rng.shuffle(groups)
    return groups


def lr_at(step, total, peak, warmup_frac=0.03, floor_frac=0.1):
    w = max(1, int(total * warmup_frac))
    if step < w:
        return peak * (step + 1) / w
    p = (step - w) / max(1, total - w)
    return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def evaluate(tc, rows, token_budget):
    """Mean per-token NLL over the trained (unmasked) positions."""
    tot_nll = tot_tok = 0.0
    for g in batches(rows, token_budget, seed=0):
        data = [make_datum(rows[i]) for i in g]
        out = tc.forward(data, "cross_entropy").result()
        for r, o in zip((rows[i] for i in g), out.loss_fn_outputs):
            lp = np.array(o["logprobs"].to_numpy() if hasattr(o["logprobs"], "to_numpy")
                          else o["logprobs"], dtype=np.float64)
            w = np.array(r["w"][1:], dtype=np.float64)
            m = min(len(lp), len(w))
            tot_nll += float(-(lp[:m] * w[:m]).sum()); tot_tok += float(w[:m].sum())
    return tot_nll / max(1.0, tot_tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=".cache/tinker/inkling_innov.train.jsonl")
    ap.add_argument("--holdout", default=".cache/tinker/inkling_innov.holdout.jsonl")
    ap.add_argument("--base-model", default="thinkingmachines/Inkling-Small")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--token-budget", type=int, default=32768, help="tokens per forward_backward")
    ap.add_argument("--accum", type=int, default=4, help="fwd_bwd calls per optim_step")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-rows", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--state", default=".cache/tinker/inkling_run.json")
    ap.add_argument("--ckpt-name", default=None,
                    help="checkpoint label; defaults to a slug of --base-model")
    ap.add_argument("--seed", type=int, default=20260827)
    a = ap.parse_args()

    train = load(a.train)
    hold = load(a.holdout)[: a.eval_rows]
    print(f"[train] {len(train)} rows, {sum(r['n'] for r in train)/1e6:.2f}M tok, "
          f"{sum(r['n_train'] for r in train)/1e6:.2f}M trained", flush=True)

    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=a.base_model, rank=a.rank,
                                        user_metadata={"run": "innovation-inkling-small"})
    print(f"[train] model_id={tc.get_info().model_id}", flush=True)

    groups = batches(train, a.token_budget, a.seed)
    n_ep = max(1, int(round(a.epochs)))
    groups = [g for _ in range(n_ep) for g in groups]
    total_steps = max(1, len(groups) // a.accum)
    print(f"[train] {len(groups)} micro-batches -> {total_steps} optim steps "
          f"(accum {a.accum}, budget {a.token_budget} tok)", flush=True)

    log = open(a.state.replace(".json", ".log.jsonl"), "a")
    t0 = time.time(); step = 0; pending = []
    ev = evaluate(tc, hold, a.token_budget)
    print(f"[eval] step 0  holdout nll {ev:.4f}", flush=True)
    log.write(json.dumps({"step": 0, "holdout_nll": ev}) + "\n"); log.flush()

    for bi, g in enumerate(groups):
        data = [make_datum(train[i]) for i in g]
        pending.append(tc.forward_backward(data, "cross_entropy"))
        if len(pending) < a.accum:
            continue
        losses = []
        for f in pending:
            out = f.result()
            for o in out.loss_fn_outputs:
                lp = o["logprobs"]
                lp = lp.to_numpy() if hasattr(lp, "to_numpy") else np.array(lp)
                losses.append(float(-np.mean(lp)))
        pending = []
        lr = lr_at(step, total_steps, a.lr)
        tc.optim_step(tt.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95,
                                    eps=1e-8, weight_decay=0.0)).result()
        step += 1
        if step % 5 == 0 or step == 1:
            print(f"[train] step {step}/{total_steps} lr {lr:.2e} "
                  f"loss~{np.mean(losses):.4f} {time.time()-t0:.0f}s", flush=True)
            log.write(json.dumps({"step": step, "lr": lr,
                                  "train_nll_all_pos": float(np.mean(losses)),
                                  "t": time.time() - t0}) + "\n"); log.flush()
        if a.eval_every and step % a.eval_every == 0:
            ev = evaluate(tc, hold, a.token_budget)
            print(f"[eval] step {step}  holdout nll {ev:.4f}", flush=True)
            log.write(json.dumps({"step": step, "holdout_nll": ev}) + "\n"); log.flush()
        if a.save_every and step % a.save_every == 0:
            p = tc.save_state(name=f"innov-step{step}").result().path
            print(f"[ckpt] step {step} -> {p}", flush=True)

    ev = evaluate(tc, hold, a.token_budget)
    print(f"[eval] final  holdout nll {ev:.4f}", flush=True)
    # durable pointer for the sampling stage: save_weights_for_sampler returns the path
    ckpt = a.ckpt_name or ("innov-" + a.base_model.split("/")[-1].replace(".", "").lower())
    sw = tc.save_weights_for_sampler(name=ckpt).result()
    model_path = getattr(sw, "path", None) or getattr(sw, "model_path", None) or str(sw)
    st = {"model_path": model_path, "base_model": a.base_model, "rank": a.rank,
          "ckpt_name": ckpt, "steps": step, "final_holdout_nll": ev,
          "train_rows": len(train), "train_tokens": sum(r["n"] for r in train)}
    json.dump(st, open(a.state, "w"), indent=2)
    print(f"[done] {json.dumps(st)}", flush=True)
    # prove the pointer round-trips before we walk away from a multi-hour run
    try:
        sc.create_sampling_client(model_path=model_path)
        print("[done] sampling client resolves from model_path", flush=True)
    except Exception as e:
        print(f"[warn] model_path did not resolve: {type(e).__name__} {e}", flush=True)


if __name__ == "__main__":
    main()
