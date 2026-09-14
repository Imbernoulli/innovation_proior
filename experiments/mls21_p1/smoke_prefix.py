"""Prefix smoke test: first-call prompt_tokens per task, p1 run vs old run of the same arm."""
import glob, json, sys, os
D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi"
def first_pt(path):
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                for k in ("prompt_tokens", "input_tokens"):
                    if k in d:
                        return d[k]
                u = d.get("usage") or {}
                if "prompt_tokens" in u:
                    return u["prompt_tokens"]
                return None
    except Exception as e:
        return f"ERR:{e}"
def collect(tag):
    out = {}
    for p in glob.glob(f"{D}/mlsroot/logs/*/vllm/{tag}__cc-*/agent/tokens.jsonl"):
        task = p.split("/mlsroot/logs/")[1].split("/")[0]
        out.setdefault(task, []).append(first_pt(p))
    return out
for arm in sys.argv[1:]:
    new = collect(arm + "_p1")
    old = collect(arm)
    print(f"== {arm}  new tasks={len(new)} old tasks={len(old)}")
    bad = 0
    for task in sorted(new):
        n = new[task]; o = old.get(task, [None])
        n0 = n[-1]; o0 = [x for x in o if x is not None]
        o0 = o0[-1] if o0 else None
        st = "OK" if (n0 == o0 and n0 is not None) else ("old=None" if o0 is None else "MISMATCH")
        if st == "MISMATCH": bad += 1
        print(f"  {task:45s} new={n0!s:>6} old={o0!s:>6} {st}")
    print(f"  -> mismatches: {bad}")
