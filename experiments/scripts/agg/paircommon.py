# paircommon.py -- pair every arm against the anchor on ONE common sample set.
#
# User directive 2026-09-02: "你得保证都跑完，分母相同啊，所有bench都一样."
# pair.py intersects each arm with the anchor SEPARATELY, so every arm got its own
# denominator (n ran 756..848 on frontiercs) -- different arms were scored on different
# questions. This tool intersects across the anchor AND every arm at once, so all the
# printed deltas come off one identical question set.
#
# A key is (data_source, str(ground_truth), sample_idx); a row is usable only if it has
# no `error` field and carries a numeric score. Rows any arm failed on are dropped for
# everyone, which is the price of a common denominator -- COVERAGE below reports what
# each arm lost so the cost is visible rather than silent.
#
#   python3 paircommon.py <anchor> <tag1,tag2,...> <frontiercs|alebench|frontiercs_research>
import json, glob, sys, random

ROOT = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"

def dirs(tag, src):
    sub = "research_thinking_32k_vllm" if src == "frontiercs_research" else "thinking_32k_both_vllm"
    return glob.glob(f"{ROOT}/cc_eval_{tag}_{sub}/shard_*/samples.jsonl")

def load(tag, src):
    ok, seen = {}, set()
    for f in dirs(tag, src):
        for line in open(f):
            try: r = json.loads(line)
            except Exception: continue
            if r.get("data_source") != src: continue
            k = (r["data_source"], str(r["ground_truth"]), r.get("sample_idx"))
            seen.add(k)
            if r.get("error"): continue
            s = (r.get("metrics") or {}).get("score", r.get("score"))
            if s is None: continue
            ok[k] = (float(s), r.get("completion_tokens"))
    return ok, seen

def boot(d, n=5000):
    random.seed(0); N = len(d); out = []
    for _ in range(n):
        out.append(sum(d[random.randrange(N)] for _ in range(N)) / N)
    out.sort(); return out[int(0.025 * n)], out[int(0.975 * n)]

anchor, arms, src = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
B, Bseen = load(anchor, src)
A = {}
for a in arms:
    A[a] = load(a, src)

common = set(B)
for a in arms: common &= set(A[a][0])
common = sorted(common, key=str)

print(f"=== {src} | anchor={anchor} | COMMON n={len(common)} across {len(arms)} arms ===")
print("COVERAGE (rows collected / usable / kept in common):")
print(f"  {anchor:28s} {len(Bseen):5d} / {len(B):5d} / {len(common):5d}")
for a in arms:
    ok, seen = A[a]
    print(f"  {a:28s} {len(seen):5d} / {len(ok):5d} / {len(common):5d}")
if not common:
    sys.exit("no common samples")

mb = sum(B[k][0] for k in common) / len(common)
print(f"\nanchor mean on common set: {mb:.3f}")
for a in arms:
    ok = A[a][0]
    d = [ok[k][0] - B[k][0] for k in common]
    ma = sum(ok[k][0] for k in common) / len(common)
    lo, hi = boot(d)
    ct = sorted(x for k in common if (x := ok[k][1]) is not None)
    ctm = ct[len(ct) // 2] if ct else -1
    pa = 100 * sum(1 for k in common if ok[k][0] > 0) / len(common)
    pb = 100 * sum(1 for k in common if B[k][0] > 0) / len(common)
    print(f"  {a:28s} {mb:.3f} -> {ma:.3f}  {ma-mb:+.3f}  CI [{lo:+.3f},{hi:+.3f}]  "
          f">0: {pb:.1f}%->{pa:.1f}%  ctok_med={ctm}")
