# mls_merge.py -- stitch cc_mls2p_<TAG> (the 2-task patch run) onto cc_mls21_<TAG>
# and report every arm on ONE identical task set.
#
# User directive: "分母相同啊,所有bench都一样." So this never averages over
# whatever each arm happened to score -- it intersects across all arms first and
# prints what each arm lost, the way paircommon.py does for FCS/ALE.
#
# Status semantics matter here and are NOT interchangeable:
#   scored          -- real measurement, usable
#   timeout+scored  -- the task was starved, score forced to 0.0; NOT a measurement
#   agent_failed    -- the driver died before the model was queried; no data
# Only `scored` counts. base_s15 in particular had 2 timeout+scored zeros that
# would otherwise be silently averaged in as if the model had earned them.
#
#   python3 mls_merge.py <tag1> <tag2> ...
import json, os, sys

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"

def load(tag):
    out = {}
    for base in (f"{D}/cc_mls21_{tag}/summary.json", f"{D}/cc_mls2p_{tag}/summary.json"):
        if not os.path.exists(base):
            continue
        s = json.load(open(base))
        tasks = s.get("tasks") or s
        items = tasks.items() if isinstance(tasks, dict) else [(x.get("task"), x) for x in tasks]
        for n, r in items:
            if not isinstance(r, dict):
                continue
            # the patch run wins: it is the one that had the libraries present
            if n in out and "cc_mls2p" not in base:
                continue
            out[n] = r
    return out

tags = sys.argv[1:]
per = {t: load(t) for t in tags}
for t in tags:
    st = {}
    for r in per[t].values():
        st[r.get("status")] = st.get(r.get("status"), 0) + 1
    print(f"{t:28s} {len(per[t]):2d} tasks  " + " ".join(f"{k}={v}" for k, v in sorted(st.items())))

scored = {t: {n for n, r in per[t].items() if r.get("status") == "scored"
              and r.get("score") is not None} for t in tags}
common = sorted(set.intersection(*scored.values())) if scored else []
allnames = sorted(set().union(*(set(p) for p in per.values()))) if per else []
print(f"\nunion={len(allnames)}  COMMON scored across all arms n={len(common)}")
for t in tags:
    print(f"  {t:28s} scored={len(scored[t]):2d}  dropped_from_common={len(scored[t]) - len(common):2d}")

if not common:
    sys.exit("no common scored tasks")

print(f"\n=== mean over the {len(common)} common tasks ===")
rows = []
for t in tags:
    v = [per[t][n]["score"] for n in common]
    rows.append((sum(v) / len(v), sum(1 for x in v if x > 0), t))
for m, nz, t in sorted(rows, reverse=True):
    print(f"  {t:28s} {m:.4f}   nonzero {nz}/{len(common)}")

print(f"\n=== per-task (common set only) ===")
w = max(len(t) for t in tags)
print("  " + " " * 44 + "".join(f"{t[-14:]:>16s}" for t in tags))
for n in common:
    print(f"  {n:44s}" + "".join(f"{per[t][n]['score']:16.3f}" for t in tags))

excl = [n for n in allnames if n not in common]
if excl:
    print(f"\n=== excluded from the common set ({len(excl)}) ===")
    for n in excl:
        print(f"  {n}")
        for t in tags:
            r = per[t].get(n)
            print(f"      {t:28s} {r.get('status') if r else 'ABSENT'!s:16s} score={r.get('score') if r else None}")
