# TIER: greedy
# The "obvious" first attempt: use the probe samples, compute a local
# curvature (second-difference) indicator at every interior probe node,
# and put a cut at the top (B-1) highest-curvature nodes. No notion of a
# per-region budget, no coarsening, no recycling -- just chase the biggest
# raw indicator values wherever they are.
import sys, json


def main():
    inst = json.load(sys.stdin)
    N0 = inst["N0"]
    B = inst["B"]
    BW = inst["BW"]
    pf = inst["probe_f"]

    curv = []
    for j in range(1, N0):
        c = abs(pf[j - 1] - 2 * pf[j] + pf[j + 1])
        curv.append((c, j))
    curv.sort(key=lambda t: (-t[0], t[1]))
    top = curv[:B - 1]
    idxs = sorted(j for _, j in top)
    cuts = [j * BW for j in idxs]
    print(json.dumps({"cuts": cuts}))


main()
