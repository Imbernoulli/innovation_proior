# TIER: strong
# Insight: the optimal bolt schedule is a SHORTEST PATH over (epoch, bolt-set)
# states.  Two things frequency ranking cannot see fall out of it:
#   (1) rank lamps by AVOIDED COST (freq x cost-if-not-bolted), estimated from a
#       single no-bolt LRU replay -- so the clustered DECOY (served by L2 for 4)
#       is correctly worthless, while a thrashing warm lamp (misses -> 40) is not;
#   (2) because a boundary admits only Bmax new bolts, the DP will bolt an
#       incoming lamp the epoch BEFORE it peaks (a swap split across two epochs)
#       whenever a regime turns >Bmax lamps hot at once.
#
# The per-key benefit ignores L2 coupling between bolted keys, so the schedule is
# strong but NOT optimal -- headroom remains above it.
import sys
from collections import OrderedDict
from itertools import combinations


def main():
    toks = sys.stdin.read().split()
    idx = 0
    def nxt():
        nonlocal idx
        v = toks[idx]; idx += 1; return v
    tid = int(nxt())
    N = int(nxt()); E = int(nxt()); L = int(nxt()); p = int(nxt()); q = int(nxt()); Bmax = int(nxt())
    cpin = int(nxt()); cl2 = int(nxt()); cmiss = int(nxt()); cswap = int(nxt())
    T = int(nxt())
    seq = [int(nxt()) for _ in range(T)]

    # ---- (1) benefit[t][key] = sum over epoch-t accesses of (no-bolt cost - cpin) ----
    benefit = [dict() for _ in range(E)]
    l2 = OrderedDict()
    for i, x in enumerate(seq):
        t = i // L
        if t >= E:
            t = E - 1
        if x in l2:
            c = cl2; l2.move_to_end(x)
        else:
            c = cmiss; l2[x] = 1
            if len(l2) > q:
                l2.popitem(last=False)
        benefit[t][x] = benefit[t].get(x, 0) + (c - cpin)

    # ---- per-epoch candidate lamps: top (p+3) by max benefit over window {t-1,t,t+1} ----
    CAP = p + 3
    cand = []
    for t in range(E):
        agg = {}
        for tt in (t - 1, t, t + 1):
            if 0 <= tt < E:
                for k, v in benefit[tt].items():
                    if v > agg.get(k, 0):
                        agg[k] = v
        top = sorted(agg.items(), key=lambda kv: -kv[1])[:CAP]
        cand.append([k for k, _ in top])

    # global bit index
    bit = {}
    for t in range(E):
        for k in cand[t]:
            if k not in bit:
                bit[k] = len(bit)

    def mask_of(keys):
        m = 0
        for k in keys:
            m |= (1 << bit[k])
        return m

    # per-epoch list of (mask, keys_tuple, gain)
    states = []
    for t in range(E):
        lst = []
        keys = cand[t]
        for r in range(0, min(p, len(keys)) + 1):
            for comb in combinations(keys, r):
                g = 0
                for k in comb:
                    g += benefit[t].get(k, 0)
                lst.append((mask_of(comb), comb, g))
        states.append(lst)

    NEG = float("-inf")
    # DP: maximise sum(gain) - sum(swap).  dp[t] : mask -> (best_value, keys, prev_mask)
    dp0 = {}
    for mask, keys, g in states[0]:
        val = g - cswap * len(keys)                  # install charge
        if mask not in dp0 or val > dp0[mask][0]:
            dp0[mask] = (val, keys, None)
    dp = [dp0]
    for t in range(1, E):
        cur = {}
        prevdp = dp[t - 1]
        prev_items = list(prevdp.items())
        for mask, keys, g in states[t]:
            best = NEG; bestpm = None
            for pm, (pv, pk, ppm) in prev_items:
                added = bin(mask & ~pm).count("1")
                if added > Bmax:
                    continue
                val = pv + g - cswap * added
                if val > best:
                    best = val; bestpm = pm
            if best > NEG and (mask not in cur or best > cur[mask][0]):
                cur[mask] = (best, keys, bestpm)
        if not cur:                                  # safety: keep empty schedule
            cur[0] = (prevdp.get(0, (NEG,))[0] if 0 in prevdp else max(v[0] for v in prevdp.values()),
                      (), None)
        dp.append(cur)

    # backtrack
    last = dp[E - 1]
    bmask = max(last, key=lambda m: last[m][0])
    chosen = [None] * E
    m = bmask
    for t in range(E - 1, -1, -1):
        val, keys, pm = dp[t][m]
        chosen[t] = keys
        m = pm if pm is not None else 0

    out = []
    for t in range(E):
        ks = chosen[t]
        out.append(str(len(ks)) + ("" if not ks else " " + " ".join(str(k) for k in ks)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
