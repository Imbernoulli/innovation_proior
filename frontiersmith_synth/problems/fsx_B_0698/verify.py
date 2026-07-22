#!/usr/bin/env python3
# Deterministic checker for order-sensitive-blend-ladders (format C, MAXIMIZE revenue).
# CLI: python3 verify.py <in> <out> <ans>   (ans ignored).
# Prints "... Ratio: <r>" with r in [0,1]; any feasibility breach -> Ratio: 0.0.
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blend_common import power_mean, best_price, read_instance


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    try:
        inst = read_instance(sys.argv[1])
    except Exception:
        fail("bad instance")
        return
    F, M, K = inst["F"], inst["M"], inst["K"]
    a, feed, cap, corridors, p0 = inst["a"], inst["feed"], inst["cap"], inst["corridors"], inst["p0"]

    try:
        raw = open(sys.argv[2]).read()
    except Exception:
        fail("no output")
        return
    lines = [ln.split() for ln in raw.splitlines() if ln.strip()]
    if len(lines) > 3 * M + 64:
        fail("too many instructions")
        return

    pours = {}          # tank(1-based) -> list of (feed_idx, vol)
    pour_feeds = {}      # tank -> set of distinct feedstock idx
    transfer_in = {}     # dest tank -> src tank
    transfer_out_used = set()
    transfer_dest_used = set()

    for tok in lines:
        if not tok:
            continue
        op = tok[0]
        if op == "POUR":
            if len(tok) != 4:
                fail("bad POUR arity")
                return
            try:
                j = int(tok[1]); i = int(tok[2]); v = int(tok[3])
            except Exception:
                fail("bad POUR tokens (non-finite/non-integer)")
                return
            if not (1 <= j <= M):
                fail("POUR tank index out of range")
                return
            if not (1 <= i <= F):
                fail("POUR feedstock index out of range")
                return
            if v <= 0 or v > 10 ** 7:
                fail("POUR volume out of range")
                return
            pours.setdefault(j, []).append((i, v))
            s = pour_feeds.setdefault(j, set())
            s.add(i)
            if len(s) > 2:
                fail("tank %d fed directly by >2 distinct feedstocks" % j)
                return
        elif op == "TRANSFER":
            if len(tok) != 3:
                fail("bad TRANSFER arity")
                return
            try:
                si = int(tok[1]); dj = int(tok[2])
            except Exception:
                fail("bad TRANSFER tokens (non-finite/non-integer)")
                return
            if not (1 <= si <= M) or not (1 <= dj <= M):
                fail("TRANSFER tank index out of range")
                return
            if si >= dj:
                fail("TRANSFER must go strictly forward (source < dest)")
                return
            if si in transfer_out_used:
                fail("tank %d transferred out more than once" % si)
                return
            if dj in transfer_dest_used:
                fail("tank %d receives more than one incoming transfer" % dj)
                return
            transfer_out_used.add(si)
            transfer_dest_used.add(dj)
            transfer_in[dj] = si
        else:
            fail("unknown instruction %r" % op)
            return

    # ---- feedstock availability ----
    used = [0] * (F + 1)
    for j, lst in pours.items():
        for (i, v) in lst:
            used[i] += v
    for i in range(1, F + 1):
        if used[i] > feed[i - 1][0]:
            fail("feedstock %d overused (%d > %d available)" % (i, used[i], feed[i - 1][0]))
            return

    # ---- resolve tanks in increasing order (transfers only reference earlier tanks) ----
    z = {}
    active = set()
    for j in range(1, M + 1):
        items_k = [[] for _ in range(K)]
        total_w = 0
        have_input = False
        if j in pours:
            for (i, v) in pours[j]:
                have_input = True
                total_w += v
                xv = feed[i - 1][1]
                for k in range(K):
                    items_k[k].append((v, xv[k]))
        if j in transfer_in:
            si = transfer_in[j]
            if si not in active:
                fail("tank %d transfers from tank %d which is not filled/resolved" % (j, si))
                return
            have_input = True
            w = cap[si - 1]
            total_w += w
            zs = z[si]
            for k in range(K):
                items_k[k].append((w, zs[k]))
        if not have_input:
            continue
        if total_w != cap[j - 1]:
            fail("tank %d not filled exactly to capacity (got %d need %d)" % (j, total_w, cap[j - 1]))
            return
        try:
            zj = [power_mean(items_k[k], a[k]) for k in range(K)]
        except Exception:
            fail("blend computation failed for tank %d" % j)
            return
        if not all(v == v and abs(v) != float("inf") for v in zj):
            fail("non-finite blend result in tank %d" % j)
            return
        z[j] = zj
        active.add(j)

    # ---- revenue ----
    F_obj = 0.0
    for j in range(1, M + 1):
        if j in active and j not in transfer_out_used:
            price = best_price(z[j], corridors, p0)
            F_obj += cap[j - 1] * price

    # ---- checker's own baseline B: fill each tank (in order) with ONE homogeneous
    #      feedstock, greedily picking whichever single feedstock (with enough remaining
    #      availability) yields the best corridor price. No blending, no premixing. ----
    remaining = [feed[i][0] for i in range(F)]
    B = 0.0
    for j in range(M):
        capj = cap[j]
        best_i = -1
        best_pr = -1.0
        for i in range(F):
            if remaining[i] >= capj:
                pr = best_price(feed[i][1], corridors, p0)
                if pr > best_pr:
                    best_pr = pr
                    best_i = i
        if best_i >= 0:
            B += capj * best_pr
            remaining[best_i] -= capj
    if B <= 0:
        B = 1e-9

    sc = min(1000.0, 100.0 * F_obj / max(1e-9, B))
    print("F_obj=%.3f B=%.3f Ratio: %.6f" % (F_obj, B, sc / 1000.0))


if __name__ == "__main__":
    main()
