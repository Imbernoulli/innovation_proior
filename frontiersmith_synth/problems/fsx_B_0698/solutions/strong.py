# TIER: strong
# Insight: the mixing law is a power mean (nonlinear, exponent a != 1), and each tank may
# blend at most 2 distinct feedstocks DIRECTLY -- but a tank may also receive ONE wholesale
# TRANSFER from an earlier, already-filled tank, which counts as a single extra "ingredient"
# whose own value was itself shaped by a nested power-mean blend. That lets a final tank
# reach a bespoke composition that no direct <=2-feedstock recipe of the same capacity can
# reach. Strategy: process tanks from LAST to FIRST (so a later tank gets first refusal on
# reserving an earlier tank as a sacrificial premix before that earlier tank is separately
# sold), comparing -- using the TRUE power-mean law -- (a) the best direct 2-feedstock recipe
# against (b) the best 2-stage premix-then-blend recipe net of the opportunity cost of NOT
# selling the premix tank on its own.
import sys


def power_mean(items, a):
    tw = 0.0
    s = 0.0
    for w, v in items:
        tw += w
        s += w * (v ** a)
    return (s / tw) ** (1.0 / a)


def best_price(z, corridors, p0):
    best = p0
    for (lo, hi, price) in corridors:
        if all(lo[k] <= z[k] <= hi[k] for k in range(len(z))) and price > best:
            best = price
    return best


def search_direct(capacity, remaining, feed, exps, corridors, p0, K):
    """Best (price, i1, i2, v) filling `capacity` with <=2 distinct feedstocks, TRUE law."""
    F = len(feed)
    best = None
    for i1 in range(F):
        if remaining[i1] <= 0:
            continue
        if remaining[i1] >= capacity:
            z = feed[i1][1]
            pr = best_price(z, corridors, p0)
            if best is None or pr > best[0]:
                best = (pr, i1, i1, capacity)
        for i2 in range(F):
            if i2 == i1 or remaining[i2] <= 0:
                continue
            lo_v = max(1, capacity - remaining[i2])
            hi_v = min(capacity - 1, remaining[i1])
            for v in range(lo_v, hi_v + 1):
                w2 = capacity - v
                z = [power_mean([(v, feed[i1][1][k]), (w2, feed[i2][1][k])], exps[k]) for k in range(K)]
                pr = best_price(z, corridors, p0)
                if best is None or pr > best[0]:
                    best = (pr, i1, i2, v)
    return best


def search_premix_for_target(cap_p, remainder, remaining, feed, exps, corridors, p0, K):
    """Best (price_B, i1, i2, vp, i3) : premix (i1,i2,vp) fills cap_p, transferred whole
    plus i3 filling `remainder` into the destination tank. Uses TRUE power-mean law."""
    F = len(feed)
    best = None
    for i1 in range(F):
        if remaining[i1] <= 0:
            continue
        cands = []
        if remaining[i1] >= cap_p:
            cands.append((i1, i1, cap_p))
        for i2 in range(F):
            if i2 == i1 or remaining[i2] <= 0:
                continue
            lo_v = max(1, cap_p - remaining[i2])
            hi_v = min(cap_p - 1, remaining[i1])
            for v in range(lo_v, hi_v + 1):
                cands.append((i1, i2, v))
        for (a1, a2, v) in cands:
            w2 = cap_p - v
            zp = [power_mean([(max(v, 1e-9), feed[a1][1][k]), (max(w2, 1e-9), feed[a2][1][k])], exps[k])
                  for k in range(K)]
            temp = remaining[:]
            temp[a1] -= v
            if a2 != a1:
                temp[a2] -= w2
            else:
                temp[a1] -= w2
            if remainder == 0:
                zf = zp
                pr = best_price(zf, corridors, p0)
                cand = (pr, a1, a2, v, -1)
                if best is None or cand[0] > best[0]:
                    best = cand
                continue
            for i3 in range(F):
                if temp[i3] < remainder:
                    continue
                zf = [power_mean([(cap_p, zp[k]), (remainder, feed[i3][1][k])], exps[k]) for k in range(K)]
                pr = best_price(zf, corridors, p0)
                cand = (pr, a1, a2, v, i3)
                if best is None or cand[0] > best[0]:
                    best = cand
    return best


def main():
    toks = sys.stdin.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = toks[p]
        p += 1
        return v

    F = int(nxt()); M = int(nxt()); K = int(nxt()); R = int(nxt())
    exps = [float(nxt()) for _ in range(K)]
    feed = []
    for _ in range(F):
        A = int(nxt())
        x = [float(nxt()) for _ in range(K)]
        feed.append((A, x))
    cap = [int(nxt()) for _ in range(M)]
    corridors = []
    for _ in range(R):
        lo = [0.0] * K
        hi = [0.0] * K
        for k in range(K):
            lo[k] = float(nxt())
            hi[k] = float(nxt())
        price = float(nxt())
        corridors.append((lo, hi, price))
    p0 = float(nxt())

    remaining = [feed[i][0] for i in range(F)]
    reserved = set()          # tanks (0-based) committed as a premix source
    decided = set()           # tanks (0-based) already decided (sold, unused, or reserved)
    moves = {}                # tank(0-based) -> list of instruction strings

    for j in range(M - 1, -1, -1):
        if j in decided:
            continue
        capj = cap[j]
        A = search_direct(capj, remaining, feed, exps, corridors, p0, K)
        price_A = A[0] if A else -1.0

        best_choice = ("A", A)
        best_margin = 0.0
        for pidx in range(j):
            if pidx in decided or pidx in reserved:
                continue
            cap_p = cap[pidx]
            if cap_p > capj:
                continue
            remainder = capj - cap_p
            Bp = search_premix_for_target(cap_p, remainder, remaining, feed, exps, corridors, p0, K)
            if Bp is None:
                continue
            price_B = Bp[0]
            Dp = search_direct(cap_p, remaining, feed, exps, corridors, p0, K)
            dp_direct = Dp[0] if Dp else 0.0
            total_B = price_B * capj
            total_A_plus_p = (price_A if price_A > 0 else 0.0) * capj + dp_direct * cap_p
            margin = total_B - total_A_plus_p
            if margin > best_margin + 1e-9:
                best_margin = margin
                best_choice = ("B", (pidx, Bp))

        if best_choice[0] == "B":
            pidx, Bp = best_choice[1]
            price_B, a1, a2, vp, i3 = Bp
            cap_p = cap[pidx]
            remainder = capj - cap_p
            w2 = cap_p - vp
            mv = []
            if a1 == a2:
                mv.append("POUR %d %d %d" % (pidx + 1, a1 + 1, cap_p))
                remaining[a1] -= cap_p
            else:
                mv.append("POUR %d %d %d" % (pidx + 1, a1 + 1, vp))
                mv.append("POUR %d %d %d" % (pidx + 1, a2 + 1, w2))
                remaining[a1] -= vp
                remaining[a2] -= w2
            mv.append("TRANSFER %d %d" % (pidx + 1, j + 1))
            if remainder > 0 and i3 >= 0:
                mv.append("POUR %d %d %d" % (j + 1, i3 + 1, remainder))
                remaining[i3] -= remainder
            moves[j] = mv
            reserved.add(pidx)
            decided.add(pidx)
            decided.add(j)
        elif A is not None:
            _, i1, i2, v = A
            mv = []
            if i1 == i2:
                mv.append("POUR %d %d %d" % (j + 1, i1 + 1, capj))
                remaining[i1] -= capj
            else:
                w2 = capj - v
                mv.append("POUR %d %d %d" % (j + 1, i1 + 1, v))
                mv.append("POUR %d %d %d" % (j + 1, i2 + 1, w2))
                remaining[i1] -= v
                remaining[i2] -= w2
            moves[j] = mv
            decided.add(j)
        else:
            decided.add(j)

    out = []
    for j in range(M):
        if j in moves:
            out.extend(moves[j])
    print("\n".join(out))


if __name__ == "__main__":
    main()
