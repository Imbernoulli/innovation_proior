# TIER: greedy
# The "obvious" approach: classic merit-order dispatch (load the cheapest
# units -- by linear fuel rate b_i -- up to their efficiency sweet spot
# first, then spread any remaining demand by raw headroom), and only THEN
# check the N-1 reserve rule. If it's violated, patch it by shrinking the
# reserve-eligible group PROPORTIONALLY to each unit's current load and
# dumping the freed output onto whatever headroom is available elsewhere.
# This repair is curvature-blind: it does not know or care that some units
# are far cheaper to displace from their sweet spot than others.
import sys


def merit_order_dispatch(D, N, caps, ms, bs):
    order = sorted(range(N), key=lambda i: bs[i])
    p = [0.0] * N
    rem = D
    for i in order:
        take = min(ms[i], rem)
        p[i] = take
        rem -= take
        if rem <= 1e-9:
            break
    if rem > 1e-9:
        avail = sum(caps[i] - p[i] for i in range(N))
        if avail > 1e-12:
            for i in range(N):
                headroom = caps[i] - p[i]
                p[i] += rem * headroom / avail
    return p


def greedy_dispatch(D, N, caps, ms, bs, fast, J):
    Gmask = [(fast[i] == 1) or (i == J) for i in range(N)]
    RHS = sum(caps[i] for i in range(N) if fast[i] == 1)
    p = merit_order_dispatch(D, N, caps, ms, bs)
    sumG = sum(p[i] for i in range(N) if Gmask[i])
    if sumG <= RHS + 1e-9:
        return p
    over = sumG - RHS
    avail = sum(caps[i] - p[i] for i in range(N) if not Gmask[i])
    transfer = min(over, avail)
    r = (sumG - transfer) / sumG if sumG > 1e-12 else 1.0
    for i in range(N):
        if Gmask[i]:
            p[i] *= r
    if avail > 1e-12:
        for i in range(N):
            if not Gmask[i]:
                headroom = caps[i] - p[i]
                p[i] += transfer * headroom / avail
    return p


def main():
    it = iter(sys.stdin.read().split())
    N = int(next(it)); T = int(next(it))
    caps = []; ms = []; as_ = []; bs = []; fast = []
    for _ in range(N):
        caps.append(int(next(it)))
        ms.append(float(next(it)))
        as_.append(float(next(it)))
        bs.append(float(next(it)))
        fast.append(int(next(it)))
    J = int(next(it)) - 1
    D = [float(next(it)) for _ in range(T)]

    out_lines = []
    for t in range(T):
        p = greedy_dispatch(D[t], N, caps, ms, bs, fast, J)
        out_lines.append(" ".join("%.6f" % x for x in p))
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
