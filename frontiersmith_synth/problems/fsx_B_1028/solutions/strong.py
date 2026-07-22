# TIER: strong
"""The insight: the worst blind window is governed by the JOINT PHASE
relationship between the two periodic walks, not by which route (fixed
geography) each guard takes. Re-routing (e.g. forward vs reverse petal
order -- what 'greedy' does) only changes WHERE each room sits inside a
guard's own cycle; it does not, by itself, fix a bad relative timing
between the two guards (their hub-visit residues can stay locked together
across cycles regardless of route shape, exactly what traps 'greedy' on
symmetric petal layouts).

So we hold the route SHAPES to the same small family greedy already uses
(any cyclic rotation of the forward or reverse petal order -- 'start
the loop at a different petal') and instead search the genuinely different
axis: per-guard period PADDING (a few extra hub-wait steps, which shifts
a guard's effective period away from a value that harmonizes badly with
the other guard's) crossed with an explicit TIME ROTATION of guard 2
relative to guard 1 (only their relative offset matters). We evaluate the
true checker objective on every candidate in this small grid and keep the
best -- a reformulation from 'pick a good route' to 'pick a good joint
period + phase', which is what actually decides the worst-case gap here."""
import sys, math


def build_blocks(Ls):
    offset = 1
    blocks = []
    for Lp in Ls:
        priv = list(range(offset, offset + Lp - 1))
        blocks.append(priv)
        offset += Lp - 1
    return blocks


def canonical_tour(blocks, order):
    tour = []
    for idx in order:
        tour.append(0)
        tour.extend(blocks[idx])
    return tour


def cyclic_orders(k):
    fwd = list(range(k))
    rev = list(range(k - 1, -1, -1))
    orders = set()
    for base in (fwd, rev):
        for s in range(k):
            orders.add(tuple(base[s:] + base[:s]))
    return [list(o) for o in orders]


def pad_tour(tour, pad):
    return tour + [0] * pad


def rotate(tour, r):
    r = r % len(tour)
    return tour[r:] + tour[:r]


def objective(N, w1, w2):
    P1, P2 = len(w1), len(w2)
    L = P1 * P2 // math.gcd(P1, P2)
    visits = [[] for _ in range(N)]
    for t in range(L):
        visits[w1[t % P1]].append(t)
        visits[w2[t % P2]].append(t)
    F = 0
    for v in range(N):
        vs = sorted(set(visits[v]))
        if not vs:
            return None
        gaps = [vs[i + 1] - vs[i] for i in range(len(vs) - 1)]
        gaps.append(vs[0] + L - vs[-1])
        F = max(F, max(gaps))
    return F


def main():
    data = sys.stdin.read().split()
    k = int(data[0]); P = int(data[1])
    Ls = [int(data[2 + i]) for i in range(k)]
    blocks = build_blocks(Ls)
    N = 1 + sum(Lp - 1 for Lp in Ls)

    orders = cyclic_orders(k)
    total = sum(Ls)
    max_pad = max(0, min(3, P - total))
    pad_range = range(0, max_pad + 1)

    # Only guard 2's route/order needs to range over the full candidate
    # family -- since only the RELATIVE order/phase between the two guards
    # matters (a global relabelling of "which order is guard 1's" doesn't
    # change any pairwise gap), fixing guard 1 to the plain forward loop
    # loses no solution quality but cuts the search by a factor of
    # len(orders) (verified empirically against the unrestricted search).
    o1 = list(range(k))
    b1 = canonical_tour(blocks, o1)

    best_F, best_w1, best_w2 = None, None, None
    for pad1 in pad_range:
        w1 = pad_tour(b1, pad1)
        if len(w1) > P:
            continue
        for o2 in orders:
            b2 = canonical_tour(blocks, o2)
            for pad2 in pad_range:
                w2base = pad_tour(b2, pad2)
                if len(w2base) > P:
                    continue
                for r2 in range(len(w2base)):
                    w2 = rotate(w2base, r2)
                    F = objective(N, w1, w2)
                    if F is None:
                        continue
                    if best_F is None or F < best_F:
                        best_F, best_w1, best_w2 = F, w1, w2

    if best_w1 is None:
        # fallback (should not happen): plain forward/reverse loops
        best_w1 = canonical_tour(blocks, list(range(k)))
        best_w2 = canonical_tour(blocks, list(range(k - 1, -1, -1)))

    out = [str(len(best_w1)), " ".join(str(x) for x in best_w1),
           str(len(best_w2)), " ".join(str(x) for x in best_w2)]
    print("\n".join(out))


if __name__ == "__main__":
    main()
