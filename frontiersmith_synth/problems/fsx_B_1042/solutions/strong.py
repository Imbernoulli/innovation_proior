# TIER: strong
"""The insight: under a shared feed cap, the fleet objective only decouples into
independent per-pond optimal-stopping problems if growth phases are STAGGERED.
Instead of letting every pond fight for a 1/P sliver of the cap from t=0, we pick
a concurrency level K (how many ponds may grow at once), sort ponds by how fast
their harvest efficiency decays (fast-decaying ponds cannot afford to sit in a
late batch), and place them into ceil(P/K) sequential batches that each get a
near-exclusive C/K share of the cap for their own window. Within its window each
pond still solves its own optimal-stopping problem (now against the batch's
effective, much larger, per-step feed), so the classic single-pond calculus is
recovered -- it's the OFFSET assignment across ponds that makes it work. We
brute-force K (cheap: P and T are small) and keep the best fleet total."""
import sys
import math


def best_window(a, b0, e0, decay, tau, start, L, feed_per_step):
    """Best switch offset w in [0, L] for a pond that begins growing at `start`
    and receives `feed_per_step` every active step; returns (w, fuel). If the
    per-step share falls below the pond's activation threshold, growth is zero
    for that window -- so an under-threshold batch is never worth entering."""
    best_w, best_v = 0, e0 * (decay ** start) * b0
    if feed_per_step >= tau - 1e-9:
        for w in range(1, L + 1):
            B = b0 + a * w * math.sqrt(feed_per_step)
            v = e0 * (decay ** (start + w)) * B
            if v > best_v:
                best_v, best_w = v, w
    return best_w, best_v


def eval_order(P, T, C, ponds, K, order):
    """Given a concurrency level K and a permutation `order` of pond indices,
    lay ponds into ceil(P/K) sequential batches (in that order) and solve each
    pond's own optimal-stopping problem against its batch's effective feed."""
    nb = (P + K - 1) // K
    L = T // nb
    if L <= 0:
        return None
    starts = [0] * P
    switches = [0] * P
    feeds = [[] for _ in range(P)]
    per_pond_v = [0.0] * P
    for bi in range(nb):
        batch = order[bi * K:(bi + 1) * K]
        if not batch:
            continue
        m = len(batch)
        start = bi * L
        fps = C / m
        for p in batch:
            a, b0, e0, decay, tau = ponds[p]
            w, v = best_window(a, b0, e0, decay, tau, start, L, fps)
            starts[p] = start
            switches[p] = start + w
            feeds[p] = [fps] * w
            per_pond_v[p] = v
    return sum(per_pond_v), starts, switches, feeds, per_pond_v


def local_search(P, T, C, ponds, K, order):
    """Pairwise-exchange descent: swap two ponds' slots whenever that strictly
    raises the fleet total. Positions are fixed (K, batch layout); only WHO
    occupies which slot changes. Converges to a swap-local optimum -- this is
    the exchange argument behind the staggering insight, applied directly."""
    order = list(order)
    best = eval_order(P, T, C, ponds, K, order)
    if best is None:
        return None
    improved = True
    passes = 0
    while improved and passes < 6:
        improved = False
        passes += 1
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                cand = list(order)
                cand[i], cand[j] = cand[j], cand[i]
                res = eval_order(P, T, C, ponds, K, cand)
                if res is not None and res[0] > best[0] + 1e-9:
                    best = res
                    order = cand
                    improved = True
    return best


def plan_for_K(P, T, C, ponds, K):
    decay_order = sorted(range(P), key=lambda p: ponds[p][3])  # fast decayers first
    identity_order = list(range(P))  # guarantees we never fall below the naive
                                      # fixed-order, fixed-window construction
    candidates = []
    for seed in (decay_order, identity_order):
        res = local_search(P, T, C, ponds, K, seed)
        if res is not None:
            candidates.append(res)
    if not candidates:
        return None
    total, starts, switches, feeds, _ = max(candidates, key=lambda r: r[0])
    return total, starts, switches, feeds


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    P = int(next(it))
    T = int(next(it))
    C = float(next(it))
    ponds = []
    for _ in range(P):
        a = float(next(it))
        b0 = float(next(it))
        e0 = float(next(it))
        decay = float(next(it))
        tau = float(next(it))
        ponds.append((a, b0, e0, decay, tau))

    best = None
    for K in range(1, P + 1):
        plan = plan_for_K(P, T, C, ponds, K)
        if plan is None:
            continue
        total, starts, switches, feeds = plan
        if best is None or total > best[0]:
            best = (total, starts, switches, feeds)

    if best is None:
        # Degenerate fallback: everyone harvests immediately, no growth.
        starts = [0] * P
        switches = [0] * P
        feeds = [[] for _ in range(P)]
    else:
        _, starts, switches, feeds = best

    out = [str(P)]
    for p in range(P):
        out.append(f"{starts[p]} {switches[p]}")
        out.append(" ".join(f"{v:.6f}" for v in feeds[p]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
