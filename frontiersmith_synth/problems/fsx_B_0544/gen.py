#!/usr/bin/env python3
"""
gen.py <testId>   ->   prints ONE instance to stdout.

Family: pin-budget-drift-cache  (skin: a lighthouse keeper bolting lamps to a
rotating cliff).  A request trace over `N` lamp keys is split into `E` equal
epochs of `L` requests.  A tiny L1 has `p` BOLT (pin) slots and there is a
fixed-LRU L2 of size `q`.  The solver emits, for every epoch, which lamps are
bolted; the checker replays the trace charging pinned=1, L2=4, miss=40, and
150 per newly-bolted lamp.  The *reconfiguration bandwidth* is `Bmax`: at each
epoch boundary at most Bmax new lamps may be bolted in (unbolting is free).

Planted structure (why the obvious approach is a trap):
  * A DRIFTING working set of W>q warm lamps thrashes L2, so in the no-pin
    baseline the warm lamps mostly MISS -> bolting the right ones is worth ~39.
  * A DECOY lamp with the single highest access frequency but ACCESSED IN TIGHT
    CLUSTERS, so L2 already serves it for 4.  Frequency ranking bolts it and
    wastes a slot; benefit ranking (freq x avoided-cost) skips it.
  * REGIME epochs where >Bmax top lamps turn hot at once.  Because only Bmax
    lamps can be bolted per boundary, the hot set for a regime epoch cannot be
    assembled reactively -- one incoming lamp must be bolted the epoch BEFORE it
    peaks (a swap split across two epochs).  Frequency ranking never sees this.

Everything is seeded from testId only.  testId 1..10 is a difficulty ladder.
"""
import sys, random

# ---- fixed cost model (shared with verify.py) ----
C_PIN, C_L2, C_MISS, C_SWAP = 1, 4, 40, 150
P_SLOTS, Q_L2, BMAX = 5, 4, 2
N_KEYS = 6000
W_WARM = 9                        # concurrent warm working-set size (> Q_L2 -> thrash)
DECOY_KEYS = [1, 2, 3]            # reserved ids for the three decoy lamps
# Flat top ranks: the lamps greedy DROPS to make room for the decoys are just as hot.
WEIGHTS = [52, 51, 50, 49, 48, 20, 14, 9, 5]

ALPHA_WARM = 0.66                 # fraction of a round that is warm-lamp traffic
DECOY_FRAC = 0.105                # PER decoy lamp -> each outranks a hot lamp by frequency
COLD_FRAC  = 0.025
RAMP_FRAC  = 0.03                 # pre-heat of NEXT epoch's incoming top lamps
DECOY_BURSTS = 4                  # each decoy is emitted in this many LONG consecutive runs


def build_schedule(rng, E):
    """Return warm_sets[t] = list of W warm key ids (index 0 = hottest rank),
    plus the set of regime epochs.  Fresh key ids come from a counter so drifted
    lamps are genuinely distinct keys (the hot set moves through key space)."""
    nxt = [100]
    def fresh():
        k = nxt[0]; nxt[0] += 1; return k
    warm = [fresh() for _ in range(W_WARM)]
    warm_sets = []
    # regime epochs: >=3 of them once E is large enough; several top lamps swap at once.
    regimes = set()
    if E >= 6:
        cand = list(range(2, E - 1))
        rng.shuffle(cand)
        k = max(4, E // 2)
        regimes = set(cand[:k])
    for t in range(E):
        warm_sets.append(list(warm))
        # drift INTO t+1
        if (t + 1) in regimes:
            churn = rng.choice([3, 4])
            # rotate several TOP lamps out and bring fresh hot lamps into the top ranks
            for _ in range(churn):
                warm.pop()                     # drop a cold-tail lamp
            newk = [fresh() for _ in range(churn)]
            warm = newk + warm                 # new lamps enter at the HOT ranks
        else:
            warm.pop()                         # drop coldest
            warm.insert(rng.randint(0, W_WARM // 2), fresh())
        warm = warm[:W_WARM]
        while len(warm) < W_WARM:
            warm.append(fresh())
    return warm_sets, regimes


def epoch_requests(rng, L, warm_t, incoming, cold_lo):
    """Build the ordered request list for one epoch."""
    n_decoy = int(round(L * DECOY_FRAC))          # per decoy lamp
    n_cold  = int(round(L * COLD_FRAC))
    n_ramp  = int(round(L * RAMP_FRAC)) if incoming else 0
    n_warm  = L - len(DECOY_KEYS) * n_decoy - n_cold - n_ramp

    # warm multiset by integer Zipf split
    wsum = sum(WEIGHTS)
    counts = [max(1, int(round(n_warm * WEIGHTS[i] / wsum))) for i in range(W_WARM)]
    # fix rounding so total warm == n_warm
    diff = n_warm - sum(counts)
    i = 0
    while diff != 0:
        j = i % W_WARM
        if diff > 0:
            counts[j] += 1; diff -= 1
        elif counts[j] > 1:
            counts[j] -= 1; diff += 1
        i += 1
    base = []
    for i in range(W_WARM):
        base.extend([warm_t[i]] * counts[i])
    # cold: fresh low-reuse keys
    for _ in range(n_cold):
        base.append(rng.randint(cold_lo, N_KEYS - 1))
    # ramp: pre-heat incoming top lamps (few accesses this epoch)
    if incoming:
        for k in range(n_ramp):
            base.append(incoming[k % len(incoming)])
    rng.shuffle(base)                          # spread warm lamps -> reuse gaps > q -> thrash

    # Insert each decoy as a few LONG consecutive bursts.  Inside a burst every
    # access after the first is an L2 hit, so the decoy's per-access cost is close
    # to C_L2 -> its BOLT benefit is low even though its frequency is high.
    # Frequency ranking still bolts it (wasting a slot); benefit ranking does not.
    clusters = []                              # (position_index, decoy_key, run_len)
    for di, dk in enumerate(DECOY_KEYS):
        nb = DECOY_BURSTS
        run = max(1, n_decoy // nb)
        remaining = n_decoy
        step = max(1, len(base) // (nb + 1))
        off = (di * step) // len(DECOY_KEYS)   # stagger the two decoys
        for b in range(nb):
            if remaining <= 0:
                break
            r = run if b < nb - 1 else remaining
            r = min(r, remaining)
            pos = min(len(base), off + b * step)
            clusters.append((pos, dk, r)); remaining -= r
    clusters.sort()
    seq = []
    ci = 0
    for idx in range(len(base) + 1):
        while ci < len(clusters) and clusters[ci][0] == idx:
            _, dk, r = clusters[ci]
            seq.extend([dk] * r); ci += 1
        if idx < len(base):
            seq.append(base[idx])
    while ci < len(clusters):
        seq.extend([clusters[ci][1]] * clusters[ci][2]); ci += 1
    return seq


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rng = random.Random(20250544 + 7919 * tid)

    E = 8 + tid                                 # 9 .. 18 epochs
    L = 900 + 40 * tid                          # 940 .. 1300 requests/epoch
    cold_lo = 2000

    warm_sets, regimes = build_schedule(rng, E)

    seq = []
    for t in range(E):
        nxt_set = set(warm_sets[t + 1]) if t + 1 < E else set()
        incoming = [k for k in (warm_sets[t + 1] if t + 1 < E else []) if k not in set(warm_sets[t])]
        seq.extend(epoch_requests(rng, L, warm_sets[t], incoming[:BMAX + 1], cold_lo))

    T = len(seq)
    out = []
    out.append(str(tid))
    out.append("%d %d %d %d %d %d" % (N_KEYS, E, L, P_SLOTS, Q_L2, BMAX))
    out.append("%d %d %d %d" % (C_PIN, C_L2, C_MISS, C_SWAP))
    out.append(str(T))
    # requests, 40 per line
    for i in range(0, T, 40):
        out.append(" ".join(str(x) for x in seq[i:i + 40]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
