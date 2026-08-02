# TIER: strong
"""Insight: under a memory budget, per-key exact tracking is not an option once
the key population explodes, so spend the FEW explicit bucket slots you can
afford on the keys whose *net* contribution (good-minus-abusive) is largest in
magnitude -- not on whoever is simply loudest. A heavy hitter that is mostly
abusive gets slammed to near-zero (cap=1, rate=0); a heavy hitter that is
mostly good gets a generous dedicated bucket sized to its own burst. Every
other key (including the entire cardinality-explosion tail, which is far too
numerous to isolate) is pooled into many small, independent, hash-routed
shared buckets sized to the *aggregate* leftover legitimate rate -- this is
the hierarchical / sketch-style structure that keeps memory bounded while
still isolating the keys that actually matter."""
import sys

RATE_MAX = 5000


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it)); M = int(next(it))
    next(it); next(it); next(it)  # A, B, P -- the group-routing hash is fixed by the
                                   # instance; we only choose H, G and per-bucket params.
    R = int(next(it))

    good = {}
    bad = {}
    for _ in range(R):
        t = int(next(it)); key = int(next(it)); lab = int(next(it))
        if lab == 1:
            good[key] = good.get(key, 0) + 1
        else:
            bad[key] = bad.get(key, 0) + 1

    keys = set(good) | set(bad)
    stats = []
    for k in keys:
        g = good.get(k, 0); b = bad.get(k, 0)
        stats.append((k, g, b, g - b, g + b))

    # rank by |net contribution| (not raw volume) -- the discriminating insight
    stats.sort(key=lambda s: (-abs(s[3]), -s[4], s[0]))

    H = max(0, min(M - 5, 15, len(stats)))
    explicit_keys = stats[:H]
    explicit_set = {s[0] for s in explicit_keys}
    G = M - H
    if G < 1:
        G = 1
        H = M - 1
        explicit_keys = stats[:H]
        explicit_set = {s[0] for s in explicit_keys}

    lines = [f"{H} {G}"]
    for k, g, b, net, vol in explicit_keys:
        if net >= 0:
            cap = min(RATE_MAX, max(50, vol))
            rate = max(1, min(RATE_MAX, vol // T + 2))
        else:
            cap, rate = 1, 0
        lines.append(f"{k} {cap} {rate}")

    # The pooled tail is composition-blind: we cannot tell good from abusive once
    # keys are hashed together, and the tail of a cardinality-explosion attack is
    # typically abuse-majority. Any *sustained* refill there lets the flood keep
    # draining through, so pools get NO refill at all (grate=0) and only a tiny
    # one-shot capacity, scaled gently by the (small) leftover-good signal -- just
    # enough to catch a little genuine long-tail traffic without reopening the
    # floodgates to the flood. This is the deliberate trade-off: sacrifice a sliver
    # of anonymous good traffic to keep the shared pools from bleeding net-negative.
    leftover_good = sum(g for k, g, b, net, vol in stats if k not in explicit_set)
    gcap = 1 + min(4, leftover_good // max(1, G * 20))
    grate = 0
    lines.append(f"{gcap} {grate}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
