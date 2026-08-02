#!/usr/bin/env python3
"""
gen.py <testId> -- prints ONE compiler-inline-budget instance to stdout.

A call site i has a dynamic execution frequency, a per-invocation body cost, and the
static code size it would ADD to the caller if inlined. Some call sites sit in a linear
"unlock chain": call site i's parent (an earlier call site, index 0 = none) must ALSO be
inlined, with its OWN entire ancestor chain inlined too, before i's post-inline constant
folding bonus can be realized (constant propagation only specializes an ACTUALLY-PASTED
copy, and only if the constant reaches it through an unbroken inlined path).

testId 1-3   : "warm-up" -- mostly independent call sites, at most one short, cheap chain.
               A plain frequency/density-sorted greedy fill is close to the best achievable.
testId 4-10  : "trap" cases -- 1-3 unlock chains whose EARLY links look unattractive in
               isolation (low frequency, large size, no bonus of their own) so a
               density-sorted greedy skips them for showier standalone call sites -- but
               the chain's LATE link carries a large bonus that only fires if the whole
               chain got inlined, and that combined chain value beats what greedy fills
               the icache budget with instead.

All randomness is seeded ONLY from testId (deterministic, reproducible).
"""
import random
import sys


def add_chain(items, rng, length, freq_range, cost_range, size_range, bonus_frac,
              root_freq_range=None, root_size_range=None):
    """root_freq_range/root_size_range (if given) deliberately understate the first TWO
    links' own removed-overhead density (low freq, large size) so a density-sorted
    recipe ranks them near the bottom and, once the icache budget runs out, excludes them
    -- which breaks the WHOLE chain's bonus even if later links look individually
    attractive."""
    parent = 0
    idxs = []
    for pos in range(length):
        if pos < 2 and root_freq_range is not None:
            freq = rng.randint(*root_freq_range)
            inline_size = rng.randint(*root_size_range)
        else:
            freq = rng.randint(*freq_range)
            inline_size = rng.randint(*size_range)
        base_cost = rng.randint(*cost_range)
        idx = len(items) + 1
        bonus = 0
        if pos > 0:
            bonus = min(base_cost - 1, max(2, int(round(base_cost * bonus_frac))))
        items.append([freq, base_cost, inline_size, parent, bonus])
        idxs.append(idx)
        parent = idx
    return idxs


def build_instance(test_id):
    rng = random.Random(4200 + 7 * test_id)
    is_trap = test_id >= 4

    n_std = 10 + (test_id - 1)
    S_base = rng.randint(300, 500)
    CALL_OVERHEAD = rng.randint(20, 30)
    PENALTY_COEF = rng.randint(2, 4)

    items = []  # each: [freq, base_cost, inline_size, parent(0=none,1-indexed else), bonus]

    for _ in range(n_std):
        freq = rng.randint(20, 500)
        base_cost = rng.randint(3, 15)
        inline_size = rng.randint(15, 220)
        items.append([freq, base_cost, inline_size, 0, 0])

    if is_trap:
        n_chains = rng.randint(3, 4) if test_id < 7 else rng.randint(4, 6)
        for _ in range(n_chains):
            L = rng.randint(4, 7)
            # every link individually has LOW removed-overhead density (freq is modest,
            # inline_size is large) so a density-sorted recipe never ranks any single
            # link near the top -- but base_cost is large and bonus_frac is high, so the
            # WHOLE chain, once fully inlined, unlocks a bonus pool that dwarfs anything
            # a single standalone call site offers.
            add_chain(
                items, rng, L,
                freq_range=(100, 320), cost_range=(90, 170), size_range=(150, 230),
                bonus_frac=rng.uniform(0.90, 0.99),
                root_freq_range=(10, 45), root_size_range=(180, 260),
            )
    else:
        if test_id >= 2:
            add_chain(
                items, rng, rng.randint(2, 3),
                freq_range=(60, 200), cost_range=(30, 90), size_range=(40, 110),
                bonus_frac=0.4,
            )

    # ICACHE_CAP sized relative to the actual instance so the budget is genuinely tight:
    # tight enough that greedy's density fill leaves no slack for a full trap chain.
    total_size = sum(it[2] for it in items)
    frac = rng.uniform(0.46, 0.54) if is_trap else rng.uniform(0.55, 0.68)
    ICACHE_CAP = S_base + int(total_size * frac)

    return S_base, ICACHE_CAP, CALL_OVERHEAD, PENALTY_COEF, items


def main():
    test_id = int(sys.argv[1])
    S_base, ICACHE_CAP, CALL_OVERHEAD, PENALTY_COEF, items = build_instance(test_id)
    m = len(items)
    lines = [str(m), f"{S_base} {ICACHE_CAP} {CALL_OVERHEAD} {PENALTY_COEF}"]
    for freq, base_cost, inline_size, parent, bonus in items:
        lines.append(f"{freq} {base_cost} {inline_size} {parent} {bonus}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
