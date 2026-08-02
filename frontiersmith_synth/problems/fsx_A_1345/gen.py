#!/usr/bin/env python3
"""
gen.py <testId> -> prints ONE instance to stdout.

Board of N cells (1..N). A pool of P winning lines (each of size 2 or 3) is
built so that the first c_j lines of the pool form the density-j family
L_j (nested, L_1 subset ... subset L_K).

Pool layout (in printed order):
  - 6 "decoy" size-3 lines among a shuffled set of non-hub cells (rotation of
    a shuffled outer-cell list, so no two are identical as sets).
  - 3 size-2 lines: two plain decoy pairs + one "fake hub" pair (a cell with
    exactly ONE size-2 line -- looks structurally similar to the real hub but
    carries no double threat).
  - HUBLINES size-2 lines, each pairing the (seeded-random) real hub cell
    with a distinct outer cell.

Density breakpoints c_1..c_7 are fixed offsets into this layout except for
c_6, whose position (how many hub-lines are already visible) is dialed down
for the two hardest test ids so even the intended construction can only
certify the top level there (extra headroom / difficulty ramp).
"""
import sys
import random


def hub_at_c6_hint(test_id: int) -> int:
    """How many hub lines are visible at density level 6 (before level 7,
    which always includes every hub line)."""
    return 1 if test_id >= 9 else 2


def build(test_id: int):
    rng = random.Random(1000 + 7 * test_id)

    N = 8 + (test_id - 1)          # 8 .. 17
    cells = list(range(1, N + 1))
    hub = rng.choice(cells)
    outer = [c for c in cells if c != hub]
    rng.shuffle(outer)
    L = len(outer)                 # N - 1, >= 7

    # --- hub lines: hub paired with up to 8 distinct outer cells ---
    hublines_n = min(L, 8)
    hub_order = outer[:hublines_n]
    hub_lines = [(2, sorted({hub, o})) for o in hub_order]

    # For the two hardest test ids only (large N, plenty of spare cells),
    # keep decoys OFF the hub-line partners that can already be "live" at
    # low density (c_6 with only 1 hub line included) -- otherwise a decoy
    # pair could coincidentally share a cell with that one hub line and
    # hand the double-threat structure to every test for free. For all
    # other test ids there is no such low hub-count level to protect, so
    # decoys are drawn from the full outer pool (keeps small-N boards from
    # running out of distinct cells).
    if hub_at_c6_hint(test_id) == 1:
        reserved = set(outer[:2])
        dpool = [x for x in outer if x not in reserved]
    else:
        dpool = outer
    dp = len(dpool)
    if dp < 7:
        dpool = outer
        dp = len(dpool)

    # --- 6 decoy size-3 lines: 6 distinct consecutive-window triples ---
    d3 = []
    for i in range(6):
        tri = sorted({dpool[i % dp], dpool[(i + 1) % dp], dpool[(i + 2) % dp]})
        d3.append((3, tri))

    # --- 3 size-2 lines: two plain decoys + one fake-hub pair, all on
    # pairwise-disjoint cells (needs 6 distinct cells out of dp >= 7) ---
    plain1 = sorted({dpool[0], dpool[3]})
    plain2 = sorted({dpool[1], dpool[4]})
    fake_hub = dpool[(dp - 1) % dp]
    fake_partner = dpool[(dp - 2) % dp]
    d2 = [(2, plain1), (2, plain2), (2, sorted({fake_hub, fake_partner}))]

    pool = d3 + d2 + hub_lines
    P = len(pool)

    hub_at_c6 = min(hub_at_c6_hint(test_id), hublines_n)
    # Blocker-safe (via the weight-floor rule) needs sum(2^-size) < 1/2:
    # 3 size-3 decoy lines (3/8 = 0.375) is the largest safe prefix here,
    # so c_1..c_3 sit inside that safe zone and c_4 onward is deliberately
    # past it (unresolved by the floor rule alone).
    c = [1, 2, 3, 6, 9, 9 + hub_at_c6, P]
    # safety: keep strictly increasing and within pool bounds
    fixed = []
    prev = 0
    for v in c:
        v = max(v, prev + 1)
        v = min(v, P)
        fixed.append(v)
        prev = v
    c = fixed
    K = len(c)

    return N, K, pool, c


def main():
    test_id = int(sys.argv[1])
    N, K, pool, c = build(test_id)
    out = [f"{N} {K}", f"{len(pool)}"]
    for size, cellset in pool:
        out.append(f"{size} " + " ".join(str(x) for x in cellset))
    out.append(" ".join(str(x) for x in c))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
