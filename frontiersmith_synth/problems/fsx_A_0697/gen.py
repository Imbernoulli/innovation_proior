#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE lifetime-shelf-packer instance to stdout.

Format:
  N M PAGE LAMBDA
  size_1 birth_1 death_1        (x N, 1-indexed crates)
  t_1 c_1                       (x M, 1-indexed check events)

Difficulty/trap ladder: testId 1..10, N grows; a narrow size range keeps many
holes mutually interchangeable (best-fit ambiguity), and a "hot" minority of
crates (frequently checked) is planted among a "cold" majority (rarely
checked) so a footprint-only best-fit sweep has no reason to keep the hot
crates clustered in a few aisles -- it will happily let cold crates steal a
just-freed hole and push the next hot crate into brand-new shelf space.
"""
import sys
import random

SIZE_MIN, SIZE_MAX = 4, 7
HOT_SIZES = (8, 9)
PAGE = 4
LAMBDA = 14

N_LADDER = [10, 16, 24, 34, 48, 66, 90, 120, 160, 210]
HOT_TOUCH_RANGE = (6, 10)
COLD_TOUCH_CHOICES = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
BLOCK = 20


def build(test_id: int) -> str:
    rng = random.Random(900041 * test_id + 17)
    idx = min(max(test_id, 1), len(N_LADDER)) - 1
    N = N_LADDER[idx]

    # --- Planted hot chain: K_hot crates whose stays are pairwise disjoint in
    # time (spread across the whole timeline), all sharing a size that no
    # ordinary crate ever uses, all heavily checked. Between every consecutive
    # pair of chain members sits one "decoy" crate of the SAME size, alive
    # exactly across the next chain member's birth instant -- so the one hole
    # that size could reuse is always occupied at the critical moment, and a
    # plain best-fit sweep is forced to open a fresh hole for every single
    # chain member. An allocator that reads the check trace can see the decoys
    # are never checked and simply pin the whole (mutually disjoint) hot
    # chain to one reserved slot, immune to the decoys entirely.
    K_hot = max(3, N // 6)
    HOT_SIZE = rng.choice(HOT_SIZES)

    hot_pairs = []
    decoy_pairs = []
    used_times = set()
    prev_hot_death = None
    for j in range(K_hot):
        S = j * BLOCK
        span = rng.randint(10, 13)
        hb, hd = S + 5, S + 5 + span
        hot_pairs.append((hb, hd))
        used_times.add(hb)
        used_times.add(hd)
        if j > 0:
            db = prev_hot_death + 1
            dd = S + 7
            decoy_pairs.append((db, dd))
            used_times.add(db)
            used_times.add(dd)
        prev_hot_death = hd
    T = K_hot * BLOCK

    remaining = [t for t in range(T) if t not in used_times]
    rng.shuffle(remaining)
    n_cold = N - K_hot - len(decoy_pairs)

    cold_pairs = []
    for i in range(n_cold):
        a, b = remaining[2 * i], remaining[2 * i + 1]
        birth, death = (a, b) if a < b else (b, a)
        cold_pairs.append((birth, death))

    crates = []  # (size, birth, death)
    is_hot = []
    for (b, d) in hot_pairs:
        crates.append([HOT_SIZE, b, d])
        is_hot.append(True)
    for (b, d) in decoy_pairs:
        crates.append([HOT_SIZE, b, d])
        is_hot.append(False)
    for (b, d) in cold_pairs:
        size = rng.randint(SIZE_MIN, SIZE_MAX)
        crates.append([size, b, d])
        is_hot.append(False)

    combined = list(zip(crates, is_hot))
    rng.shuffle(combined)
    crates = [c for c, _ in combined]
    is_hot = [h for _, h in combined]

    events = []
    for i, (size, birth, death) in enumerate(crates):
        span = death - birth
        if is_hot[i]:
            k = rng.randint(*HOT_TOUCH_RANGE)
        else:
            k = rng.choice(COLD_TOUCH_CHOICES)
        k = min(k, span)
        times = set()
        tries = 0
        while len(times) < k and tries < 4 * (k + 5):
            times.add(rng.randint(birth, death - 1))
            tries += 1
        for t in times:
            events.append((t, i + 1))

    events.sort()
    M = len(events)

    lines = [f"{N} {M} {PAGE} {LAMBDA}"]
    for size, birth, death in crates:
        lines.append(f"{size} {birth} {death}")
    for t, c in events:
        lines.append(f"{t} {c}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    test_id = int(sys.argv[1])
    sys.stdout.write(build(test_id))
