#!/usr/bin/env python3
"""
gen.py <testId> -> prints ONE instance to stdout.

Theme: "one keyboard for three quarrelling languages."
A keyboard has N slots arranged on a 3-row staggered grid. N symbols (0..N-1)
must be assigned one-to-one to the N slots. K=3 synthetic languages each supply
a digraph-frequency table over the N symbols. Every language shares a common
"consensus" sub-alphabet with correlated top digraphs, but each language ALSO
has its own private, highly concentrated "conflict" digraph that the other
languages barely use -- and the three languages' corpora are deliberately of
very different absolute size (a big-corpus language, a medium one, a tiny one).

Determinism: all randomness comes from Python's `random.Random(testId)`.
"""
import sys
import random


def slot_coords(n):
    """3-row staggered keyboard grid; returns list of (x, y) for slots 0..n-1."""
    r0 = (n + 2) // 3
    remaining = n - r0
    r1 = (remaining + 1) // 2
    r2 = remaining - r1
    rows = [r0, r1, r2]
    coords = []
    row_offsets = [0.0, 0.28, 0.55]
    for r, size in enumerate(rows):
        for c in range(size):
            x = c + row_offsets[r]
            y = float(r)
            coords.append((x, y))
    assert len(coords) == n
    return coords


def build_instance(test_id):
    rng = random.Random(test_id)

    n_schedule = [6, 8, 10, 12, 14, 16, 18, 20, 24, 28]
    n = n_schedule[(test_id - 1) % len(n_schedule)]
    k = 3

    coords = slot_coords(n)

    symbols = list(range(n))
    rng.shuffle(symbols)
    core_size = max(2, n // 2)
    core = symbols[:core_size]
    periphery = symbols[core_size:]

    # --- consensus core: a deterministic-fraction subset of core pairs gets a
    # shared "base weight" (with small per-language jitter) -- true common ground.
    core_pairs = [(min(core[a], core[b]), max(core[a], core[b]))
                  for a in range(len(core)) for b in range(a + 1, len(core))]
    rng.shuffle(core_pairs)
    n_consensus = max(1, round(0.35 * len(core_pairs)))
    consensus_pairs = set(core_pairs[:n_consensus])

    # --- per-language conflict pair: exactly ONE distinct, exclusive periphery
    # pair per language (a concentrated private "signature digraph").
    periph_pairs = [(min(periphery[a], periphery[b]), max(periphery[a], periphery[b]))
                     for a in range(len(periphery)) for b in range(a + 1, len(periphery))]
    rng.shuffle(periph_pairs)
    conflict_owner = {}  # pair -> language index
    for lang in range(k):
        if lang < len(periph_pairs):
            conflict_owner[periph_pairs[lang]] = lang

    # --- per-language scale: different absolute corpus sizes (this is the
    # trap: pooled/summed raw counts are dominated by whichever language has the
    # largest scale, even though the graded objective normalizes per language).
    # Severity ramps with test_id (difficulty ladder): early/small cases keep a
    # mild disparity so the pooled-frequency recipe still behaves sensibly;
    # later/larger cases use a severe disparity that traps it badly.
    if test_id <= 6:
        scale_options = [4.0, 2.0, 1.0]
    else:
        scale_options = [800.0, 40.0, 1.0]
    rng.shuffle(scale_options)

    # Fixed, N-independent SIGNAL budgets so the consensus/conflict structure
    # stays a robust, dominant fraction of each language's mass at every scale
    # of N (background noise mass would otherwise grow ~N^2 and drown the
    # planted structure on the largest instances).
    CONSENSUS_TOTAL = 100.0
    CONFLICT_WEIGHT = 340.0
    NOISE_LO, NOISE_HI = 0.0, 0.3

    per_consensus_pair = CONSENSUS_TOTAL / max(1, len(consensus_pairs))

    # base (pre-scale) weight tables, symmetric-by-pair, later split into freq[i][j]
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    freq = [[[0] * n for _ in range(n)] for _ in range(k)]

    for (i, j) in all_pairs:
        is_consensus = (i, j) in consensus_pairs
        owner = conflict_owner.get((i, j), None)
        for lang in range(k):
            val = rng.uniform(NOISE_LO, NOISE_HI)  # background noise, always present
            if is_consensus:
                jitter = rng.uniform(0.85, 1.15)
                val += per_consensus_pair * jitter
            if owner == lang:
                val += CONFLICT_WEIGHT
            scaled = val * scale_options[lang]
            cnt = int(round(scaled))
            if cnt < 0:
                cnt = 0
            freq[lang][i][j] = cnt

    return n, k, coords, freq


def main():
    test_id = int(sys.argv[1])
    n, k, coords, freq = build_instance(test_id)

    out = []
    out.append(f"{n} {k}")
    for (x, y) in coords:
        out.append(f"{x:.6f} {y:.6f}")
    for lang in range(k):
        for i in range(n):
            out.append(" ".join(str(freq[lang][i][j]) for j in range(n)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
