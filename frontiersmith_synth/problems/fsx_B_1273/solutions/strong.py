# TIER: strong
"""The genuine insight: de-risking should be a function of funded ratio AND
remaining horizon JOINTLY, not of age alone. Rather than hand-pick a
formula, this solution runs a coordinate-descent local search directly on
the (year, bucket) policy grid, using the exact same replay mechanics the
checker uses (spelled out in the statement) as its own internal objective.
Because the state space is shared across scenario blocks, this converges
to a policy that: de-risks fast once a block's funded ratio climbs into an
overfunded bucket (protecting the surplus from a later crash, regardless
of how "young" that block still is by age); and stays meaningfully risky
in underfunded buckets as long as horizon remains (needing the growth,
cushioned by the larger catch-up contribution flex gives it there) -- the
exact state-dependent shape a linear age-only glidepath cannot express."""
import random
import sys

LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SWEEPS = 3
SEED = 123
WBB_INIT = [0.85, 0.70, 0.50, 0.30, 0.12]


def read_instance():
    toks = sys.stdin.read().split()
    ptr = 0

    def nxt():
        nonlocal ptr
        v = toks[ptr]
        ptr += 1
        return v

    T = int(nxt())
    M = int(nxt())
    A0 = float(nxt())
    L0 = float(nxt())
    c_base = float(nxt())
    boundaries = [float(nxt()) for _ in range(4)]
    flex = [float(nxt()) for _ in range(5)]
    blocks = []
    for _ in range(M):
        block = []
        for _ in range(T):
            r_risky = float(nxt())
            r_safe = float(nxt())
            dr = float(nxt())
            g = float(nxt())
            block.append((r_risky, r_safe, dr, g))
        blocks.append(block)
    return T, M, A0, L0, c_base, boundaries, flex, blocks


def bucket(fr, boundaries):
    for i, b in enumerate(boundaries):
        if fr < b:
            return i
    return len(boundaries)


def simulate(T, A0, L0, c_base, boundaries, flex, blocks, grid):
    total = 0.0
    for block in blocks:
        A = A0
        L = L0
        for t in range(1, T + 1):
            r_risky, r_safe, dr, g = block[t - 1]
            fr_prev = A / L
            b = bucket(fr_prev, boundaries)
            C = c_base * flex[b]
            w = grid[t - 1][b]
            remaining = T - t + 1
            mult = 1.0 + g - remaining * dr
            mult = min(1.5, max(0.5, mult))
            L = L * mult
            invested_base = A + C
            A = invested_base * (1.0 + w * r_risky + (1.0 - w) * r_safe)
        fr_final = A / L
        if fr_final >= 1.0:
            total += min(1.5, fr_final)
        else:
            total += fr_final * fr_final
    return total / len(blocks)


def main():
    T, M, A0, L0, c_base, boundaries, flex, blocks = read_instance()

    grid = [[WBB_INIT[b] for b in range(5)] for _ in range(T)]
    best_val = simulate(T, A0, L0, c_base, boundaries, flex, blocks, grid)

    rng = random.Random(SEED)
    cells = [(t, b) for t in range(T) for b in range(5)]
    for _ in range(SWEEPS):
        rng.shuffle(cells)
        for (t, b) in cells:
            orig = grid[t][b]
            best_local = orig
            best_local_val = best_val
            for lv in LEVELS:
                if lv == orig:
                    continue
                grid[t][b] = lv
                val = simulate(T, A0, L0, c_base, boundaries, flex, blocks, grid)
                if val > best_local_val:
                    best_local_val = val
                    best_local = lv
            grid[t][b] = best_local
            best_val = best_local_val

    lines = [" ".join(f"{w:.6f}" for w in grid[t]) for t in range(T)]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
