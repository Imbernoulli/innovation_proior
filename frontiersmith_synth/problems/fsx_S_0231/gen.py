import sys, random

# --- polyomino palette ---
AWKWARD = {  # type-0 candidates: awkward pentominoes (single-type coverage leaves gaps)
    "X5": [(1,0),(0,1),(1,1),(2,1),(1,2)],
    "W5": [(0,0),(0,1),(1,1),(1,2),(2,2)],
    "U5": [(0,0),(2,0),(0,1),(1,1),(2,1)],
    "Z5": [(0,0),(1,0),(1,1),(1,2),(2,2)],
    "T5": [(0,0),(1,0),(2,0),(1,1),(1,2)],
    "Y5": [(1,0),(0,1),(1,1),(1,2),(1,3)],
    "F5": [(1,0),(2,0),(0,1),(1,1),(1,2)],
}
FILLERS = {  # supporting templates that pack well and can be steered onto hotspots
    "O4": [(0,0),(1,0),(0,1),(1,1)],
    "I3": [(0,0),(1,0),(2,0)],
    "L3": [(0,0),(0,1),(1,1)],
    "T4": [(0,0),(1,0),(2,0),(1,1)],
    "S4": [(0,0),(1,0),(1,1),(2,1)],
    "L4": [(0,0),(0,1),(0,2),(1,2)],
    "I4": [(0,0),(1,0),(2,0),(3,0)],
    "P5": [(0,0),(1,0),(0,1),(1,1),(0,2)],
    "V3": [(0,0),(0,1),(1,0)],
}


def main():
    i = int(sys.argv[1])
    rng = random.Random(70231 + 131 * i)

    # difficulty ladder: board grows 6 -> 12, more types, more locked cells,
    # weight skew intensifies with i.
    W = H = min(12, 6 + (i - 1) * 3 // 4)   # 6,6,7,8,9,9,10,11,12,12
    P = 3 + (i - 1) % 4                     # 3..6 cycling
    total = W * H
    r_frac = 0.02 + 0.010 * ((i - 1) % 5)   # 0.02 .. 0.06
    R = int(round(r_frac * total))

    # locked cells: keep the top-left 3x3 clear so the baseline can always place a copy
    all_cells = [(x, y) for y in range(H) for x in range(W)]
    pool = [(x, y) for (x, y) in all_cells if not (x < 3 and y < 3)]
    R = min(R, len(pool))
    locked = set(rng.sample(pool, R)) if R > 0 else set()
    locked_sorted = sorted(locked)

    # demand grid: base 1 everywhere, plus a skewed minority of high-demand hotspots.
    # Strong skew makes WHICH cells you cover matter (bounded inventory can't cover all).
    w = {}
    for (x, y) in all_cells:
        w[(x, y)] = 1
    hot_frac = 0.18 + 0.03 * ((i - 1) % 4)   # ~18%..27% hotspots
    n_hot = max(3, int(round(hot_frac * total)))
    hot_cells = rng.sample(all_cells, min(n_hot, total))
    for (x, y) in hot_cells:
        w[(x, y)] = rng.randint(5, 9)

    # panel types
    a_keys = list(AWKWARD.keys())
    f_keys = list(FILLERS.keys())
    t0_shape = AWKWARD[a_keys[i % len(a_keys)]]
    types = [t0_shape]
    chosen_f = rng.sample(f_keys, P - 1)
    for k in chosen_f:
        types.append(FILLERS[k])

    free = total - R

    # stock: TIGHTLY bounded inventory -- total capacity covers only ~60% of free cells,
    # so which intersections you service is what matters (a genuine bounded knapsack).
    # type 0 has small stock (a few copies) -> a modest, non-dominant baseline.
    stocks = []
    stocks.append(3)                            # type-0 baseline: a few awkward copies
    budget = int(round(0.60 * free))            # total cells the inventory may cover
    budget -= 3 * len(types[0])                 # reserve type-0's contribution
    for t in range(1, P):
        sz = len(types[t])
        # split the remaining budget across the P-1 filler types
        cap = max(1, int(round(budget / max(1, (P - 1) * sz))))
        stocks.append(cap)

    out = []
    out.append("%d %d" % (W, H))
    out.append(str(R))
    for (x, y) in locked_sorted:
        out.append("%d %d" % (x, y))
    for y in range(H):
        out.append(" ".join(str(w[(x, y)]) for x in range(W)))
    out.append(str(P))
    for t in range(P):
        shp = types[t]
        out.append("%d %d" % (stocks[t], len(shp)))
        for (dx, dy) in shp:
            out.append("%d %d" % (dx, dy))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
