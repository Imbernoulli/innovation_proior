import sys, random

# --- polyomino palette (base shapes as offset lists) ---
AWKWARD = {  # type-0 candidates: irregular pentominoes (limited stock baseline)
    "X5": [(1,0),(0,1),(1,1),(2,1),(1,2)],
    "W5": [(0,0),(0,1),(1,1),(1,2),(2,2)],
    "U5": [(0,0),(2,0),(0,1),(1,1),(2,1)],
    "Z5": [(0,0),(1,0),(1,1),(1,2),(2,2)],
    "T5": [(0,0),(1,0),(2,0),(1,1),(1,2)],
    "Y5": [(1,0),(0,1),(1,1),(1,2),(1,3)],
}
FILLERS = {  # generous-stock helper shapes that pack well together
    "O4": [(0,0),(1,0),(0,1),(1,1)],
    "I3": [(0,0),(1,0),(2,0)],
    "L3": [(0,0),(0,1),(1,1)],
    "T4": [(0,0),(1,0),(2,0),(1,1)],
    "S4": [(0,0),(1,0),(1,1),(2,1)],
    "L4": [(0,0),(0,1),(0,2),(1,2)],
    "I4": [(0,0),(1,0),(2,0),(3,0)],
    "P5": [(0,0),(1,0),(0,1),(1,1),(0,2)],
}

def main():
    i = int(sys.argv[1])
    rng = random.Random(43000 + 101 * i)

    # difficulty ladder: board grows 6 -> 12, more types, more bedrock
    W = H = min(12, 6 + (i - 1) * 3 // 4)  # 6,6,7,8,9,9,10,11,12,12
    P = 3 + (i - 1) % 4                    # 3..6 cycling
    total = W * H
    r_frac = 0.02 + 0.013 * ((i - 1) % 5)  # 0.02 .. 0.072
    R = int(round(r_frac * total))

    # bedrock: keep the top-left 3x3 clear so the baseline can always place a copy
    all_cells = [(x, y) for y in range(H) for x in range(W)]
    pool = [(x, y) for (x, y) in all_cells if not (x < 3 and y < 3)]
    R = min(R, len(pool))
    bedrock = sorted(rng.sample(pool, R))

    free = total - R

    # panel types
    a_keys = list(AWKWARD.keys())
    f_keys = list(FILLERS.keys())
    t0_shape = AWKWARD[a_keys[i % len(a_keys)]]
    types = [t0_shape]
    chosen_f = rng.sample(f_keys, P - 1)
    for k in chosen_f:
        types.append(FILLERS[k])

    # stock: type 0 small (drives the baseline B), others generous (overfill)
    stocks = []
    s0 = max(2, int(round(0.15 * free / 5.0)))
    stocks.append(s0)
    for t in range(1, P):
        sz = len(types[t])
        stocks.append(free // sz + 1)

    out = []
    out.append("%d %d" % (W, H))
    out.append(str(R))
    for (x, y) in bedrock:
        out.append("%d %d" % (x, y))
    out.append(str(P))
    for t in range(P):
        shp = types[t]
        out.append("%d %d" % (stocks[t], len(shp)))
        for (dx, dy) in shp:
            out.append("%d %d" % (dx, dy))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
