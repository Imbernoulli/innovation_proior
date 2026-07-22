import sys, random

# Difficulty ladder: an R x C grid nesting layout of unit-ish rectangular parts.
# Each interior segment of the grid is a cut edge SHARED by two adjacent parts
# (must be cut exactly once); each boundary segment belongs to one part.
# testId encodes (R, C, spacing jitter, pierce cost P).
PLAN = {
    1:  (2, 2, False, 40),
    2:  (2, 4, False, 40),
    3:  (3, 3, False, 45),
    4:  (3, 5, True,  45),
    5:  (4, 4, True,  50),
    6:  (1, 10, True, 35),   # thin strip: fewer T-junctions -> smaller (but still real) savings
    7:  (5, 5, True,  55),
    8:  (5, 6, True,  58),
    9:  (6, 6, True,  60),
    10: (6, 7, True,  62),
}

def main():
    i = int(sys.argv[1])
    R, C, jitter, P = PLAN[i]
    rng = random.Random(19260817 + 97 * i)

    # cumulative integer coordinates along each axis
    def axis(n):
        vals = [0]
        for _ in range(n):
            step = rng.randint(8, 16) if jitter else 10
            vals.append(vals[-1] + step)
        return vals

    xs = axis(C)   # C+1 values
    ys = axis(R)   # R+1 values

    def vid(r, c):
        return r * (C + 1) + c + 1   # 1-indexed

    n = (R + 1) * (C + 1)

    # horizontal edges: row r in 0..R, col c in 0..C-1  -- index = r*C + c + 1
    # vertical edges:   row r in 0..R-1, col c in 0..C  -- index = H + r*(C+1) + c + 1
    H = (R + 1) * C
    def hidx(r, c):
        return r * C + c + 1
    def vidx(r, c):
        return H + r * (C + 1) + c + 1

    edges = [None] * (H + R * (C + 1))
    for r in range(R + 1):
        for c in range(C):
            edges[hidx(r, c) - 1] = (vid(r, c), vid(r, c + 1))
    for r in range(R):
        for c in range(C + 1):
            edges[vidx(r, c) - 1] = (vid(r, c), vid(r + 1, c))
    m = len(edges)

    parts = []
    for r in range(R):
        for c in range(C):
            parts.append((hidx(r, c), hidx(r + 1, c), vidx(r, c), vidx(r, c + 1)))

    out = []
    out.append(f"{n} {m} {P}")
    for r in range(R + 1):
        for c in range(C + 1):
            out.append(f"{xs[c]} {ys[r]}")
    for (u, v) in edges:
        out.append(f"{u} {v}")
    out.append(str(len(parts)))
    for (a, b, c, d) in parts:
        out.append(f"{a} {b} {c} {d}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
