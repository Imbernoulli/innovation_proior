import sys, random

# Random small-case generator for half-plane feasibility.
# Usage: python3 gen.py <seed>
# Prints: m, then m lines "a b c" meaning  a*x + b*y <= c, |a|,|b|,|c| <= CAP,
# with (a,b) != (0,0).

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    mode = rng.randint(0, 6)
    # Keep coefficients small to provoke degenerate/tangent configurations,
    # but stay within the problem cap 1e6.
    SMALL = 6

    def rnd_ab():
        while True:
            a = rng.randint(-SMALL, SMALL)
            b = rng.randint(-SMALL, SMALL)
            if a != 0 or b != 0:
                return a, b

    planes = []

    if mode == 0:
        # Random constraints (often infeasible).
        m = rng.randint(1, 12)
        for _ in range(m):
            a, b = rnd_ab()
            c = rng.randint(-SMALL, SMALL)
            planes.append((a, b, c))
    elif mode == 1:
        # Feasible by construction: pick a center point, all constraints satisfied
        # there (with slack), then maybe add a couple random ones.
        cx = rng.randint(-4, 4)
        cy = rng.randint(-4, 4)
        m = rng.randint(1, 12)
        for _ in range(m):
            a, b = rnd_ab()
            base = a * cx + b * cy
            c = base + rng.randint(0, SMALL)  # center satisfies a*cx+b*cy <= c
            planes.append((a, b, c))
    elif mode == 2:
        # Parallel / opposite pairs to test tangency and emptiness.
        m = rng.randint(1, 6)
        for _ in range(m):
            a, b = rnd_ab()
            c = rng.randint(-SMALL, SMALL)
            planes.append((a, b, c))
            # opposite direction
            c2 = rng.randint(-SMALL, SMALL)
            planes.append((-a, -b, c2))
    elif mode == 3:
        # Tight box around a point: tangent / single-point feasibility.
        cx = rng.randint(-3, 3)
        cy = rng.randint(-3, 3)
        planes = [(1, 0, cx), (-1, 0, -cx), (0, 1, cy), (0, -1, -cy)]
        # exactly the single point (cx,cy); maybe perturb one to break it
        if rng.random() < 0.4:
            i = rng.randrange(4)
            a, b, c = planes[i]
            planes[i] = (a, b, c - 1)  # may make it empty
    elif mode == 4:
        # Lines through origin in many directions -> intersection is origin or empty.
        m = rng.randint(2, 10)
        for _ in range(m):
            a, b = rnd_ab()
            c = rng.randint(-1, 1)
            planes.append((a, b, c))
    elif mode == 5:
        # Single half-plane (always feasible) or a few same-direction ones.
        a, b = rnd_ab()
        k = rng.randint(1, 5)
        for _ in range(k):
            c = rng.randint(-SMALL, SMALL)
            planes.append((a, b, c))
    else:
        # Larger random mix.
        m = rng.randint(1, 20)
        for _ in range(m):
            a, b = rnd_ab()
            c = rng.randint(-SMALL, SMALL)
            planes.append((a, b, c))

    rng.shuffle(planes)
    out = [str(len(planes))]
    for (a, b, c) in planes:
        out.append(f"{a} {b} {c}")
    sys.stdout.write("\n".join(out) + "\n")

main()
