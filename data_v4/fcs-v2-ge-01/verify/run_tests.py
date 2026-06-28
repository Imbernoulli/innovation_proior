#!/usr/bin/env python3
import subprocess, sys, random

SOL = "/tmp/fcs-v2-ge-01_x"

def brute(inp):
    data = inp.split()
    it = iter(data)
    n = int(next(it))
    pts = []
    for _ in range(n):
        x = int(next(it)); y = int(next(it))
        pts.append((x, y))
    best = 0
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            ax, ay = pts[j][0] - xi, pts[j][1] - yi
            for k in range(j + 1, n):
                bx, by = pts[k][0] - xi, pts[k][1] - yi
                a = abs(ax * by - ay * bx)
                if a > best:
                    best = a
    return str(best)

def gen(seed):
    rng = random.Random(seed)
    regime = rng.randint(0, 9)
    if regime == 0:
        n = rng.randint(0, 2)
    elif regime == 1:
        n = rng.randint(3, 8)
    else:
        n = rng.randint(3, 40)
    if regime in (2, 3):
        C = rng.choice([2, 3, 4])
    else:
        C = rng.choice([5, 10, 30, 100])
    pts = []
    if regime == 4 and n >= 1:
        dx = rng.randint(-3, 3); dy = rng.randint(-3, 3)
        if dx == 0 and dy == 0: dx = 1
        for _ in range(n):
            t = rng.randint(-C, C); pts.append((t*dx, t*dy))
    elif regime == 5 and n >= 1:
        pool = [(rng.randint(-C,C), rng.randint(-C,C)) for _ in range(max(1, n//3))]
        for _ in range(n):
            pts.append(rng.choice(pool))
    else:
        for _ in range(n):
            pts.append((rng.randint(-C,C), rng.randint(-C,C)))
    lines = [str(n)] + [f"{x} {y}" for (x,y) in pts]
    return "\n".join(lines) + "\n"

def run_sol(inp):
    r = subprocess.run([SOL], input=inp, capture_output=True, text=True)
    return r.stdout.strip()

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    mism = 0
    for s in range(1, N + 1):
        inp = gen(s)
        a = run_sol(inp)
        b = brute(inp)
        if a != b:
            print(f"MISMATCH seed={s} sol={a!r} brute={b!r}")
            print(inp)
            mism += 1
            if mism >= 8:
                break
    # explicit edge cases
    edges = [
        "0\n",
        "1\n3 4\n",
        "2\n0 0\n5 5\n",
        "3\n0 0\n1 0\n2 0\n",                 # collinear
        "3\n0 0\n4 0\n0 3\n",                  # simple right triangle, area2=12
        "4\n0 0\n4 0\n4 4\n0 4\n",             # square, area2=16
        "6\n5 0\n1 0\n1 0\n5 0\n1 0\n5 0\n",   # duplicates collinear
        "5\n0 0\n10 0\n10 10\n0 10\n5 5\n",    # interior point present
        "3\n1000000000 1000000000\n-1000000000 1000000000\n0 -1000000000\n", # large coords
        "4\n-1000000000 -1000000000\n1000000000 -1000000000\n1000000000 1000000000\n-1000000000 1000000000\n",
    ]
    for e in edges:
        a = run_sol(e); b = brute(e)
        tag = "OK" if a == b else "MISMATCH"
        if a != b:
            mism += 1
            print(f"EDGE {tag} sol={a!r} brute={b!r} :: {e!r}")
        else:
            print(f"EDGE OK sol={a} :: {e.splitlines()[0]}...")
    print(f"TOTAL MISMATCHES (random+edge): {mism} over {N} random + {len(edges)} edge")

if __name__ == "__main__":
    main()
