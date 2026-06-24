import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)
    n = random.randint(1, 9)
    q = random.randint(1, 14)
    # Small value domain to force ties / equal heights, where a strict-vs-non-
    # strict prefix-max bug and the boundary double-count surface.
    vmax = random.choice([2, 2, 3, 4, 6, 12])
    lines = []
    lines.append(f"{n} {q}")
    h = [random.randint(1, vmax) for _ in range(n)]
    lines.append(" ".join(map(str, h)))
    for _ in range(q):
        t = random.randint(1, 2)
        if t == 1:
            p = random.randint(0, n - 1)
            x = random.randint(1, vmax)
            lines.append(f"1 {p} {x}")
        else:
            l = random.randint(0, n - 1)
            r = random.randint(l, n - 1)
            lines.append(f"2 {l} {r}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
