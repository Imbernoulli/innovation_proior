import sys, random

# larger / wider stress generator
def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)
    n = random.randint(1, 40)
    q = random.randint(1, 40)
    vmax = random.choice([2, 3, 5, 20, 1000000000])
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
