import sys, random

# Wider stress: larger n/q, sometimes full value range (to exercise 64-bit and
# many-distinct cases), still small enough for the O(n*q) brute.
def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)
    n = random.randint(1, 50)
    q = random.randint(1, 50)
    vmax = random.choice([2, 3, 7, 50, 1000000000])
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
