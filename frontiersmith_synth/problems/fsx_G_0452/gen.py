import sys, random

def main():
    t = int(sys.argv[1])
    random.seed(1000 + t)
    size = 3 + t                     # t=1..10  ->  4..13
    n = size
    m = size
    cmax_list = [3, 3, 7, 7, 15, 15, 31, 31, 63, 63]
    cmax = cmax_list[t - 1] if 1 <= t <= 10 else 63

    rows = []
    for i in range(m):
        row = [0] * n
        for j in range(n):
            if random.random() < 0.6:
                row[j] = random.randint(-cmax, cmax)
        # guarantee at least one strictly positive coefficient
        if not any(v > 0 for v in row):
            j = random.randrange(n)
            row[j] = random.randint(1, cmax)
        # guarantee at least two nonzero coefficients (real work; not a unit row)
        if sum(1 for v in row if v != 0) < 2:
            for j in range(n):
                if row[j] == 0:
                    row[j] = random.randint(1, cmax)
                    break
        rows.append(row)

    out = ["%d %d" % (m, n)]
    for row in rows:
        out.append(" ".join(str(v) for v in row))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
