import sys, random


def main():
    t = int(sys.argv[1])
    # difficulty ladder: (n1, n2, n3, planted_rank). n1 largest, n3 smallest.
    # planted_rank > max(dims)  => OVERCOMPLETE => true tensor rank is genuinely
    # unknown (Jennrich / simultaneous diagonalization only recover rank <= dim).
    shapes = [
        (6, 5, 2, 8),
        (7, 5, 2, 9),
        (7, 6, 3, 9),
        (8, 6, 3, 10),
        (8, 7, 3, 10),
        (9, 7, 3, 11),
        (9, 8, 3, 11),
        (10, 7, 2, 12),
        (10, 8, 3, 12),
        (11, 8, 3, 13),
    ]
    idx = (t - 1) % len(shapes)
    n1, n2, n3, Rp = shapes[idx]
    rnd = random.Random(1000 + t)
    T = [[[0] * n3 for _ in range(n2)] for _ in range(n1)]
    choices = [-3, -2, -1, 1, 2, 3]
    for _ in range(Rp):
        u = [rnd.choice(choices) for _ in range(n1)]
        v = [rnd.choice(choices) for _ in range(n2)]
        w = [rnd.choice(choices) for _ in range(n3)]
        for i in range(n1):
            ui = u[i]
            for j in range(n2):
                uv = ui * v[j]
                Tij = T[i][j]
                for k in range(n3):
                    Tij[k] += uv * w[k]
    out = ["%d %d %d" % (n1, n2, n3)]
    for k in range(n3):
        for i in range(n1):
            out.append(" ".join(str(T[i][j][k]) for j in range(n2)))
    sys.stdout.write("\n".join(out) + "\n")


main()
