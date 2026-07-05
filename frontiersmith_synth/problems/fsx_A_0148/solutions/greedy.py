# TIER: greedy
"""Best-axis slab decomposition: slab along whichever mode makes the two
remaining dimensions' product smallest.  Rank = min(a*b, a*c, b*c)."""
import sys


def main():
    toks = sys.stdin.read().split()
    a, b, c = int(toks[0]), int(toks[1]), int(toks[2])
    body = list(map(int, toks[3:3 + a * b * c]))
    T = [[[0] * c for _ in range(b)] for _ in range(a)]
    idx = 0
    for i in range(a):
        for j in range(b):
            for k in range(c):
                T[i][j][k] = body[idx]; idx += 1

    # counts: slab mode k -> a*b ; slab mode j -> a*c ; slab mode i -> b*c
    opt = min((a * b, 3), (a * c, 2), (b * c, 1))
    mode = opt[1]
    lines = []
    if mode == 3:                      # slab along k, fiber over k
        for i in range(a):
            for j in range(b):
                u = [0] * a; u[i] = 1
                v = [0] * b; v[j] = 1
                w = [T[i][j][k] for k in range(c)]
                lines.append(u + v + w)
    elif mode == 2:                    # slab along j, fiber over j
        for i in range(a):
            for k in range(c):
                u = [0] * a; u[i] = 1
                v = [T[i][j][k] for j in range(b)]
                w = [0] * c; w[k] = 1
                lines.append(u + v + w)
    else:                              # slab along i, fiber over i
        for j in range(b):
            for k in range(c):
                u = [T[i][j][k] for i in range(a)]
                v = [0] * b; v[j] = 1
                w = [0] * c; w[k] = 1
                lines.append(u + v + w)

    out = [str(len(lines))] + [" ".join(map(str, row)) for row in lines]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
