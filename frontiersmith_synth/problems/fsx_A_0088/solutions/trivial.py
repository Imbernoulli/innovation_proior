# TIER: trivial
# One combiner stage per nonzero gain entry: rank = B (the checker's baseline).
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    a = int(next(it)); b = int(next(it)); c = int(next(it))
    G = [[[0] * c for _ in range(b)] for _ in range(a)]
    for i in range(a):
        for j in range(b):
            for k in range(c):
                G[i][j][k] = int(next(it))

    stages = []
    for i in range(a):
        for j in range(b):
            for k in range(c):
                g = G[i][j][k]
                if g == 0:
                    continue
                u = [0] * a; u[i] = g
                v = [0] * b; v[j] = 1
                w = [0] * c; w[k] = 1
                stages.append((u, v, w))

    outp = [str(len(stages))]
    for (u, v, w) in stages:
        outp.append(" ".join(map(str, u + v + w)))
    sys.stdout.write("\n".join(outp) + "\n")

if __name__ == "__main__":
    main()
