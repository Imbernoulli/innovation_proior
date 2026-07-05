# TIER: trivial
# Naive: one interaction channel per nonzero tensor entry (reproduces the
# checker baseline B = number of nonzeros -> ratio ~ 0.1).
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    a = int(next(it)); b = int(next(it)); c = int(next(it))
    T = [[[0] * c for _ in range(b)] for _ in range(a)]
    for k in range(c):
        for i in range(a):
            for j in range(b):
                T[i][j][k] = int(next(it))
    channels = []
    for i in range(a):
        for j in range(b):
            for k in range(c):
                t = T[i][j][k]
                if t != 0:
                    u = [0] * a; u[i] = 1
                    v = [0] * b; v[j] = 1
                    w = [0] * c; w[k] = t
                    channels.append(u + v + w)
    outp = [str(len(channels))]
    for ch in channels:
        outp.append(" ".join(str(x) for x in ch))
    sys.stdout.write("\n".join(outp) + "\n")

if __name__ == "__main__":
    main()
