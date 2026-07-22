# TIER: trivial
import sys


def main():
    it = iter(sys.stdin.read().split())
    N = int(next(it)); T = int(next(it))
    caps = []; ms = []; as_ = []; bs = []; fast = []
    for _ in range(N):
        caps.append(int(next(it)))
        ms.append(float(next(it)))
        as_.append(float(next(it)))
        bs.append(float(next(it)))
        fast.append(int(next(it)))
    J = int(next(it)) - 1
    D = [float(next(it)) for _ in range(T)]

    rest = sum(caps[i] for i in range(N) if i != J)
    out_lines = []
    for t in range(T):
        p = [0.0] * N
        for i in range(N):
            p[i] = 0.0 if i == J else D[t] * caps[i] / rest
        out_lines.append(" ".join("%.6f" % x for x in p))
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
