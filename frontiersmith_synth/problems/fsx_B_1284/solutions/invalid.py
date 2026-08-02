# TIER: invalid
# Ignores the exclusion constraint entirely (puts weight back on excluded names) and
# overshoots capacity caps -> fails feasibility -> scores 0.
import sys


def main():
    toks = sys.stdin.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = toks[p]
        p += 1
        return v

    N = int(nxt())
    S = int(nxt())
    K = S + 1
    nxt()  # T
    for _ in range(K * K):
        nxt()
    for _ in range(N):
        for _ in range(6):
            nxt()

    out = ["%.10f" % (1.0 / N) for _ in range(N)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
