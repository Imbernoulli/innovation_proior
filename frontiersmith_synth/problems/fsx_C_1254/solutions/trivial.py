# TIER: trivial
"""
Put every block into ONE domain and set its threshold above any possible idle-run length
(1_000_000 >> T always), i.e. NEVER gate. This domain is therefore held ON for the entire
horizon on every trace -- exactly the checker's own reference baseline construction, so this
reproduces B (the trivial ~0.1 reference point) exactly.
"""
import sys


def main():
    data = sys.stdin.read().split()
    p = iter(data)

    def nx():
        return next(p)

    N = int(nx())
    D = int(nx())
    K = int(nx())
    T = int(nx())
    nx(); nx()  # L, W (unused)
    for _ in range(K * N):
        nx()  # consume trace rows

    out = ["1"]
    out.append(" ".join("1" for _ in range(N)))
    out.append("1000000")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
