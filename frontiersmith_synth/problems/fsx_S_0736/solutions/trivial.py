# TIER: trivial
"""Always take candidate 0 in every stall (reproduces the checker's own baseline B)."""
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = data[pos]; pos += 1
        return v

    N = int(nxt()); R = int(nxt())
    for _ in range(R):
        nxt()
    out = []
    for _i in range(N):
        M = int(nxt())
        for _k in range(M):
            cost = int(nxt())
            L = int(nxt())
            for _c in range(L):
                nxt()
        out.append("0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
