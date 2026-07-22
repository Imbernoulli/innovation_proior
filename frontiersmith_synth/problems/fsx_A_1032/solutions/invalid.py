# TIER: invalid
"""
Deliberately infeasible: every dish declares the SAME kitchen R times (fails
the "R distinct replica kitchens" check on dish 0 already), so the checker
must reject with Ratio: 0.0.
"""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0

    def nxt():
        nonlocal idx
        v = data[idx]
        idx += 1
        return v

    K = int(nxt()); R = int(nxt()); D = int(nxt()); B = int(nxt())
    _cap = [int(nxt()) for _ in range(K)]
    courses = []
    for _ in range(B):
        s = int(nxt())
        dishes = [int(nxt()) for _ in range(s)]
        courses.append(dishes)

    out = []
    for _d in range(D):
        out.append(" ".join(["0"] * R))  # duplicate kitchen ids -> infeasible
    for dishes in courses:
        out.append(" ".join(["0"] * len(dishes)))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
