# TIER: trivial
"""Naive id-order first-fit: no ranking heuristic at all -- walk candidates 0..C-1 and
take each one that still fits under both the encoding-space (K) and area (A) budgets.
This is exactly the checker's own baseline construction, so it should land near the
calibrated ~0.1 floor."""
import sys


def main():
    data = sys.stdin.read().split("\n")
    idx = 0
    K, A = map(int, data[idx].split()); idx += 1
    C = int(data[idx]); idx += 1
    area = [0] * C
    for c in range(C):
        a, s, cst = map(int, data[idx].split()); idx += 1
        area[c] = a
    # M / app body irrelevant to this construction

    sel = []
    used_area = 0
    for c in range(C):
        if len(sel) >= K:
            break
        if used_area + area[c] <= A:
            sel.append(c)
            used_area += area[c]

    out = [str(len(sel)), " ".join(map(str, sel))]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
