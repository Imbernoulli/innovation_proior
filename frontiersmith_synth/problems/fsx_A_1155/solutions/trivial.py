# TIER: trivial
"""Equal scrip endowment, no tax, no refill. Reproduces the checker's own baseline."""
import sys


def largest_remainder(weights, total):
    n = len(weights)
    sw = sum(weights)
    if sw == 0:
        base = total // n
        rem = total - base * n
        out = [base] * n
        for i in range(rem):
            out[i] += 1
        return out
    alloc = [total * w // sw for w in weights]
    rem = total - sum(alloc)
    order = sorted(range(n), key=lambda i: (-weights[i], i))
    for k in range(rem):
        alloc[order[k]] += 1
    return alloc


def main():
    data = sys.stdin.read().split()
    pos = 0
    def nxt():
        nonlocal pos
        v = data[pos]; pos += 1; return v
    N = int(nxt()); R = int(nxt())
    ALPHA_NUM = int(nxt()); ALPHA_DEN = int(nxt()); S = int(nxt()); TAX_DEN = int(nxt())
    for _ in range(R):
        for _ in range(N):
            nxt()  # skip v matrix, unused

    E = largest_remainder([1] * N, S)
    T = [0] * R
    W = [[0] * N for _ in range(R)]

    out = []
    out.append(" ".join(map(str, E)))
    out.append(" ".join(map(str, T)))
    for r in range(R):
        out.append(" ".join(map(str, W[r])))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
