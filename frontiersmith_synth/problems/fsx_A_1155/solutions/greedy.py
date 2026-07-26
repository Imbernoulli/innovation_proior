# TIER: greedy
"""The obvious 'clever' fix: front-load scrip to nurses who need it most (endowment
proportional to their total disutility-relief demand across all rounds). No ongoing
monetary policy -- zero tax, zero refill. This is a one-round-reasoning heuristic:
it optimizes the INITIAL allocation only and ignores that money never returns to
active bidders once spent, so heavy bidders still run dry over a long horizon."""
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
    v = []
    for _r in range(R):
        v.append([int(nxt()) for _ in range(N)])

    demand = [sum(v[r][i] for r in range(R)) for i in range(N)]
    E = largest_remainder(demand, S)
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
