# TIER: strong
"""Monetary-policy insight: money that leaves the active bidders' hands (paid to
losers each round) never returns without an explicit recirculation mechanism, so any
finite up-front endowment eventually runs dry over a long horizon no matter how it is
allocated. Instead, start from a neutral equal split and run a steady demurrage tax
on every wallet each round, then refill the collected pool back out weighted by THAT
round's own disutility-relief values -- so scrip flows continuously back toward
whoever needs it most *this week*, not whoever needed it most on average. This keeps
market velocity high across the whole horizon (including regime changes in who is
'active') instead of letting scrip go idle in low-demand nurses' wallets."""
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

    E = largest_remainder([1] * N, S)          # neutral start
    TAX_RATE = TAX_DEN * 3 // 10                # ~30% demurrage every round
    T = [TAX_RATE] * R
    W = [[v[r][i] for i in range(N)] for r in range(R)]  # refill chases this round's need

    out = []
    out.append(" ".join(map(str, E)))
    out.append(" ".join(map(str, T)))
    for r in range(R):
        out.append(" ".join(map(str, W[r])))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
