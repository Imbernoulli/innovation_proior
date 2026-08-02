# TIER: greedy
"""
The "obvious" opponent-modeling idea: you have real data on how this opponent
has played historically (H, normalized to q). Best-respond to it -- weight
each row by how well it would have scored against that empirical distribution,
concentrating mass on whichever rows look best against q (a soft/logit best
response, rather than a hard single-row commitment, since that is the natural
"smoothed" refinement an average coder reaches for after noticing hard argmax
is fragile). This exploits the *sampled* opponent, not the true adversary who
will always best-respond to whatever you publish -- rows that merely look
good against q can be catastrophic against the column the log rarely used.
"""
import math
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = data[pos]
        pos += 1
        return v

    m = int(nxt())
    n = int(nxt())
    A = [[int(nxt()) for _ in range(n)] for _ in range(m)]
    N = int(nxt())
    H = [int(nxt()) for _ in range(n)]

    q = [h / N for h in H]
    val = [sum(q[j] * A[i][j] for j in range(n)) for i in range(m)]

    BETA = 0.002
    mx = max(val)
    w = [math.exp(BETA * (v - mx)) for v in val]
    s = sum(w)
    p = [x / s for x in w]

    print(" ".join(f"{x:.9f}" for x in p))


if __name__ == "__main__":
    main()
