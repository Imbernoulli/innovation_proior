# TIER: greedy
import sys

def main():
    d = sys.stdin.buffer.read().split()
    idx = 0
    n = int(d[idx]); idx += 1
    m = int(d[idx]); idx += 1

    # per-variable weighted vote: how much satisfied weight prefers deep(+) vs shallow(-)
    pos_w = [0] * (n + 1)
    neg_w = [0] * (n + 1)
    for _ in range(m):
        w = int(d[idx]); k = int(d[idx + 1]); idx += 2
        for _j in range(k):
            l = int(d[idx]); idx += 1
            if l > 0:
                pos_w[l] += w
            else:
                neg_w[-l] += w

    # one-pass, interaction-free choice: dig deep iff the weight of clauses that
    # want this cell deep strictly exceeds the weight that wants it shallow.
    x = []
    for v in range(1, n + 1):
        x.append("1" if pos_w[v] > neg_w[v] else "0")

    sys.stdout.write(" ".join(x) + "\n")

main()
