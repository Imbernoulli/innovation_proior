# TIER: invalid
# Emits a garbage roster: assigns every slot to staff 0, which double-books staff 0
# within a day and ignores skill requirements -> must be rejected (Ratio 0).
import sys

def main():
    toks = sys.stdin.read().split()
    it = iter(toks); nxt = lambda: next(it)
    S = int(nxt()); D = int(nxt()); C = int(nxt())
    for _ in range(S): nxt()          # maxshift
    for _ in range(S): nxt()          # base
    for _ in range(S):
        cnt = int(nxt())
        for _ in range(cnt): nxt()
    total = 0
    for _ in range(D):
        T = int(nxt())
        total += T
        for _ in range(T): nxt(); nxt()
    sys.stdout.write(" ".join(["0"] * total) + "\n")

main()
