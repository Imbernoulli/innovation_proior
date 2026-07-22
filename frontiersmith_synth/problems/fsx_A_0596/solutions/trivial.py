# TIER: trivial
# Baseline construction: attach every feature to its first-listed allowed datum
# (the deep-trunk option) and buy zero precise slots. Reproduces the checker's
# internal baseline, so it scores ~0.1.
import sys

def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    n = int(next(it)); k = int(next(it))
    a = [int(next(it)) for _ in range(n)]
    p = [int(next(it)) for _ in range(n)]
    allowed = []
    for i in range(n):
        d = int(next(it)); allowed.append([int(next(it)) for _ in range(d)])
    par = [-1] + [allowed[i][0] for i in range(1, n)]
    print(" ".join(map(str, par)))
    print("0")

main()
