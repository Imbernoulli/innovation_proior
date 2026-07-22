# TIER: trivial
# Rotate-and-contract, but sloppily: walk several EXTRA full turns of the cycle between
# contractions.  Valid (R^n acts as the identity on the state set) but deliberately long --
# this reproduces the checker's baseline, so it scores ~0.1.
from collections import Counter
import sys

def apply_word(delta, word, S):
    cur = set(S)
    for s in word:
        cur = set(delta[s][i] for i in cur)
    return cur

def find_RC(delta, m, n):
    cons = [s for s in range(m) if len(set(delta[s])) == n - 1]
    perms = [s for s in range(m) if len(set(delta[s])) == n]
    Rs = None
    for s in perms:
        cur = 0; length = 0; vis = set()
        while cur not in vis:
            vis.add(cur); cur = delta[s][cur]; length += 1
        if length == n:
            Rs = s; break
    if Rs is None:
        return None
    for Cs in cons:
        cnt = Counter(delta[Cs])
        merged = [v for v, c in cnt.items() if c == 2]
        if not merged:
            continue
        preimg = [i for i in range(n) if delta[Cs][i] == merged[0]]
        fp = [i for i in preimg if delta[Cs][i] == i]
        mv = [i for i in preimg if delta[Cs][i] != i]
        if not fp or not mv:
            continue
        if delta[Rs][mv[0]] == fp[0]:
            return Rs, Cs
    return None

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it))
    delta = [[int(next(it)) for _ in range(n)] for _ in range(m)]
    rc = find_RC(delta, m, n)
    if rc is None:
        # extremely defensive fallback: brute reset by repeated contraction+rotation attempts
        Cs = min(range(m), key=lambda s: len(set(delta[s])))
        Rs = max(range(m), key=lambda s: len(set(delta[s])))
    else:
        Rs, Cs = rc
    word = [Cs] + ([Rs] * (3 * n - 1) + [Cs]) * (n - 2)
    sys.stdout.write(" ".join(map(str, word)) + "\n")

if __name__ == "__main__":
    main()
