# TIER: strong
# INSIGHT: don't merge pairs one-by-one -- read the automaton's hidden group/contraction
# algebra.  Recover the full n-cycle permutation R and the single contraction C (which pulls
# state x onto its R-successor y).  Up to relabelling this is exactly the Cerny automaton, so
# the Cerny reset word
#         C (R^{n-1} C)^{n-2}          of length (n-1)^2
# collapses all n states -- near-quadratic, vs. the pairwise heuristic's much longer word.
# We verify by simulation and only emit if it truly resets (else fall back to pairwise merge).
from collections import deque, Counter
import sys

def apply_word(delta, word, S):
    cur = set(S)
    for s in word:
        cur = set(delta[s][i] for i in cur)
    return cur

def n_cycles(delta, m, n):
    res = []
    for s in range(m):
        if len(set(delta[s])) != n:
            continue
        cur = 0; length = 0; vis = set()
        while cur not in vis:
            vis.add(cur); cur = delta[s][cur]; length += 1
        if length == n:
            res.append(s)
    return res

def contractions(delta, m, n):
    res = []
    for s in range(m):
        if len(set(delta[s])) != n - 1:
            continue
        cnt = Counter(delta[s])
        merged = [v for v, c in cnt.items() if c == 2]
        if len(merged) != 1:
            continue
        preimg = [i for i in range(n) if delta[s][i] == merged[0]]
        fp = [i for i in preimg if delta[s][i] == i]
        mv = [i for i in preimg if delta[s][i] != i]
        if len(fp) == 1 and len(mv) == 1:
            res.append((s, mv[0], fp[0]))   # (symbol, x, y)  with C(x)=C(y)=y
    return res

def cerny_word(delta, m, n):
    for Rs in n_cycles(delta, m, n):
        for (Cs, x, y) in contractions(delta, m, n):
            if delta[Rs][x] != y:
                continue
            word = [Cs] + ([Rs] * (n - 1) + [Cs]) * (n - 2)
            if len(apply_word(delta, word, range(n))) == 1:
                return word
    return None

# ---- fallback: pairwise merge (only used if the algebra is not the clean Cerny form) ----
def all_pairs(delta, m, n):
    dist = {}; nsym = {}; npair = {}
    dq = deque()
    for c in range(n):
        dist[(c, c)] = 0; dq.append((c, c))
    pre = [[[] for _ in range(n)] for _ in range(m)]
    for s in range(m):
        for x in range(n):
            pre[s][delta[s][x]].append(x)
    while dq:
        u = dq.popleft(); ua, ub = u
        for s in range(m):
            for x in pre[s][ua]:
                for y in pre[s][ub]:
                    key = (x, y) if x < y else (y, x)
                    if key not in dist:
                        dist[key] = dist[u] + 1; nsym[key] = s; npair[key] = u
                        dq.append(key)
    return dist, nsym, npair

def merge_word(nsym, npair, p, q):
    key = (p, q) if p < q else (q, p); w = []
    while key[0] != key[1]:
        s = nsym[key]; w.append(s); key = npair[key]
    return w

def pairwise(delta, m, n):
    dist, nsym, npair = all_pairs(delta, m, n)
    S = set(range(n)); word = []
    while len(S) > 1:
        Sl = sorted(S); sel = None; selv = None
        for i in range(len(Sl)):
            for j in range(i + 1, len(Sl)):
                k = (Sl[i], Sl[j]); d = dist.get(k)
                if d is not None and (selv is None or d < selv):
                    selv = d; sel = k
        w = merge_word(nsym, npair, sel[0], sel[1]); word += w
        S = apply_word(delta, w, S)
    return word

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it))
    delta = [[int(next(it)) for _ in range(n)] for _ in range(m)]
    word = cerny_word(delta, m, n)
    if word is None:
        word = pairwise(delta, m, n)
    sys.stdout.write(" ".join(map(str, word)) + "\n")

if __name__ == "__main__":
    main()
