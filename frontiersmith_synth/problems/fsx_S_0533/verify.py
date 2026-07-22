#!/usr/bin/env python3
# verify.py <in> <out> <ans>
#
# Scores a candidate synchronizing word (a "reset word") for the DFA in <in>.
# The participant writes the word to stdout as whitespace-separated symbol indices.
#
# Feasibility (ANY violation -> Ratio: 0.0):
#   * every token is an integer in [0, m-1]
#   * word length L in [1, MAXLEN]
#   * simulating the word from the set of ALL states collapses it to exactly ONE state
#
# Objective (minimize):  F = length of the submitted reset word.
# Internal baseline B (deterministic, built by the checker):  a VALID reset word from the
#   textbook rotate-and-contract schema WITHOUT minimising the rotations (it walks the cycle
#   several extra full turns between contractions).  Because R is a permutation, R^n acts as
#   the identity on the state set, so padding with whole turns keeps the word valid while
#   making it deliberately long.  If the planted algebra cannot be recovered, B falls back to
#   a lexicographic pairwise-merge word (also valid).
#
#   ratio = min(1.0, 0.1 * B / F)      (fewer symbols -> higher ratio; trivial baseline ~ 0.1)
from collections import deque, Counter
import sys

MAXLEN = 200000

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def read_ints(path):
    with open(path) as f:
        data = f.read().split()
    return data

def apply_word(delta, word, S):
    cur = set(S)
    for s in word:
        cur = set(delta[s][i] for i in cur)
    return cur

def all_pairs(delta, m, n):
    # backward BFS from the diagonal: shortest merging word for every pair
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
    if p == q:
        return []
    key = (p, q) if p < q else (q, p)
    w = []
    while key[0] != key[1]:
        s = nsym[key]; w.append(s); key = npair[key]
    return w

def lex_reset(delta, m, n, dist, nsym, npair):
    S = set(range(n)); word = []
    while len(S) > 1:
        Sl = sorted(S)
        w = merge_word(nsym, npair, Sl[0], Sl[1])
        if not w:
            # Sl[0],Sl[1] cannot be merged directly; find any mergeable active pair
            found = None
            for i in range(len(Sl)):
                for jj in range(i + 1, len(Sl)):
                    k = (Sl[i], Sl[jj])
                    if dist.get(k, None):
                        found = k; break
                if found:
                    break
            if not found:
                return None
            w = merge_word(nsym, npair, found[0], found[1])
        word += w; S = apply_word(delta, w, S)
    return word

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
        y = fp[0]; x = mv[0]
        if delta[Rs][x] == y:
            return Rs, Cs
    return None

def baseline(delta, m, n):
    rc = find_RC(delta, m, n)
    if rc is not None:
        Rs, Cs = rc
        # padded rotate-and-contract: full extra turns between contractions
        word = [Cs] + ([Rs] * (3 * n - 1) + [Cs]) * (n - 2)
        if len(apply_word(delta, word, range(n))) == 1:
            return len(word)
    # fallback: lexicographic pairwise-merge word
    dist, nsym, npair = all_pairs(delta, m, n)
    w = lex_reset(delta, m, n, dist, nsym, npair)
    if w is None:
        return max(1, n * n)
    return max(1, len(w))

def main():
    inp = read_ints(sys.argv[1])
    it = iter(inp)
    try:
        n = int(next(it)); m = int(next(it))
    except Exception:
        fail("bad header")
    if n < 2 or m < 1 or n > 100000 or m > 1000:
        fail("bad n/m")
    delta = []
    try:
        for _ in range(m):
            row = [int(next(it)) for _ in range(n)]
            for v in row:
                if v < 0 or v >= n:
                    fail("bad transition")
            delta.append(row)
    except StopIteration:
        fail("truncated automaton")
    except Exception:
        fail("bad automaton token")

    # ---- participant word ----
    out_tokens = read_ints(sys.argv[2])
    if len(out_tokens) == 0:
        fail("empty output")
    if len(out_tokens) > MAXLEN:
        fail("word too long")
    word = []
    for t in out_tokens:
        try:
            s = int(t)
        except Exception:
            fail("non-integer symbol %r" % t)
        if s < 0 or s >= m:
            fail("symbol out of range %d" % s)
        word.append(s)

    # ---- feasibility: must collapse ALL states to one ----
    final = apply_word(delta, word, range(n))
    if len(final) != 1:
        fail("not a reset word (|image|=%d)" % len(final))

    F = len(word)
    B = baseline(delta, m, n)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
