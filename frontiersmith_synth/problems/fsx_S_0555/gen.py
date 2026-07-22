import sys, random

def satisfies(x, r, c):
    return len(x) >= r and x[len(x)-r] == c

def member(x, pats):
    return any(satisfies(x, r, c) for (r, c) in pats)

def satisfied_set(x, pats):
    return frozenset(i for i, (r, c) in enumerate(pats) if satisfies(x, r, c))

def trie_size(P):
    pref = set()
    for s in P:
        for i in range(len(s) + 1):
            pref.add(s[:i])
    return len(pref)

def strong_nfa(P, N, alphabet):
    Plist = list(P); Nset = list(N)
    maxlen = max((len(x) for x in P + N), default=1)
    usable = []
    for r in range(1, maxlen + 1):
        for c in alphabet:
            if any(satisfies(nx, r, c) for nx in Nset):
                continue
            cov = frozenset(i for i, p in enumerate(Plist) if satisfies(p, r, c))
            if cov:
                usable.append((r, c, cov))
    U = set(range(len(Plist)))
    for maxR in range(1, maxlen + 1):
        pats = [u for u in usable if u[0] <= maxR]
        covered = set(); ncov = 0
        while covered != U:
            bp = None; bg = 0
            for u in pats:
                g = len(u[2] - covered)
                if g > bg:
                    bg = g; bp = u
            if bp is None:
                break
            ncov += 1; covered |= bp[2]
        if covered == U:
            return 1 + maxR, ncov
    return None, None

RTABLE = [3, 4, 4, 5, 5, 6, 6, 7, 7, 8]

def build(testId):
    rng = random.Random(1000 + testId * 7919)
    alphabet = "01"
    R = RTABLE[testId - 1]
    npat = 2 + (testId % 2)
    offs = set([R])
    while len(offs) < npat:
        offs.add(rng.randint(1, R))
    pats = [(r, rng.choice(alphabet)) for r in sorted(offs)]
    Lmin = R; Lmax = R + 4
    members = []; nonmembers = []; seen = set(); tries = 0
    while tries < 60000 and (len(members) < 800 or len(nonmembers) < 500):
        tries += 1
        L = rng.randint(Lmin, Lmax)
        x = ''.join(rng.choice(alphabet) for _ in range(L))
        if x in seen:
            continue
        seen.add(x)
        (members if member(x, pats) else nonmembers).append(x)
    rng.shuffle(members); rng.shuffle(nonmembers)
    nN = 8 + R
    N = nonmembers[:nN]
    excl = {i: [] for i in range(npat)}; rest = []
    for x in members:
        s = satisfied_set(x, pats)
        if len(s) == 1:
            excl[next(iter(s))].append(x)
        else:
            rest.append(x)
    P = []
    for i in range(npat):
        P += excl[i][:2]
    pool = rest + [x for i in range(npat) for x in excl[i][2:]]
    rng.shuffle(pool)
    pool.sort(key=len)
    j = 0
    while j < len(pool):
        nfa, ncov = strong_nfa(P, N, alphabet)
        b = trie_size(P)
        if nfa and b >= 8.2 * nfa and len(P) >= 7 and ncov >= 2:
            break
        P.append(pool[j]); j += 1
    while len(P) > 7:
        nfa, ncov = strong_nfa(P, N, alphabet); b = trie_size(P)
        if b <= 8.4 * nfa:
            break
        cand = P[:-1]
        nfa2, ncov2 = strong_nfa(cand, N, alphabet)
        if nfa2 and ncov2 >= 2 and trie_size(cand) >= 8.0 * nfa2:
            P = cand
        else:
            break
    # stable order for determinism
    P = sorted(set(P)); N = sorted(set(N))
    return alphabet, P, N

def main():
    testId = int(sys.argv[1])
    alphabet, P, N = build(testId)
    out = []
    out.append(alphabet)
    out.append("%d %d" % (len(P), len(N)))
    out.extend(P)
    out.extend(N)
    sys.stdout.write("\n".join(out) + "\n")

main()
