# TIER: strong
import sys

def satisfies(x, r, c):
    return len(x) >= r and x[len(x) - r] == c

def emit(S, starts, acc, edges):
    out = [str(S)]
    out.append(str(len(starts)) + " " + " ".join(map(str, starts)))
    out.append(str(len(acc)) + " " + " ".join(map(str, acc)))
    out.append(str(len(edges)))
    for (f, sym, t) in edges:
        out.append("%d %s %d" % (f, sym, t))
    sys.stdout.write("\n".join(out) + "\n")

def trie_nfa(P, alphabet):
    ids = {"": 0}; edges = []; accept = set()
    for s in P:
        cur = ""
        for ch in s:
            nxt = cur + ch
            if nxt not in ids:
                ids[nxt] = len(ids); edges.append((ids[cur], ch, ids[nxt]))
            cur = nxt
        accept.add(ids[s])
    return len(ids), [0], sorted(accept), edges

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    alphabet = next(it)
    nP = int(next(it)); nN = int(next(it))
    P = [next(it) for _ in range(nP)]
    N = [next(it) for _ in range(nN)]

    # INSIGHT: an NFA state that guesses "the k-th symbol from the end is c"
    # is a COVER ELEMENT. A candidate pattern (r,c) is USABLE iff it fires on
    # no reject string (positive/negative separation); the accept strings are
    # the universe; a small set-cover by usable patterns -> few shared states.
    maxlen = max((len(x) for x in P + N), default=1)
    Plist = list(P)
    usable = []
    for r in range(1, maxlen + 1):
        for c in alphabet:
            if any(satisfies(nx, r, c) for nx in N):
                continue
            cov = frozenset(i for i, p in enumerate(Plist) if satisfies(p, r, c))
            if cov:
                usable.append((r, c, cov))

    U = set(range(len(Plist)))
    chosen = None; best_maxR = None
    for maxR in range(1, maxlen + 1):
        pats = [u for u in usable if u[0] <= maxR]
        covered = set(); pick = []
        while covered != U:
            bp = None; bg = 0
            for u in pats:
                g = len(u[2] - covered)
                if g > bg:
                    bg = g; bp = u
            if bp is None:
                break
            pick.append(bp); covered |= bp[2]
        if covered == U:
            chosen = pick; best_maxR = maxR
            break

    if chosen is None:
        # positional hypothesis infeasible for this sample: fall back to trie
        S, starts, acc, edges = trie_nfa(P, alphabet)
        emit(S, starts, acc, edges)
        return

    # Build the shared-chain NFA:
    #   state 0 = start, self-loops on all symbols.
    #   states 1..R are the "consume then accept" chain; state 1 = accept (offset-0),
    #   generic state j accepts after consuming j-1 more symbols.
    # pattern (r,c): transition 0 -(c)-> chain state (r) which then walks down to 1.
    R = best_maxR
    # chain state index j in 1..R ; accept state = 1 (needs 0 more symbols).
    S = 1 + R
    ACC = 1
    edges = []
    # start self-loops
    for c in alphabet:
        edges.append((0, c, 0))
    # chain down transitions: j -(any)-> j-1  for j=2..R
    for j in range(2, R + 1):
        for c in alphabet:
            edges.append((j, c, j - 1))
    # one branch per chosen pattern: (r,c): 0 -(c)-> state r
    seen = set()
    for (r, c, cov) in chosen:
        key = (0, c, r)
        if key in seen:
            continue
        seen.add(key)
        edges.append((0, c, r))
    starts = [0]
    acc = [ACC]
    emit(S, starts, acc, edges)

main()
