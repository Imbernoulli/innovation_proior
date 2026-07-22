# TIER: strong
"""The insight: the creature is not data to trace, it is the FIXPOINT of an unknown
recursion.  Compression = recovering the generative rule, not the outline.

The transcript T is phi^n(a) for a uniform, prolongable morphism phi.  Such a string is
an m-automatic sequence, so phi is recoverable in a GENERAL, non-hardcoded way: for a
prolongable morphism, phi(x) is exactly the length-m block at the FIRST occurrence of x,
because u[k*m : k*m+m] = phi(u[k]) for the fixpoint u and T is a prefix of u.  We search
m in {2,3,4,5}, read off the rules, and VERIFY by re-expanding.  The recovered L-system is
axiom + a constant number of length-m rules + the integer n -> O(1) tokens, INDEPENDENT of
image size.  That is what a grammar (greedy RePair) structurally cannot achieve: it must
spend a fresh rule per level (~ n rules).  On failure we fall back to RePair (never worse)."""
import sys


def expand(axiom, rules, n, cap=4_000_000):
    cur = list(axiom)
    for _ in range(n):
        nxt = []
        for s in cur:
            r = rules.get(s)
            if r is None:
                nxt.append(s)
            else:
                nxt.extend(r)
            if len(nxt) > cap:
                return None
        cur = nxt
    return cur


def recover(T):
    L = len(T)
    for m in range(2, 6):
        n = 0
        p = 1
        while p < L:
            p *= m
            n += 1
        if p != L or n == 0:
            continue
        a = T[0]
        first = {}
        for i, s in enumerate(T):
            if s not in first:
                first[s] = i
        rules = {}
        okflag = True
        for s, k in first.items():
            if (k + 1) * m > L:
                okflag = False
                break
            rules[s] = list(T[k * m:(k + 1) * m])
        if not okflag:
            continue
        if expand([a], rules, n) == list(T):
            return (a, rules, n)
    return None


# ---- RePair fallback (only used if recovery ever fails) ----
def repair(seq):
    seq = list(seq)
    rules = []
    nid = 0
    while True:
        counts = {}
        for i in range(len(seq) - 1):
            pr = (seq[i], seq[i + 1])
            counts[pr] = counts.get(pr, 0) + 1
        if not counts:
            break
        best = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        pair, cnt = best
        if cnt < 2:
            break
        name = "N%d" % nid
        nid += 1
        rules.append((name, [pair[0], pair[1]]))
        newseq = []
        i = 0
        ln = len(seq)
        while i < ln:
            if i < ln - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                newseq.append(name)
                i += 2
            else:
                newseq.append(seq[i])
                i += 1
        seq = newseq
    return seq, rules


def emit_repair(T):
    reduced, rules = repair(T)
    rmap = {name: rhs for name, rhs in rules}
    depth = {}

    def d(sym):
        if sym not in rmap:
            return 0
        if sym in depth:
            return depth[sym]
        depth[sym] = 1 + max(d(c) for c in rmap[sym])
        return depth[sym]

    n = 0
    for name, _ in rules:
        n = max(n, d(name))
    for s in reduced:
        n = max(n, d(s))
    out = [str(n), " ".join(reduced), str(len(rules))]
    for name, rhs in rules:
        out.append(name + " " + " ".join(rhs))
    sys.stdout.write("\n".join(out) + "\n")


def main():
    data = sys.stdin.read().split("\n")
    T = data[1].split() if len(data) > 1 else []
    rec = recover(T)
    if rec is None:
        emit_repair(T)
        return
    a, rules, n = rec
    lines = [str(n), a, str(len(rules))]
    for s, rhs in rules.items():
        lines.append(s + " " + " ".join(rhs))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
