# TIER: strong
# Insight over raw de-duplication: NORMALIZE each word by the given rules before
# comparing. Reduce every word to its forward normal form (repeatedly apply the
# first applicable shortlex-decreasing rule until irreducible), group words by
# normal form, and emit one shortlex-minimal representative per normal form.
#
# This recognizes every equivalence the given rules make *directly* visible, so it
# beats the raw-string greedy by a wide margin. It is still NOT optimal: the system
# is deliberately non-confluent (divergent critical pairs), so two irreducible words
# can be congruent via a longer intermediate. Forward normalization cannot see those
# merges, so it wastes some of the Nmax budget on words that collapse together. The
# genuinely optimal solver completes the system / closes the congruence over all
# bounded words -- that headroom lives above this reference.
import sys, itertools

def forward_nf(w, rules):
    changed = True
    while changed:
        changed = False
        for (l, r) in rules:
            j = w.find(l)
            if j >= 0:
                w = w[:j] + r + w[j + len(l):]
                changed = True
                break
    return w

def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    k = int(next(it)); L = int(next(it)); Nmax = int(next(it))
    m = int(next(it))
    rules = []
    for _ in range(m):
        l = next(it); r = next(it)
        rules.append((l, r))
    al = [str(i) for i in range(k)]

    def sk(w):
        return (len(w), w)

    # enumerate U in shortlex order so the first word mapping to a normal form is
    # that class's shortlex-minimal representative
    words = []
    for ln in range(1, L + 1):
        for t in itertools.product(al, repeat=ln):
            words.append("".join(t))
    words.sort(key=sk)

    rep = {}
    for w in words:
        nf = forward_nf(w, rules)
        if nf not in rep:
            rep[nf] = w
    reps = sorted(rep.values(), key=sk)[:Nmax]
    sys.stdout.write("\n".join(reps) + "\n")

if __name__ == "__main__":
    main()
