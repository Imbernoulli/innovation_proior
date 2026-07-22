# TIER: greedy
"""
The obvious first pass: only ever look for the SHORT (length-2 and length-3)
substitutions that are easy to spot by direct diffing of (word, root) pairs,
and apply them with the standard "maximal munch" convention (prefer the
longer of two candidate matches at a position).  This reproduces the training
corpus's short-window behaviour essentially perfectly, but it cannot even
REPRESENT a rule whose left-hand side is 4 or 5 glyphs long -- those rules
almost never get a chance to fire inside a length<=9 training word, yet they
fire constantly (and interact with the short rules) inside the long,
held-out inscriptions.
"""
import sys


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0 AA A")
        return
    idx = 0
    n = int(data[idx]); idx += 1
    _t = int(data[idx]); idx += 1
    rules = {}
    for _ in range(n):
        w = data[idx]; idx += 1
        r = data[idx]; idx += 1
        if len(w) in (2, 3) and r != w and len(r) == 1:
            rules[w] = r

    out = []
    # maximal munch: longer left-hand side gets a LOWER (better) priority number
    for lhs, rhs in sorted(rules.items(), key=lambda kv: (-len(kv[0]), kv[0])):
        pr = -len(lhs)
        out.append("%d %s %s" % (pr, lhs, rhs))
    if not out:
        out.append("0 AA A")
    print("\n".join(out))


if __name__ == "__main__":
    main()
