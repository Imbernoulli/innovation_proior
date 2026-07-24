# TIER: greedy
# RePair-style: repeatedly commit to the most frequent adjacent PAIR of tokens,
# replacing all non-overlapping occurrences left-to-right, while the direct
# savings are positive and the rule budget lasts. No lookahead, pairs only.
import sys


def main():
    data = sys.stdin.read().split()
    n, K, c, L, q = map(int, data[:5])
    tok = [ord(ch) - 97 for ch in data[5]]
    rules = []
    for _ in range(K):
        ncur = len(tok)
        if ncur < 2:
            break
        cnt = {}
        a0 = tok[0]
        for i in range(1, ncur):
            b = tok[i]
            key = (a0, b)
            cnt[key] = cnt.get(key, 0) + 1
            a0 = b
        # pick max count, deterministic tie-break on the pair itself
        best = None
        bestc = 0
        for key, v in cnt.items():
            if v > bestc or (v == bestc and best is not None and key < best):
                best = key
                bestc = v
        # non-overlapping occurrences of a pair (a,a) overlap, count greedily
        a, b = best
        occ = 0
        i = 0
        while i < ncur - 1:
            if tok[i] == a and tok[i + 1] == b:
                occ += 1
                i += 2
            else:
                i += 1
        savings = occ - 2 - c   # occ*(2-1) tokens saved minus rhs+overhead
        if savings <= 0:
            break
        sym = q + len(rules)
        new = []
        i = 0
        while i < ncur - 1:
            if tok[i] == a and tok[i + 1] == b:
                new.append(sym)
                i += 2
            else:
                new.append(tok[i])
                i += 1
        if i == ncur - 1:
            new.append(tok[ncur - 1])
        tok = new
        rules.append((a, b))

    print(len(rules))
    out = []
    for a, b in rules:
        out.append(fmt(a, q) + fmt(b, q))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


def fmt(t, q):
    if t < q:
        return chr(97 + t)
    return "#" + str(t - q + 1)


main()
