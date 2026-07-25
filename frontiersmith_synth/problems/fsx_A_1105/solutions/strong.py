# TIER: strong
# Insight (innovation hook): a replacement's TRUE cost includes the overlapping
# repeats it destroys for future rules, so the most-frequent pair is often the
# wrong commitment. Instead of pair-frequency greedy:
#   * consider candidate patterns of ALL lengths 2..Lp (a long tandem repeat
#     should be captured by ONE rule on its period, not many pair rules);
#   * score each shortlisted candidate by direct savings PLUS the best savings
#     any other top candidate retains AFTER this replacement is applied
#     (1-step lookahead over overlap-destruction), and commit to the argmax.
import sys
from bisect import bisect_right
from collections import defaultdict


def fmt(t, q):
    if t < q:
        return chr(97 + t)
    return "#" + str(t - q + 1)


def nonoverlap_sel(pos, l):
    sel = []
    last = -10 ** 18
    for p in pos:
        if p >= last:
            sel.append(p)
            last = p + l
    return sel


def main():
    data = sys.stdin.read().split()
    n, K, c, L, q = map(int, data[:5])
    tok = [ord(ch) - 97 for ch in data[5]]
    rules = []

    for _round in range(K):
        ncur = len(tok)
        if ncur < 4:
            break
        # adaptive knobs keep the largest instances inside the time budget
        Lp = min(L, 8) if ncur <= 5000 else min(L, 6)
        T = 10 if ncur <= 5000 else 6
        R = 24 if ncur <= 5000 else 14

        occ = defaultdict(list)
        for l in range(2, Lp + 1):
            tl = tok
            for i in range(ncur - l + 1):
                occ[tuple(tl[i:i + l])].append(i)

        cands = []
        for key, pos in occ.items():
            if len(pos) < 2:
                continue
            l = len(key)
            sel = nonoverlap_sel(pos, l)
            k = len(sel)
            if k < 2:
                continue
            direct = k * (l - 1) - l - c
            if direct > 0:
                cands.append((direct, key, sel))
        if not cands:
            break
        cands.sort(key=lambda x: (-x[0], x[1]))
        pool = cands[:R]
        short = cands[:T]

        best = None  # (score, direct, key, sel)
        for direct, key, sel in short:
            l = len(key)
            ends = [p + l for p in sel]  # replaced intervals [p, p+l)
            residual = 0
            for d2, k2, sel2 in pool:
                if k2 is key:
                    continue
                l2 = len(k2)
                cnt = 0
                last = -10 ** 18
                for p in sel2:
                    if p < last:
                        continue
                    # does [p, p+l2) meet any replaced interval?
                    j = bisect_right(sel, p + l2 - 1)
                    if j > 0 and sel[j - 1] + l > p:
                        continue
                    cnt += 1
                    last = p + l2
                sv = cnt * (l2 - 1) - l2 - c
                if sv > residual:
                    residual = sv
            score = direct + residual
            if best is None or score > best[0] or (score == best[0] and key < best[2]):
                best = (score, direct, key, sel)
        if best is None or best[1] <= 0:
            break

        _, _, key, _ = best
        sym = q + len(rules)
        lp = len(key)
        new = []
        i = 0
        while i <= ncur - lp:
            if tok[i:i + lp] == list(key):
                new.append(sym)
                i += lp
            else:
                new.append(tok[i])
                i += 1
        new.extend(tok[i:])
        tok = new
        rules.append(list(key))

    print(len(rules))
    out = []
    for pat in rules:
        out.append("".join(fmt(t, q) for t in pat))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


main()
