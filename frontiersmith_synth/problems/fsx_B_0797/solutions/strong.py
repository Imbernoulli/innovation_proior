# TIER: strong
# The insight: the objective is a MIN over 12 documents, so the score is set by
# whichever document is currently worst off -- spending budget on whatever is
# globally most frequent (the greedy recipe) is the wrong currency. Instead this
# runs a max-min WATER-FILLING allocation: repeatedly find the document that is
# currently worst, and spend the next budget unit on the substring that helps
# THAT document the most per character of budget -- with a bonus for substrings
# that also occur in several OTHER documents (a cross-domain shared stem raises
# more than one document's ratio for the same one-time budget cost, which is
# worth more than its single-domain frequency alone would suggest). This is an
# exchange-argument reformulation (marginal-gain equalization), not "greedy with
# more iterations": the objective each step optimizes is the worst document's own
# ratio, not raw corpus-wide frequency.
import sys
from collections import Counter, defaultdict


def substrings_counter(doc, minlen, maxlen):
    c = Counter()
    n = len(doc)
    for L in range(minlen, maxlen + 1):
        for i in range(0, n - L + 1):
            c[doc[i:i + L]] += 1
    return c


def compress_tokens(doc, by_len):
    n = len(doc)
    i = 0
    tokens = 0
    lens_desc = sorted(by_len.keys(), reverse=True)
    while i < n:
        matched = False
        for L in lens_desc:
            if L > n - i:
                continue
            if doc[i:i + L] in by_len[L]:
                i += L
                tokens += 1
                matched = True
                break
        if not matched:
            i += 1
            tokens += 1
    return tokens


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    D = int(next(it)); K = int(next(it)); M = int(next(it))
    minlen = int(next(it)); maxlen = int(next(it))
    docs = [next(it) for _ in range(D)]

    doc_counters = [substrings_counter(doc, minlen, maxlen) for doc in docs]
    # candidate -> set of document indices where it occurs (>=2 times, i.e. a
    # genuine repeated stem, not an incidental one-off)
    cand_domains = defaultdict(set)
    for di, c in enumerate(doc_counters):
        for s, cnt in c.items():
            if cnt >= 2:
                cand_domains[s].add(di)

    chosen = []
    chosen_set = set()
    budget = 0
    by_len = defaultdict(set)
    tok = [len(doc) for doc in docs]
    rs = [1.0] * D  # every document starts fully literal -> ratio 1.0

    def recompute(k):
        tok[k] = compress_tokens(docs[k], by_len)
        rs[k] = len(docs[k]) / tok[k]

    while budget < K and len(chosen) < M:
        worst = min(range(D), key=lambda k: rs[k])
        wc = doc_counters[worst]
        wlen = len(docs[worst])

        # stage 1: cheap shortlist by frequency * length * cross-domain spread
        shortlist = []
        for s, cnt in wc.items():
            if cnt < 2 or s in chosen_set or len(s) > K - budget:
                continue
            spread = len(cand_domains[s])
            h = cnt * (len(s) - 1) * (1.0 + 0.6 * (spread - 1))
            shortlist.append((h, s))
        if not shortlist:
            break
        shortlist.sort(reverse=True)
        top = shortlist[:60]

        # stage 2: exact marginal ratio-gain to the WORST document, per budget char
        best_s, best_eff = None, -1.0
        for _, s in top:
            trial = {L: set(es) for L, es in by_len.items()}
            trial.setdefault(len(s), set()).add(s)
            new_tok = compress_tokens(docs[worst], trial)
            new_ratio = wlen / new_tok
            gain = new_ratio - rs[worst]
            if gain <= 1e-12:
                continue
            eff = gain / len(s)
            if eff > best_eff:
                best_eff = eff
                best_s = s
        if best_s is None:
            best_s = top[0][1]

        chosen.append(best_s)
        chosen_set.add(best_s)
        budget += len(best_s)
        by_len[len(best_s)].add(best_s)
        for k in cand_domains[best_s]:
            recompute(k)

    print(len(chosen))
    for s in chosen:
        print(s)


if __name__ == "__main__":
    main()
