# TIER: greedy
# The "obvious first pass": count every candidate substring's frequency over the
# WHOLE corpus (all 12 documents concatenated, domain identity ignored), rank by
# a standard dictionary-training heuristic (count * (length-1) = estimated chars
# saved), and greedily fill the budget from the top -- skipping a candidate that
# is already covered as a substring of a longer entry already chosen. This is a
# textbook single global frequency pass with no notion of "which document needs
# help" and no notion of "which substrings are useful in MANY domains at once".
import sys
from collections import Counter


def substrings_counter(doc, minlen, maxlen):
    c = Counter()
    n = len(doc)
    for L in range(minlen, maxlen + 1):
        for i in range(0, n - L + 1):
            c[doc[i:i + L]] += 1
    return c


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    D = int(next(it)); K = int(next(it)); M = int(next(it))
    minlen = int(next(it)); maxlen = int(next(it))
    docs = [next(it) for _ in range(D)]

    global_counts = Counter()
    for doc in docs:
        global_counts.update(substrings_counter(doc, minlen, maxlen))

    scored = sorted(global_counts.items(), key=lambda kv: -(kv[1] * (len(kv[0]) - 1)))

    chosen = []
    budget = 0
    for s, cnt in scored:
        if len(chosen) >= M:
            break
        if budget + len(s) > K:
            continue
        if any(s in existing for existing in chosen):
            continue  # already covered by a longer entry we picked
        chosen.append(s)
        budget += len(s)

    print(len(chosen))
    for s in chosen:
        print(s)


if __name__ == "__main__":
    main()
