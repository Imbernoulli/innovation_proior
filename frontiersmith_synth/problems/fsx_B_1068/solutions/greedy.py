# TIER: greedy
"""Obvious approach: rank the rhyme classes ONCE by their raw 0-inversion word count (as given
in the lexicon) and round-robin the 7 scheme pairs across only the top-3 ranked classes -- a
"just consider the best few candidates" shortlist a coder reaches for without checking whether
MORE rich classes exist that the real 7-pair demand could also use. It does track how much of
each shortlisted class it has already spent (so it isn't blindly reusing a fixed single word),
it just never reconsiders the shortlist itself, so classes ranked 4th and lower sit unused even
when the instance has plenty of them, and the couplet at the end gets whatever the 3-way
round-robin has left."""
import sys
from collections import defaultdict

PAIRS = [(0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 13)]
N_LINES = 14
N_SLOTS = 5
SHORTLIST_SIZE = 3


def main():
    data = sys.stdin.read().split()
    w = int(data[0]); budget = int(data[1])
    pos = 2
    lexicon = []
    for _ in range(w):
        st, c, letter = data[pos], data[pos + 1], data[pos + 2]
        pos += 3
        lexicon.append((st, int(c), letter))

    goods = defaultdict(list)
    bads = defaultdict(list)
    for idx, (st, c, letter) in enumerate(lexicon):
        (goods if st == "01" else bads)[c].append(idx)
    all_classes = sorted(set(c for _, c, _ in lexicon))

    ranked = sorted(all_classes, key=lambda c: (-len(goods[c]), c))
    shortlist = ranked[:SHORTLIST_SIZE] if len(ranked) >= SHORTLIST_SIZE else ranked

    lines = [[None] * N_SLOTS for _ in range(N_LINES)]
    pair_class = {}
    cursor = {c: 0 for c in shortlist}
    for pidx, (i, j) in enumerate(PAIRS):
        c = shortlist[pidx % len(shortlist)]
        pool = goods[c] if goods[c] else (bads[c] if bads[c] else [0])
        for line_idx in (i, j):
            v = pool[cursor[c] % len(pool)]
            cursor[c] += 1
            lines[line_idx][4] = v
        pair_class[i] = c
        pair_class[j] = c

    # free slots: cheaply reuse the assigned class's own good pool, cycling, no cross-line
    # coordination and no awareness of classes outside the shortlist.
    for li in range(N_LINES):
        c = pair_class[li]
        pool = goods[c] if goods[c] else [lines[li][4]]
        for s in range(4):
            lines[li][s] = pool[s % len(pool)]

    print("\n".join(" ".join(map(str, row)) for row in lines))


if __name__ == "__main__":
    main()
