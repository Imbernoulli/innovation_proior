# TIER: trivial
"""Reproduces the checker's own baseline: find the single rhyme class with the most
0-inversion words and cycle through only its distinct words for the entire poem."""
import sys
from collections import defaultdict

N_LINES = 14
N_SLOTS = 5


def main():
    data = sys.stdin.read().split()
    w = int(data[0])
    pos = 2
    lexicon = []
    for _ in range(w):
        st, c, letter = data[pos], data[pos + 1], data[pos + 2]
        pos += 3
        lexicon.append((st, int(c), letter))

    good_by_class = defaultdict(list)
    for idx, (st, c, letter) in enumerate(lexicon):
        if st == "01":
            good_by_class[c].append(idx)
    if not good_by_class:
        pool = [0]
    else:
        best_c = min(good_by_class.keys(), key=lambda c: (-len(good_by_class[c]), c))
        pool = sorted(good_by_class[best_c])

    total = N_LINES * N_SLOTS
    seq = [pool[i % len(pool)] for i in range(total)]
    lines = [seq[i * N_SLOTS:(i + 1) * N_SLOTS] for i in range(N_LINES)]
    print("\n".join(" ".join(map(str, row)) for row in lines))


if __name__ == "__main__":
    main()
