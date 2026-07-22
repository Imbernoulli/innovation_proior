# TIER: strong
# The insight: the scaffold's free choices at each position must be steered by BOTH
# statistics the objective actually scores -- not just the running letter counts but
# also which letter the target wants to follow the one just placed.  At each step,
# score every letter that keeps the word square-free by a weighted combination of its
# (normalized) frequency deficit and whether it equals the target successor of the
# previous letter, mirroring the checker's own 0.25/0.75 weighting -- then commit to
# the best-scoring safe letter, backtracking one step on a dead end.  Because it
# actively chases the transition target instead of ignoring it, it reaches a realized
# transition pattern the target actually rewards, not just a plausible-looking mix.
import sys

WU = 0.25
WD = 0.75


def is_square_free_incremental(word, newlen):
    maxp = newlen // 2
    for p in range(1, maxp + 1):
        i = newlen - 2 * p
        j = newlen - p
        eq = True
        for t in range(p):
            if word[i + t] != word[j + t]:
                eq = False
                break
        if eq:
            return False
    return True


def build(L, w, succ, letters=4):
    word = []
    counts = [0] * letters
    forbidden = [set() for _ in range(L + 1)]
    pos = 0
    while pos < L:
        n = pos
        prev = word[-1] if word else None
        deficits = [w[i] * (n + 1) - counts[i] for i in range(letters)]
        lo, hi = min(deficits), max(deficits)
        rng = (hi - lo) or 1.0

        def score(i):
            marg = (deficits[i] - lo) / rng
            bonus = 1.0 if (prev is not None and i == succ[prev]) else 0.0
            return WU * marg + WD * bonus

        order = sorted(range(letters), key=lambda i: (-score(i), i))
        placed = False
        for letter in order:
            if letter in forbidden[pos]:
                continue
            word.append(letter)
            if is_square_free_incremental(word, pos + 1):
                counts[letter] += 1
                pos += 1
                placed = True
                break
            else:
                word.pop()
        if not placed:
            if pos == 0:
                break
            forbidden[pos] = set()
            pos -= 1
            last = word.pop()
            counts[last] -= 1
            forbidden[pos].add(last)
    return word


def main():
    data = sys.stdin.read().split()
    L = int(data[0])
    w = [int(data[1 + i]) / 10000.0 for i in range(4)]
    succ = [int(data[5 + i]) for i in range(4)]
    word = build(L, w, succ)
    print(len(word))
    print("".join(str(c) for c in word))


if __name__ == "__main__":
    main()
