# TIER: greedy
# The "obvious" recipe: at every position, place the letter that is currently most
# behind its target frequency (ties broken ascending), skipping any letter that would
# create a square factor; on a dead end, undo the single most recent placement and try
# the next-best letter there (minimal recovery -- enough to reach length L reliably).
# It tracks ONLY the running per-letter counts.  It never looks at which letter should
# follow which, so it is blind to the target transition preference in the input --
# it reproduces whatever transition pattern its own tie-break rule happens to favor,
# which the generator's target deliberately does not match.
import sys


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


def build(L, w, letters=4):
    word = []
    counts = [0] * letters
    forbidden = [set() for _ in range(L + 1)]
    pos = 0
    while pos < L:
        n = pos
        order = sorted(range(letters), key=lambda i: (-(w[i] * (n + 1) - counts[i]), i))
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
    # target succ (data[5..8]) intentionally unused -- that is the point of this tier.
    word = build(L, w)
    print(len(word))
    print("".join(str(c) for c in word))


if __name__ == "__main__":
    main()
