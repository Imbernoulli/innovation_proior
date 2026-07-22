# TIER: greedy
# The "obvious" recipe: separately (a) pick whichever dictionary letter has
# the biggest standalone legibility margin -- the sharpest-looking glyph --
# and paint it into every non-center slot, and (b) pick the best-looking
# self-compatible letter for the center (the one rotation check anyone
# would obviously think to do). It never checks whether the globally
# "sharpest" letter's OWN rotation is decodable at all -- it just assumes
# looking crisp upright is enough. On planted trap instances that sharpest
# letter's rotation is unreadable, and the whole grid becomes infeasible.
import sys


def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def main():
    data = sys.stdin.read().split("\n")
    head = data[0].split()
    L, k = int(head[0]), int(head[1])
    letters = []
    pos = 1
    for _ in range(L):
        rows = data[pos:pos + 7]
        pos += 7
        letters.append("".join(r.strip() for r in rows))

    # standalone margin = gap between distance-to-self (0) and the nearest
    # other dictionary letter
    def own_margin(i):
        dists = sorted(hamming(letters[i], w) for w in letters)
        return dists[1]

    # best overall margin, tie broken toward the LAST (highest-index) letter
    best_i, best_m = 0, -1
    for i in range(L):
        m = own_margin(i)
        if m >= best_m:
            best_m, best_i = m, i

    # center: restrict to letters that already look the same upside down
    center_i, center_m = None, -1
    for i in range(L):
        if letters[i] == letters[i][::-1]:
            m = own_margin(i)
            if m >= center_m:
                center_m, center_i = m, i
    assert center_i is not None

    center = (k + 1) // 2
    rows_out = [[""] * k for _ in range(7)]
    for j in range(1, k + 1):
        pick = letters[center_i] if j == center else letters[best_i]
        for r in range(7):
            rows_out[r][j - 1] = pick[5 * r:5 * r + 5]

    print("\n".join("".join(rows_out[r]) for r in range(7)))


if __name__ == "__main__":
    main()
