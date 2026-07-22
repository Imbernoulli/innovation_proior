# TIER: trivial
# Reproduces the checker's own baseline: find the first self-compatible
# (palindromic) dictionary letter and repeat it in every slot, including
# the center. Always feasible, but wastes the whole compatibility graph
# and uses only 1 distinct letter.
import sys


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

    anchor = None
    for w in letters:
        if w == w[::-1]:
            anchor = w
            break
    assert anchor is not None

    rows = [anchor[5 * r:5 * r + 5] for r in range(7)]
    out_lines = [rows[r] * k for r in range(7)]
    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
