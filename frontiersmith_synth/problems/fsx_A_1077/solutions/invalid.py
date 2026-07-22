# TIER: invalid
# Emits a syntactically well-formed but functionally wrong SLP: every output
# row is wired to the same single instruction (x_1 + x_2), which cannot equal
# every genuinely mixed row of A -> the exact-equivalence gate rejects it.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    for _ in range(n * n):
        next(it)

    idx = n + 1
    lines = ["1", "%d 1 + %d" % (idx, min(2, n))]
    lines.append(" ".join(str(idx) for _ in range(n)))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
