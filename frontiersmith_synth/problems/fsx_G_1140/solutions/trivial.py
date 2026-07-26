# TIER: trivial
"""
Ignore which symbol occurs where: fit ONE global rate = sum(cost)/sum(length)
over the training rows, and emit a 1-state automaton that charges that same
rate for both 'a' and 'b'. This reproduces the checker's own internal
baseline predictor exactly.
"""
import sys


def main():
    data = sys.stdin.read().split("\n")
    first = data[0].split()
    n = int(first[1])
    rows = []
    for i in range(2, 2 + n):
        parts = data[i].split()
        s, c = parts[0], int(parts[1])
        rows.append((s, c))

    total_c = sum(c for _, c in rows)
    total_l = sum(len(s) for s, _ in rows)
    rate = total_c / max(1.0, total_l)

    out = []
    out.append("1 2")
    out.append("0 a 0 %.6f" % rate)
    out.append("0 b 0 %.6f" % rate)
    out.append("0")
    out.append("1")
    out.append("0 0.0")
    print("\n".join(out))


if __name__ == "__main__":
    main()
