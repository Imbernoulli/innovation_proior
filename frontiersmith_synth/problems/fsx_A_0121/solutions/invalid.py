# TIER: invalid
# Emits a catalogue that contains a resonant triple: the three profiles
# 0..0, (1,0..0), (2,0..0) sum to 0 mod 3 on every river -> must score 0.
import sys


def main():
    n = int(sys.stdin.read().split()[0])
    rows = [[0] * n, [1] + [0] * (n - 1), [2] + [0] * (n - 1)]
    out = [str(len(rows))]
    for r in rows:
        out.append(" ".join(map(str, r)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
