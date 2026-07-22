# TIER: greedy
"""
The textbook move: for a FIXED (static) list under position-cost access,
the optimal arrangement sorts items by descending raw access frequency
(classic list-update "static optimality" fact). Compute per-item
frequency over the whole day, sort once, and never touch the swap
budget again. This ignores the pair structure entirely -- it only ever
looks at an item's own total count, never at *when* it fires or who it
fires alongside.
"""
import sys


def main():
    data = sys.stdin.read().split()
    ptr = 0
    n = int(data[ptr]); ptr += 1
    m = int(data[ptr]); ptr += 1
    K = int(data[ptr]); ptr += 1
    P = int(data[ptr]); ptr += 1
    ptr += 2 * P  # pair list unused by the frequency-only heuristic
    seq = [int(x) for x in data[ptr:ptr + m]]

    freq = [0] * (n + 1)
    for x in seq:
        freq[x] += 1

    order = sorted(range(1, n + 1), key=lambda it: (-freq[it], it))

    out = []
    out.append(" ".join(str(x) for x in order))
    out.append("0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
