# TIER: greedy
"""The obvious textbook recipe for 'bit-width-selection + block-partitioning':
split the stream into FIXED-SIZE chunks (a size-agnostic constant amortizing
header cost, L ~= sqrt(N)) and, independently per chunk, pick the MINIMUM
width that fits that chunk's values. This is exactly the classic bit-packing
scheme (Parquet/PFOR-style). It never looks at the misprediction cost C at
all, so on streams whose magnitude regime changes on a scale close to L, the
width sequence it emits bounces around and racks up the branch tax."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); H = int(next(it)); C = int(next(it))
    A = [int(next(it)) for _ in range(N)]

    L = max(1, round(N ** 0.5))

    blocks = []
    i = 0
    while i < N:
        ln = min(L, N - i)
        seg = A[i:i + ln]
        d = max(seg) - min(seg)
        w = d.bit_length()
        blocks.append((ln, w))
        i += ln

    print(len(blocks))
    out = []
    for ln, w in blocks:
        out.append(f"{ln} {w}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
