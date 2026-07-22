# TIER: greedy
"""The obvious 'online-adaptive' first attempt: as each symbol arrives, judge
it only against the symbols seen SO FAR (its rank within the window arrived
up to now) and bind it a slot at the matching proportional position among
the still-unused slots. This mirrors adaptive-Huffman-style reasoning: it
never looks past the current arrival, so it cannot anticipate a big symbol
that hasn't shown up yet -- it treats every prefix as if it were the whole
stream. It ignores the fact that every f_i is already sitting in the input,
final and complete, from line one."""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    f = [int(data[idx + i]) for i in range(n)]; idx += n
    L = [int(data[idx + i]) for i in range(n)]; idx += n

    remaining = sorted(L)  # ascending, mutable pool of unused slots
    seen = []
    out = [0] * n

    for t in range(n):
        seen.append(f[t])
        # rank of f[t] among symbols arrived so far (1 = biggest so far)
        rank = sum(1 for v in seen if v > f[t]) + 1
        m = len(seen)
        k = len(remaining)
        pos = (rank - 1) * k // m
        if pos >= k:
            pos = k - 1
        if pos < 0:
            pos = 0
        out[t] = remaining.pop(pos)

    print(" ".join(str(x) for x in out))


if __name__ == "__main__":
    main()
