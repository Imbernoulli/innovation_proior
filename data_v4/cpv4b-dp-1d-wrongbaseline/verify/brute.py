import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]
    idx += n
    if n == 0:
        print(0)
        return
    # Maximum-sum NON-EMPTY circular contiguous subarray.
    # A segment is defined by a start position s (0..n-1) and a length L (1..n).
    # It uses indices s, s+1, ..., s+L-1 modulo n, each element at most once
    # (length capped at n so no element repeats).
    best = None
    for s in range(n):
        cur = 0
        for L in range(1, n + 1):
            cur += a[(s + L - 1) % n]
            if best is None or cur > best:
                best = cur
    print(best)

main()
