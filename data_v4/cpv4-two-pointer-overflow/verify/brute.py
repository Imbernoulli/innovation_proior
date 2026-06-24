import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    B = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]
    idx += n

    # Exhaustive: consider every contiguous subarray [i, j], compute its sum,
    # keep the largest sum that is <= B. Empty window (sum 0) is allowed.
    best = 0
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += a[j]
            if s <= B and s > best:
                best = s
    print(best)

main()
