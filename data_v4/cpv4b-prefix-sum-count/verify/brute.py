import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    S = int(data[idx]); idx += 1
    a = []
    for _ in range(n):
        a.append(int(data[idx])); idx += 1

    # Obviously-correct O(n^2) enumeration: for every non-empty subarray [l, r],
    # accumulate its sum and test equality with S.
    count = 0
    for l in range(n):
        s = 0
        for r in range(l, n):
            s += a[r]
            if s == S:
                count += 1
    print(count)

if __name__ == "__main__":
    main()
