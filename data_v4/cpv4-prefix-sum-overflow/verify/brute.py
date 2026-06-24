import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    q = int(data[idx]); idx += 1
    a = [0] * (n + 1)  # 1-indexed
    for i in range(1, n + 1):
        a[i] = int(data[idx]); idx += 1

    total = 0
    for _ in range(q):
        l = int(data[idx]); idx += 1
        r = int(data[idx]); idx += 1
        # Recompute the window sum from scratch by direct summation.
        s = 0
        for i in range(l, r + 1):
            s += a[i]
        total += s

    print(total)

if __name__ == "__main__":
    main()
