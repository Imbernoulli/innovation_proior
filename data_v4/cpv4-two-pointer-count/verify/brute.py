import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    L = int(data[idx]); idx += 1
    R = int(data[idx]); idx += 1
    a = []
    for _ in range(n):
        a.append(int(data[idx])); idx += 1

    # Obvious O(n^2): enumerate every contiguous subarray, sum it, count those in [L,R].
    count = 0
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += a[j]
            if L <= s <= R:
                count += 1
    print(count)

if __name__ == "__main__":
    main()
