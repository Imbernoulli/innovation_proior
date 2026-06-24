import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    t = int(data[idx]); idx += 1
    a = []
    for _ in range(n):
        a.append(int(data[idx])); idx += 1

    # Independent, obviously-correct method: enumerate every unordered pair i<j
    # and test the divisibility/residue condition directly. O(n^2).
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (a[i] + a[j]) % m == t % m:
                count += 1
    print(count)

if __name__ == "__main__":
    main()
