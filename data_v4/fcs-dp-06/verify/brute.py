import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    a = [int(next(it)) for _ in range(n)]

    # O(n^2): count unordered pairs i<j with a[i] AND a[j] == 0.
    count = 0
    for i in range(n):
        ai = a[i]
        for j in range(i + 1, n):
            if (ai & a[j]) == 0:
                count += 1
    print(count)

if __name__ == "__main__":
    main()
