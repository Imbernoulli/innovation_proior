import sys

def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    L = int(next(it))
    R = int(next(it))
    a = [int(next(it)) for _ in range(n)]

    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(a[i] - a[j])
            if L <= d <= R:
                cnt += 1
    print(cnt)

if __name__ == "__main__":
    main()
