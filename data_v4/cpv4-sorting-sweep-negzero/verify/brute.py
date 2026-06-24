import sys

def main():
    data = sys.stdin.read().split()
    if len(data) < 2:
        # No (n, T) header at all -> empty problem instance.
        print(0)
        return
    idx = 0
    n = int(data[idx]); idx += 1
    T = int(data[idx]); idx += 1
    a = []
    for _ in range(n):
        a.append(int(data[idx])); idx += 1
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            if a[i] + a[j] <= T:
                cnt += 1
    print(cnt)

main()
