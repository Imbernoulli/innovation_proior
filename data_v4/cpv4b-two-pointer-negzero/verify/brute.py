import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    lo = int(data[idx]); idx += 1
    hi = int(data[idx]); idx += 1
    a = []
    for _ in range(n):
        a.append(int(data[idx])); idx += 1

    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = a[i] + a[j]
            if lo <= s <= hi:
                cnt += 1
    print(cnt)

main()
