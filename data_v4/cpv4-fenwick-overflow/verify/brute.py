import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    a = []
    for _ in range(n):
        a.append(int(data[idx])); idx += 1
    # Direct O(n^2) definition: sum over i<j with a[i] > a[j] of a[i]*a[j].
    ans = 0
    for i in range(n):
        for j in range(i + 1, n):
            if a[i] > a[j]:
                ans += a[i] * a[j]
    print(ans)

main()
