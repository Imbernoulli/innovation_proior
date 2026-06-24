import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0]); L = int(data[1])
    s = data[2] if len(data) > 2 else ""
    if L > n or L <= 0:
        print(0)
        return
    # Independent brute force: use actual substrings as dictionary keys.
    counts = {}
    for i in range(0, n - L + 1):
        w = s[i:i+L]
        counts[w] = counts.get(w, 0) + 1
    ans = 0
    for c in counts.values():
        ans += c * (c - 1) // 2
    print(ans)

main()
