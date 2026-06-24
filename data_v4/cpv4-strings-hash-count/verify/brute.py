import sys

def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n = int(data[0]); k = int(data[1])
    s = data[2].decode() if len(data) >= 3 else ""
    # Count distinct length-k substrings that occur at >= 2 distinct starting positions.
    if k <= 0 or k > n:
        print(0)
        return
    from collections import defaultdict
    cnt = defaultdict(int)
    for i in range(0, n - k + 1):
        cnt[s[i:i+k]] += 1
    ans = sum(1 for v in cnt.values() if v >= 2)
    print(ans)

main()
