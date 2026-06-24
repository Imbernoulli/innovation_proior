import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]
    idx += n

    # Exhaustive: enumerate every contiguous window [l, r], sum it directly,
    # and check whether the sum is divisible by m. This is O(n^2) with an
    # incremental running sum, an obviously-correct independent method.
    answer = 0
    for l in range(n):
        s = 0
        for r in range(l, n):
            s += a[r]
            if s % m == 0:
                answer += 1
    print(answer)

main()
