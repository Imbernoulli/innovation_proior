import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    a = []
    for _ in range(n):
        a.append(int(data[idx])); idx += 1

    # Obviously-correct O(n^2) brute force: enumerate every contiguous window [l, r],
    # accumulate its sum, count those divisible by m (Python % gives non-negative
    # remainder for positive m, so "s % m == 0" is the plain mathematical test).
    answer = 0
    for l in range(n):
        s = 0
        for r in range(l, n):
            s += a[r]
            if s % m == 0:
                answer += 1
    print(answer)

main()
