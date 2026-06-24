import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    idx = 0
    n = int(data[idx]); idx += 1
    d = [int(data[idx + i]) for i in range(n)]
    idx += n

    # Levels: P[-1] = 0 (index 0 in our list), then P[0..n-1].
    # levels[k] is the level after k days; levels[0] = start = 0.
    levels = [0]
    cur = 0
    for x in d:
        cur += x
        levels.append(cur)

    # Maximum decline = max over i <= j of levels[i] - levels[j]; i = j gives 0.
    best = 0
    m = len(levels)
    for i in range(m):
        for j in range(i, m):
            best = max(best, levels[i] - levels[j])
    print(best)

main()
