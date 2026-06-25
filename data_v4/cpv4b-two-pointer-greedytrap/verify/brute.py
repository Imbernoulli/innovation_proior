import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    B = int(data[idx]); idx += 1
    w = [int(data[idx + k]) for k in range(n)]

    pref = [0] * (n + 1)
    for i in range(n):
        pref[i + 1] = pref[i] + w[i]
    suf = [0] * (n + 1)
    for j in range(n):
        suf[j + 1] = suf[j] + w[n - 1 - j]

    # Try every (i, j): left crane takes i from the front, right takes j from the
    # back, with i + j <= n (no overlap) and pref[i] + suf[j] <= B (shared budget).
    # Maximize i + j.
    best = 0
    for i in range(0, n + 1):
        for j in range(0, n - i + 1):
            if pref[i] + suf[j] <= B:
                if i + j > best:
                    best = i + j
    print(best)

main()
