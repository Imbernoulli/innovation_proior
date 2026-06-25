import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    iv = []
    for _ in range(n):
        s = int(data[idx]); e = int(data[idx+1]); v = int(data[idx+2]); idx += 3
        iv.append((s, e, v))

    def overlap(a, b):
        return a[0] < b[1] and b[0] < a[1]

    # Greedy 1: value-descending, take if compatible with all already-taken.
    chosen = []
    for it in sorted(iv, key=lambda x: -x[2]):
        if all(not overlap(it, c) for c in chosen):
            chosen.append(it)
    g_value = sum(c[2] for c in chosen)

    # Greedy 2: earliest finishing time (classic count-maximizer), summing values.
    chosen2 = []
    last_end = None
    for it in sorted(iv, key=lambda x: (x[1], x[0])):
        if last_end is None or it[0] >= last_end:
            chosen2.append(it)
            last_end = it[1]
    g_finish = sum(c[2] for c in chosen2)

    print(g_value, g_finish)

main()
