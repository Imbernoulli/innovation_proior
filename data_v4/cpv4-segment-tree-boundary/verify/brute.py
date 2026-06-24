import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    try:
        n = int(next(it)); q = int(next(it))
    except StopIteration:
        return
    a = [0] * (n + 1)              # 1-indexed
    for i in range(1, n + 1):
        a[i] = int(next(it))

    out = []
    for _ in range(q):
        t = int(next(it))
        if t == 1:
            p = int(next(it)); x = int(next(it))
            a[p] = x
        else:
            l = int(next(it)); r = int(next(it))
            # Longest strictly-increasing contiguous run fully inside [l, r].
            best = 1                # window has >=1 position, so a single point counts
            cur = 1
            for i in range(l + 1, r + 1):
                if a[i] > a[i - 1]:
                    cur += 1
                else:
                    cur = 1
                if cur > best:
                    best = cur
            out.append(str(best))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))

main()
