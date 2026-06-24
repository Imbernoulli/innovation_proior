import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    it = iter(data)
    n = int(next(it))
    jobs = []
    for _ in range(n):
        s = int(next(it)); f = int(next(it)); p = int(next(it))
        jobs.append((s, f, p))

    # Brute force: try every subset of jobs, check pairwise non-overlap (half-open [s,f)),
    # track max profit. Empty subset gives 0. Exponential, only for tiny n.
    best = 0
    for mask in range(1 << n):
        chosen = [jobs[i] for i in range(n) if (mask >> i) & 1]
        ok = True
        for a in range(len(chosen)):
            for b in range(a + 1, len(chosen)):
                s1, f1, _ = chosen[a]
                s2, f2, _ = chosen[b]
                # overlap if not (f1 <= s2 or f2 <= s1)
                if not (f1 <= s2 or f2 <= s1):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            tot = sum(c[2] for c in chosen)
            if tot > best:
                best = tot
    print(best)

main()
