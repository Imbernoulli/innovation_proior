# TIER: trivial
# Reproduces the checker's own reference construction: batch each direction
# to capacity K (earliest-deadline-first within a direction), but run ALL
# of one direction's lockages before ever switching to the other. Deadline
# and capacity aware, but completely blind to the water-parity bit.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it)); L = int(next(it))
    Wsame = int(next(it)); Wdiff = int(next(it)); s0 = int(next(it))
    dirs = []; arr = []; dl = []
    for i in range(N):
        d = int(next(it)); a = int(next(it)); dd = int(next(it)); w = int(next(it))
        dirs.append(d); arr.append(a); dl.append(dd)

    d0 = sorted([i for i in range(N) if dirs[i] == 0], key=lambda i: (dl[i], arr[i]))
    d1 = sorted([i for i in range(N) if dirs[i] == 1], key=lambda i: (dl[i], arr[i]))
    first_dir = 0
    if d1 and (not d0 or dl[d1[0]] < dl[d0[0]]):
        first_dir = 1
    pools = {0: d0, 1: d1}

    lockages = []
    t_prev = None
    for bd in (first_dir, 1 - first_dir):
        pool = pools[bd]
        for start in range(0, len(pool), K):
            ids = pool[start:start + K]
            if not ids:
                continue
            max_a = max(arr[i] for i in ids)
            t = max_a if t_prev is None else max(t_prev + L, max_a)
            lockages.append((t, bd, ids))
            t_prev = t

    out = [str(len(lockages))]
    for (t, bd, ids) in lockages:
        out.append(f"{t} {bd} {len(ids)} " + " ".join(str(i + 1) for i in ids))
    print("\n".join(out))


if __name__ == "__main__":
    main()
