# TIER: greedy
# The "obvious" first idea: process boats earliest-deadline-first (classic
# EDD scheduling) and pack a running lockage greedily -- keep adding to the
# current lockage while the next boat shares its direction and it isn't
# full, otherwise close it and start a new one. This looks efficient (it
# minimizes the number of lockages given the deadline order) but never once
# reasons about the water-parity bit: whenever deadline order clusters many
# same-direction boats together (which it does whenever one direction
# dominates), it happily runs a long same-direction block and pays the
# expensive W_same cost lockage after lockage.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it)); L = int(next(it))
    Wsame = int(next(it)); Wdiff = int(next(it)); s0 = int(next(it))
    boats = []
    for i in range(N):
        d = int(next(it)); a = int(next(it)); dd = int(next(it)); w = int(next(it))
        boats.append((d, a, dd, w, i + 1))

    order = sorted(boats, key=lambda b: (b[2], b[1], b[4]))

    batches = []
    cur = []; cur_dir = None
    for b in order:
        if cur and (b[0] != cur_dir or len(cur) == K):
            batches.append((cur_dir, cur))
            cur = []
        cur.append(b)
        cur_dir = b[0]
    if cur:
        batches.append((cur_dir, cur))

    lockages = []
    t_prev = None
    for (bd, members) in batches:
        ids = [b[4] for b in members]
        max_a = max(b[1] for b in members)
        t = max_a if t_prev is None else max(t_prev + L, max_a)
        lockages.append((t, bd, ids))
        t_prev = t

    out = [str(len(lockages))]
    for (t, dirn, ids) in lockages:
        out.append(f"{t} {dirn} {len(ids)} " + " ".join(map(str, ids)))
    print("\n".join(out))


if __name__ == "__main__":
    main()
