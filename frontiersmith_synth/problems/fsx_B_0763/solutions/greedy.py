# TIER: greedy
"""Textbook cuckoo-style insertion: walk the ledger in order; seat each regular in any
open cot among its 4 (room,slot) options. If none is open, evict-and-relocate (classic
cuckoo hashing insertion / bipartite-matching augmenting path): try each of the 4
options in fixed preference order, and for the first one that is occupied, recursively
try to relocate its current occupant elsewhere, then take that cot. Only if no
augmenting path exists at all does the regular go to the annex.

This always finds SOME real cot when one is reachable (strictly better than the trivial
walk, which gives up the moment its own 4 options are full) -- but the eviction target is
chosen by search order alone, never by weighing frequency, so it can spend a rescue on
the wrong occupant."""
import sys
sys.setrecursionlimit(20000)


def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    s = int(next(it))
    r1 = [0] * n
    r2 = [0] * n
    for i in range(n):
        r1[i] = int(next(it))
        r2[i] = int(next(it))
        next(it)  # frequency: irrelevant to this cost-oblivious tier

    slot_owner = {}     # (room, slot) -> key index
    key_slot = [None] * n

    def options(i):
        return ((r1[i], 1), (r1[i], 2), (r2[i], 1), (r2[i], 2))

    def try_place(i, visited):
        opts = options(i)
        # first pass: any directly-open cot?
        for opt in opts:
            if opt not in visited and opt not in slot_owner:
                slot_owner[opt] = i
                key_slot[i] = opt
                return True
        # last resort: evict along the first augmenting path found, in fixed order
        for opt in opts:
            if opt in visited:
                continue
            visited.add(opt)
            j = slot_owner[opt]
            if try_place(j, visited):
                slot_owner[opt] = i
                key_slot[i] = opt
                return True
        return False

    annex_used = 0
    for i in range(n):
        if not try_place(i, set()):
            annex_used += 1
            key_slot[i] = "ANNEX"

    out = []
    for i in range(n):
        ks = key_slot[i]
        if ks == "ANNEX":
            out.append("0")
        else:
            room, slot = ks
            if room == r1[i]:
                code = 1 if slot == 1 else 2
            else:
                code = 3 if slot == 1 else 4
            out.append(str(code))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
