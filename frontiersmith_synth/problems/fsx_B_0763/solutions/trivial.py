# TIER: trivial
"""Walk the ledger in the given order; seat each regular in the first open cot among
(choice1,slot1)->(choice1,slot2)->(choice2,slot1)->(choice2,slot2); if all four are
already taken, write them into the annex. No eviction, no lookahead."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    s = int(next(it))
    r1 = [0] * n
    r2 = [0] * n
    for i in range(n):
        r1[i] = int(next(it))
        r2[i] = int(next(it))
        next(it)  # frequency, unused by this tier

    used = set()
    annex_used = 0
    out = []
    for i in range(n):
        placed = False
        for code, room, slot in ((1, r1[i], 1), (2, r1[i], 2), (3, r2[i], 1), (4, r2[i], 2)):
            if (room, slot) not in used:
                used.add((room, slot))
                out.append(str(code))
                placed = True
                break
        if not placed:
            annex_used += 1
            out.append("0")

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
