#!/usr/bin/env python3
# Random small-case generator. Usage: gen.py <seed>
# Produces a valid operation stream. We must track each version's current length
# so that every generated (position / range / version-id) reference is in range,
# matching the problem's guarantees. Versions are created by ops of type 1 and 2.
import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    q = rng.randint(1, 30)
    lens = [0]          # lens[v] = length of version v; version 0 is empty
    lines = []
    nlines = 0

    for _ in range(q):
        # pick which operation; bias toward inserts early so sequences grow
        choices = [1, 1, 2, 3]
        # reverse/query need a non-empty version to be interesting, but are legal
        # on empty ranges too; we still allow them and clamp.
        t = rng.choice(choices)
        v = rng.randint(0, len(lens) - 1)
        L = lens[v]
        if t == 1:
            p = rng.randint(0, L)            # 0..L inclusive
            x = rng.randint(-1000, 1000)
            lines.append(f"1 {v} {p} {x}")
            lens.append(L + 1)
        elif t == 2:
            if L == 0:
                # reverse of empty -> make it an insert instead to keep things lively
                p = 0
                x = rng.randint(-1000, 1000)
                lines.append(f"1 {v} {p} {x}")
                lens.append(L + 1)
            else:
                l = rng.randint(0, L - 1)
                r = rng.randint(l, L - 1)
                lines.append(f"2 {v} {l} {r}")
                lens.append(L)               # reverse keeps length
        else:
            if L == 0:
                # query on empty: ask for sum over an empty range is undefined;
                # turn into an insert to keep all queries well-formed (l<=r<len).
                p = 0
                x = rng.randint(-1000, 1000)
                lines.append(f"1 {v} {p} {x}")
                lens.append(L + 1)
            else:
                l = rng.randint(0, L - 1)
                r = rng.randint(l, L - 1)
                lines.append(f"3 {v} {l} {r}")
                nlines += 1

    print(len(lines))
    print("\n".join(lines))

if __name__ == "__main__":
    main()
