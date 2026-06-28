#!/usr/bin/env python3
# Independent brute force for the fully-persistent versioned-sequence problem.
# We keep an explicit Python list for EVERY version and apply each operation
# literally on plain lists. This is obviously correct (no treap, no laziness);
# it is only slow/memory-heavy, which is fine for the small differential tests.
import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    q = int(data[idx]); idx += 1

    versions = [[]]   # version 0 = empty sequence
    out = []

    for _ in range(q):
        t = int(data[idx]); idx += 1
        if t == 1:                       # insert x at position p of version v
            v = int(data[idx]); p = int(data[idx+1]); x = int(data[idx+2]); idx += 3
            seq = list(versions[v])
            seq.insert(p, x)             # 0 <= p <= len: element ends up at index p
            versions.append(seq)
        elif t == 2:                     # reverse [l,r] of version v
            v = int(data[idx]); l = int(data[idx+1]); r = int(data[idx+2]); idx += 3
            seq = list(versions[v])
            seq[l:r+1] = seq[l:r+1][::-1]
            versions.append(seq)
        else:                            # query sum of [l,r] of version v
            v = int(data[idx]); l = int(data[idx+1]); r = int(data[idx+2]); idx += 3
            seq = versions[v]
            out.append(str(sum(seq[l:r+1])))

    sys.stdout.write("\n".join(out) + ("\n" if out else ""))

if __name__ == "__main__":
    main()
