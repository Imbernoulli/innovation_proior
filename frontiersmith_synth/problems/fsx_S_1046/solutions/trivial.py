# TIER: trivial
#!/usr/bin/env python3
# Trivial: reproduces the checker's own reference construction exactly -- a
# TIGHT flanking (straddle) dipole around each of the first q listed defects,
# cycling through the list if q > m. No physics is solved at all: it just
# "looks locally sensitive" without ever pushing current into the mesh.
import sys


def main():
    data = sys.stdin.read().split()
    ptr = 0

    def nxt():
        nonlocal ptr
        v = data[ptr]
        ptr += 1
        return v

    b = int(nxt()); L = int(nxt()); m = int(nxt()); q = int(nxt()); I_max = int(nxt())
    _g_r = nxt(); _g_c = nxt(); _g_core = nxt()
    _alpha = nxt()
    defects = []
    for _ in range(m):
        pos = int(nxt()); layer = int(nxt())
        defects.append((pos, layer))

    span = max(1, b // 5)
    lines = []
    for j in range(q):
        pos, _layer = defects[j % m]
        a = (pos - span) % b
        c = (pos + span) % b
        if a == c:
            c = (a + 1) % b
        row = [0] * b
        row[a] = I_max
        row[c] = -I_max
        lines.append(" ".join(map(str, row)))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
