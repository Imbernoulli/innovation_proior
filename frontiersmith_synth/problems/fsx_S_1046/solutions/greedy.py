# TIER: greedy
# Greedy / textbook first-pass: for each of the first q listed suspected
# defects, fire a SPREAD (diametrically opposite) two-electrode dipole
# straight through that defect's rim position -- the obvious "multimeter"
# idea (push current through the bulk toward the known suspect). No
# worst-case reasoning about which defects end up uncovered when q < m, and
# no multi-electrode focusing: each probe only ever uses 2 non-zero entries.
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

    half = b // 2
    lines = []
    for j in range(q):
        pos, _layer = defects[j % m]
        a = pos
        c = (pos + half) % b
        if a == c:
            c = (a + 1) % b
        row = [0] * b
        row[a] = I_max
        row[c] = -I_max
        lines.append(" ".join(map(str, row)))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
