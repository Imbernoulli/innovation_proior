import sys
from functools import lru_cache

MOD = 1000000007

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    s = data[0]

    # Parse into literals and operators; validate well-formedness exactly like the spec.
    vals = []
    ops = []
    well = True
    for i, c in enumerate(s):
        if i % 2 == 0:
            if c == 'T':
                vals.append(True)
            elif c == 'F':
                vals.append(False)
            else:
                well = False
                break
        else:
            if c in '&|^':
                ops.append(c)
            else:
                well = False
                break
    if len(s) % 2 == 0:
        well = False
    if not well:
        print(0)
        return

    m = len(vals)

    def apply(a, op, b):
        if op == '&':
            return a and b
        if op == '|':
            return a or b
        return a != b  # xor

    # Enumerate ALL full parenthesizations explicitly and tally results.
    # ways(i, j) -> dict mapping {True: count, False: count} of distinct
    # parenthesizations of literals i..j (inclusive) by their evaluated value.
    @lru_cache(maxsize=None)
    def ways(i, j):
        if i == j:
            return ((1, 0) if vals[i] else (0, 1))  # (#True, #False)
        cntT = 0
        cntF = 0
        for k in range(i, j):
            lt, lf = ways(i, k)
            rt, rf = ways(k + 1, j)
            for lv, lc in ((True, lt), (False, lf)):
                for rv, rc in ((True, rt), (False, rf)):
                    c = lc * rc
                    if c == 0:
                        continue
                    if apply(lv, ops[k], rv):
                        cntT += c
                    else:
                        cntF += c
        return (cntT % MOD, cntF % MOD)

    print(ways(0, m - 1)[0] % MOD)

main()
