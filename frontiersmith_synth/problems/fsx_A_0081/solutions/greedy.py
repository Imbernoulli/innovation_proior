# TIER: greedy
import sys, itertools

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    n = int(next(it)); b = int(next(it))
    flooded = set(next(it) for _ in range(b))

    all_cells = ["".join(c) for c in itertools.product("012", repeat=n)]
    allowed = [c for c in all_cells if c not in flooded]  # canonical lexicographic order

    chosen = []
    cs = set()
    tups = {}
    for c in allowed:
        t = tuple(ord(ch) - 48 for ch in c)
        ok = True
        for a in chosen:
            ta = tups[a]
            third = "".join(chr(48 + ((3 - (ta[i] + t[i]) % 3) % 3)) for i in range(n))
            if third in cs and third != c and third != a:
                ok = False
                break
        if ok:
            chosen.append(c)
            cs.add(c)
            tups[c] = t
    print(len(chosen))
    if chosen:
        print("\n".join(chosen))

main()
