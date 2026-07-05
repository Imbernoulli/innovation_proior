# TIER: strong
# Full-band greedy (Mian-Chowla over the whole spectrum [1..n]) plus seeded randomized
# restarts; keeps the best Sidon set found. Uses the entire band, so it strictly
# dominates the lower-sub-band greedy.
import sys, random

def build(order, forb):
    chosen = []
    sums = set()
    for c in order:
        if c in forb:
            continue
        cand = set()
        ok = True
        for a in chosen:
            s = a + c
            if s in sums or s in cand:
                ok = False
                break
            cand.add(s)
        if ok:
            s = 2 * c
            if s in sums or s in cand:
                ok = False
            else:
                cand.add(s)
        if ok:
            chosen.append(c)
            sums |= cand
    return chosen

def main():
    d = sys.stdin.read().split()
    n = int(d[0]); k = int(d[1])
    forb = set(int(x) for x in d[2:2 + k])
    allowed = [c for c in range(1, n + 1) if c not in forb]

    best = build(allowed, forb)  # ascending full-band greedy (>= lower-band greedy)

    rnd = random.Random(2027)
    tries = 30 if n <= 3000 else 10
    for _ in range(tries):
        o = allowed[:]
        # keep a small ascending head then perturb the tail; ascending head keeps density high
        head = o[: max(1, len(o) // 4)]
        tail = o[max(1, len(o) // 4):]
        rnd.shuffle(tail)
        cur = build(head + tail, forb)
        if len(cur) > len(best):
            best = cur

    print(" ".join(map(str, sorted(best))))

if __name__ == "__main__":
    main()
