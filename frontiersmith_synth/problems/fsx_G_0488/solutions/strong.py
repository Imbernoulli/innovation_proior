# TIER: strong
# Full no-carry base-3 {0,1}-digit set on the entire lower half [0, p/2) -- the
# classic Salem-Spencer progression-free construction, which is progression-free
# mod p because every pairwise sum stays below p (no modular wraparound). Then a
# bounded, deterministic greedy augmentation pass probes a fixed set of extra
# residues and keeps any that preserve the progression-free property. This is the
# strongest simple construction and beats the greedy tier.
import sys


def base3_nocarry(bound):
    if bound <= 0:
        return []
    pows = []
    v = 1
    while v < bound:
        pows.append(v)
        v *= 3
    out = []
    for mask in range(1 << len(pows)):
        s = 0
        m, i = mask, 0
        while m:
            if m & 1:
                s += pows[i]
            m >>= 1
            i += 1
        if s < bound:
            out.append(s)
    return sorted(set(out))


def main():
    p = int(sys.stdin.read().split()[0])
    half = (p + 1) // 2
    S = base3_nocarry(half)
    Sset = set(S)
    inv2 = pow(2, p - 2, p)

    # Deterministic augmentation: probe a fixed pseudo-scattered order of residues
    # (a linear-congruential walk seeded only by p -> reproducible, no randomness)
    # and keep any that do not create a 3-AP. Cheap and can only grow the set.
    probes = min(p, 4000)
    x = 1
    for _ in range(probes):
        x = (1103515245 * x + 12345) % p
        if x in Sset:
            continue
        bad = False
        for a in Sset:
            if ((x + a) * inv2) % p in Sset:
                bad = True
                break
            if (2 * a - x) % p in Sset:
                bad = True
                break
            if (2 * x - a) % p in Sset:
                bad = True
                break
        if not bad:
            Sset.add(x)

    print(" ".join(map(str, sorted(Sset))))


if __name__ == "__main__":
    main()
