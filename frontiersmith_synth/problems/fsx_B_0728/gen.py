import sys, random
from collections import defaultdict


def walk(creases, N):
    """Deterministic footprint-routing rule: flap 0 sits at column 0 heading +1.
    A Valley crease keeps the current heading (paper keeps sliding the same way);
    a Mountain crease reflects the heading (paper bounces back over itself)."""
    pos = [0] * N
    d = 1
    for i in range(N - 1):
        if creases[i] == 'V':
            pos[i + 1] = pos[i] + d
        else:
            pos[i + 1] = pos[i] - d
            d = -d
    return pos


def resolve_groups_lowfirst(creases, groups):
    """M-first / low-index tie-break: within each vertex group, whichever free
    creases are needed to hit the parity target are the LOWEST-index free ones."""
    cr = creases[:]
    for idxs, target in groups:
        vals = [cr[i] for i in idxs]
        fixed_sum = sum(1 if v == 'M' else -1 for v in vals if v != '?')
        free = [i for i in idxs if cr[i] == '?']
        nfree = len(free)
        if nfree == 0:
            continue
        need = target - fixed_sum
        m_needed = (need + nfree) // 2
        m_needed = max(0, min(nfree, m_needed))
        chosen = set(sorted(free)[:m_needed])
        for i in free:
            cr[i] = 'M' if i in chosen else 'V'
    return cr


def fill_leftover(creases, groups, default):
    covered = set()
    for idxs, _ in groups:
        covered.update(idxs)
    cr = creases[:]
    for i in range(len(cr)):
        if cr[i] == '?':
            cr[i] = default
    return cr


def main():
    testId = int(sys.argv[1])
    rng = random.Random(9137 + 97 * testId)

    ladder = [12, 18, 24, 30, 36, 42, 48, 54, 60, 66]
    N = ladder[testId - 1]
    trap = testId >= 5

    weights = [rng.randint(1, 9) for _ in range(N)]

    # vertex groups: a chain of size-4 crease groups, consecutive groups sharing
    # exactly one crease (mirrors a degree-4 flat-vertex Maekawa condition chained
    # along the strip). j = 0 .. G-1 while 3j+3 <= N-2.
    groups = []
    j = 0
    while 3 * j + 3 <= N - 2:
        idxs = [3 * j, 3 * j + 1, 3 * j + 2, 3 * j + 3]
        bias = 0.95 if trap else 0.6
        target = 2 if rng.random() < bias else -2
        groups.append((idxs, target))
        j += 1

    creases = ['?'] * (N - 1)
    ref = resolve_groups_lowfirst(creases, groups)
    ref = fill_leftover(ref, groups, 'V')
    ref_pos = walk(ref, N)

    # hinge links: within every ref-column, chain consecutive (by flap index) flaps
    # with a hinge crease. This is laminar by construction (vertex-disjoint paths),
    # so it never creates a noncrossing violation for the planted reference, while
    # remaining a live, checked feasibility rule (coincidence + order + noncrossing).
    by_col = defaultdict(list)
    for i, c in enumerate(ref_pos):
        by_col[c].append(i)
    hinges = []
    for c, flaps in by_col.items():
        flaps.sort()
        for a, b in zip(flaps, flaps[1:]):
            hinges.append([a, b, 'V'])  # convention: V => height[a] < height[b]

    rng.shuffle(hinges)
    for h in hinges:
        if rng.random() < 0.5:
            h[2] = '?'
    hinges.sort(key=lambda h: (h[0], h[1]))

    out = []
    out.append(f"{N} {len(hinges)} {len(groups)}")
    out.append(" ".join(map(str, weights)))
    out.append(" ".join(creases_all_free(N)))
    for a, b, lab in hinges:
        out.append(f"{a} {b} {lab}")
    for idxs, target in groups:
        out.append(" ".join(map(str, idxs)) + f" {target}")
    sys.stdout.write("\n".join(out) + "\n")


def creases_all_free(N):
    return ['?'] * (N - 1)


if __name__ == "__main__":
    main()
