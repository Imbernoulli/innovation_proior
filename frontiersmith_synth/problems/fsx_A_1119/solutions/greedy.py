# TIER: greedy
"""The obvious textbook approach: multi-source expansion, exactly like
BFS-growing a Voronoi diagram from the seeds. Every active grain boundary
gets an equal share of attention (round robin), so every gap between two
neighboring seeds is split at (approximately) its geometric midpoint.

This ignores the target microstructure's per-gap SKEW entirely -- it never
reads how much extra territory one seed is supposed to win -- so on any gap
where the target favors one side well past the midpoint, this strategy
freezes the disputed band with the wrong grain."""
import sys


def step(heat, solid, orient, cool_idx, F, CSTEP):
    N = len(heat)
    if cool_idx is not None:
        heat[cool_idx] -= CSTEP
    old = heat[:]
    new = heat[:]
    for i in range(N):
        if solid[i]:
            continue
        L = old[i - 1] if (i - 1 >= 0 and not solid[i - 1]) else old[i]
        R = old[i + 1] if (i + 1 < N and not solid[i + 1]) else old[i]
        new[i] = (L + R + 2 * old[i]) // 4
    heat[:] = new
    newly = [i for i in range(N) if not solid[i] and heat[i] <= F]
    assign = {}
    for i in newly:
        dl = dr = None
        ol = orr = None
        j = i - 1
        while j >= 0:
            if solid[j]:
                dl = i - j; ol = orient[j]; break
            j -= 1
        j = i + 1
        while j < N:
            if solid[j]:
                dr = j - i; orr = orient[j]; break
            j += 1
        if dl is None and dr is None:
            assign[i] = 0
        elif dr is None or (dl is not None and dl <= dr):
            assign[i] = ol
        else:
            assign[i] = orr
    for i in newly:
        solid[i] = True
        orient[i] = assign[i]


def frontier_left(solid, seed_pos, N):
    i = seed_pos + 1
    while i < N and solid[i]:
        i += 1
    return i if i < N else None


def frontier_right(solid, seed_pos, N):
    i = seed_pos - 1
    while i >= 0 and solid[i]:
        i -= 1
    return i if i >= 0 else None


def main():
    data = sys.stdin.read().split()
    p = 0
    N = int(data[p]); p += 1
    K = int(data[p]); p += 1
    H0 = int(data[p]); p += 1
    F = int(data[p]); p += 1
    CSTEP = int(data[p]); p += 1
    P = int(data[p]); p += 1
    positions = []
    orients = []
    for _ in range(P):
        pos = int(data[p]); p += 1
        o = int(data[p]); p += 1
        positions.append(pos); orients.append(o)
    # target T is intentionally not consulted -- that is the point.

    heat = [H0] * N
    solid = [False] * N
    orient = [0] * N
    for pos, o in zip(positions, orients):
        solid[pos] = True
        orient[pos] = o

    # cyclic schedule: for every internal gap, alternate its left/right side
    slots = []
    for g in range(P - 1):
        slots.append((g, "L"))
        slots.append((g, "R"))

    out = []
    ptr = 0
    for _ in range(K):
        action = -1
        tries = 0
        while tries < len(slots):
            g, side = slots[ptr % len(slots)]
            ptr += 1
            tries += 1
            pL = positions[g]
            pR = positions[g + 1]
            fl = frontier_left(solid, pL, N)
            fr = frontier_right(solid, pR, N)
            if fl is None or fr is None or fl > fr:
                continue  # this gap is fully claimed
            action = fl if side == "L" else fr
            break
        out.append(str(action))
        step(heat, solid, orient, None if action == -1 else action, F, CSTEP)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
