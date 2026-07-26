# TIER: strong
"""The insight: grain territory is decided by WHICH front's cold reaches a
contested cell first, not by "effort spent" in the abstract -- so treat the
single per-stage cooling action as a scarce resource that must be committed,
in full, to one grain boundary at a time (temporal commitment), in the order
that actually realizes the target microstructure.

For every gap between two neighboring seeds we read the target array to see
exactly how many cells on each side the target grain is supposed to claim
(this is planted structure the round-robin/"fair" approach never looks at).
We then race that gap by spending EVERY stage on the side that still owes
territory, fully finishing one side's required advance before ever touching
the other side of the same gap -- rather than splitting attention between the
two fronts, which only ever reaches the geometric midpoint."""
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
    T = [int(data[p + i]) for i in range(N)]
    p += N

    heat = [H0] * N
    solid = [False] * N
    orient = [0] * N
    for pos, o in zip(positions, orients):
        solid[pos] = True
        orient[pos] = o

    # per-gap required left-side share, read directly off the target array:
    # the split point is where T stops being the left seed's orientation.
    need_left = []
    for g in range(P - 1):
        pL, oL = positions[g], orients[g]
        pR, oR = positions[g + 1], orients[g + 1]
        gapwidth = pR - pL - 1
        cnt = 0
        for c in range(pL + 1, pR):
            if T[c] == oL:
                cnt += 1
            else:
                break
        need_left.append(cnt)

    out = []
    for g in range(P - 1):
        pL = positions[g]
        pR = positions[g + 1]
        Lneed = need_left[g]
        guard = 0
        # commit fully to the left side until it owns exactly Lneed cells
        while len(out) < K:
            fl = frontier_left(solid, pL, N)
            fr = frontier_right(solid, pR, N)
            if fl is None or fr is None or fl > fr:
                break
            cur_L = fl - (pL + 1)  # cells already solid on the left side
            if cur_L < Lneed:
                action = fl
            else:
                action = fr
            out.append(str(action))
            step(heat, solid, orient, action, F, CSTEP)
            guard += 1
            if guard > K:
                break
        if len(out) >= K:
            break

    while len(out) < K:
        out.append("-1")
        step(heat, solid, orient, None, F, CSTEP)

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
