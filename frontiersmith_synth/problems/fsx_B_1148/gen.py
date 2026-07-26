#!/usr/bin/env python3
"""
gen.py <testId> -- stage-blackout-scene-swap instance generator.
Deterministic: all randomness seeded from testId only.

Instance = a sequence of S scene layouts on an N-cell stage plus two wing
buffers (L, R), each a LIFO stack of capacity w. Between consecutive scenes
there is one blackout window with a tick budget. See statement.md for the
full move/scoring model.
"""
import sys


LADDER = {
    1: dict(N=6, M=4, S=4, trap=False),
    2: dict(N=6, M=4, S=4, trap=False),
    3: dict(N=7, M=5, S=5, trap=True),
    4: dict(N=7, M=5, S=5, trap=True),
    5: dict(N=8, M=6, S=6, trap=True),
    6: dict(N=8, M=6, S=6, trap=True),
    7: dict(N=9, M=6, S=7, trap=True),
    8: dict(N=9, M=7, S=7, trap=True),
    9: dict(N=9, M=7, S=8, trap=True),
    10: dict(N=9, M=7, S=8, trap=True),
}


def build_non_trap(rnd, N, M, S):
    """Benign warm-up cases: each scene's on-stage set / cell assignment is
    an independent random choice."""
    scenes = []
    for s in range(S):
        k = rnd.randint(max(1, M - 2), M)          # how many props are on stage
        on_props = rnd.sample(range(1, M + 1), k)
        cells = rnd.sample(range(N), k)
        arr = [0] * N
        for p, c in zip(on_props, cells):
            arr[c] = p
        scenes.append(arr)
    # first scene's off-stage props start distributed across the wings
    on1 = set(p for p in range(1, M + 1) if p in scenes[0])
    off1 = [p for p in range(1, M + 1) if p not in on1]
    return scenes, off1


def build_trap(rnd, N, M, S):
    """Planted trap: two 'shuttle' props (ids 1,2) occupy the two lowest
    cell indices and alternate on/off EVERY window (next-use = 1 window
    away, always). Several 'medium' props (ids 3..M) sit at higher cell
    indices (home cell = p-1) and leave/return on a slower ~3-window cycle
    (next-use = several windows away). Every window therefore evacuates the
    shuttle together with some mediums -- a per-window mover that pushes in
    cell-ascending order buries the soon-needed shuttle under the
    not-soon-needed mediums, forcing expensive excavation on the very next
    window when the shuttle is needed again."""
    home_cell = {p: p - 1 for p in range(1, M + 1)}
    medium_props = list(range(3, M + 1))

    on_set = set()
    on_set.add(1)                      # shuttle1 starts on stage
    for idx, p in enumerate(medium_props):
        if idx % 2 == 0:
            on_set.add(p)               # half the mediums start on stage

    def layout(on):
        arr = [0] * N
        for p in on:
            arr[home_cell[p]] = p
        return arr

    scenes = [layout(on_set)]
    return_at = {}                     # prop -> scene index (1-indexed) it returns
    rotate = 0
    for s in range(2, S + 1):
        # 1) shuttle toggles every window
        if 1 in on_set:
            on_set.discard(1); on_set.add(2)
        else:
            on_set.discard(2); on_set.add(1)
        # 2) snapshot who is eligible to be sent off THIS window (must already
        #    have been on stage before any return below -- a prop that is
        #    only just returning this scene stays visible for >=1 scene)
        eligible = [p for p in medium_props if p in on_set]
        # 3) bring back any medium prop scheduled to return this scene
        for p in list(return_at):
            if return_at[p] == s:
                on_set.add(p)
                del return_at[p]
        # 4) send off a rotating batch of the eligible (already-on) mediums
        send_target = min(4, len(medium_props))
        sent = 0
        tries = 0
        while sent < send_target and eligible and tries < len(medium_props) * 2:
            p = medium_props[rotate % len(medium_props)]
            rotate += 1
            tries += 1
            if p in eligible and p in on_set:
                on_set.discard(p)
                return_at[p] = s + 3   # not needed again for a few windows
                sent += 1
        scenes.append(layout(on_set))

    on1 = set(p for p in range(1, M + 1) if p in scenes[0])
    off1 = [p for p in range(1, M + 1) if p not in on1]
    return scenes, off1


def main():
    test_id = int(sys.argv[1])
    cfg = LADDER.get(test_id, LADDER[10])
    N, M, S, trap = cfg["N"], cfg["M"], cfg["S"], cfg["trap"]
    w = M // 2 + 2

    rnd = __import__("random").Random(20260 + test_id)

    if trap:
        prec = [0] + [M - p + 1 for p in range(1, M + 1)]   # descending w/ cell index
        scenes, off1 = build_trap(rnd, N, M, S)
    else:
        perm = list(range(1, M + 1))
        rnd.shuffle(perm)
        prec = [0] + perm
        scenes, off1 = build_non_trap(rnd, N, M, S)

    # distribute scene-1 off-stage props across the two wings (alternating)
    off1_sorted = sorted(off1)
    L0, R0 = [], []
    for i, p in enumerate(off1_sorted):
        (L0 if i % 2 == 0 else R0).append(p)
    assert len(L0) <= w and len(R0) <= w

    B = 4 * N   # tick budget per blackout window (flat, generous but not unbounded)

    out = []
    out.append(f"{N} {M} {S} {w}")
    out.append(" ".join(str(prec[p]) for p in range(1, M + 1)))
    out.append(f"{len(L0)} " + " ".join(map(str, L0)))
    out.append(f"{len(R0)} " + " ".join(map(str, R0)))
    for arr in scenes:
        out.append(" ".join(map(str, arr)))
    out.append(" ".join(str(B) for _ in range(S - 1)))
    print("\n".join(out))


if __name__ == "__main__":
    main()
