#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE instance of fsx_B_1060 (Overhang-Supported Additive Print)
to stdout. testId in 1..10 is a growing difficulty ladder; the instance for each testId is a
fixed, deterministic shape (a hardcoded parameter table, no randomness needed).

Instance format:
  line 1: Lx Ly Lz
  line 2: T  (number of target/part voxels)
  next T lines: x y z

Two planted shape families, both a thin 1x1 trunk capped by a WIDE flat overhang that the
trunk alone cannot support -- the textbook "big flat plate cantilevered off a thin stem"
trap:
  mushroom(cx,cy,R,trunk_top): the cap is a SOLID diamond disk of L1-radius R.
  ring_mushroom(cx,cy,R,r_in,trunk_top): the cap is a HOLLOW diamond ring (L1-radius in
    [r_in,R]); the trunk rises straight through the never-printed hole, so support material
    can climb up untrapped before finally paying the trapped cost right under the ring.
"""
import sys


def mushroom(cx, cy, R, trunk_top, H=1):
    """A thin 1x1 trunk of height trunk_top, capped by a SOLID flat diamond disk of
    L1-radius R and thickness H starting at trunk_top. The cap overhangs the trunk by R
    cells in every direction -- decoupling trunk_top from R keeps the naive-vs-clever cost
    gap in a bounded (non-saturating) range across the whole difficulty ladder."""
    T = set()
    for z in range(0, trunk_top):
        T.add((cx, cy, z))
    for z in range(trunk_top, trunk_top + H):
        for dx in range(-R, R + 1):
            rem = R - abs(dx)
            for dy in range(-rem, rem + 1):
                T.add((cx + dx, cy + dy, z))
    return T


def ring_mushroom(cx, cy, R, r_in, trunk_top, H=1):
    """Like mushroom(), but the cap is a HOLLOW ring (L1-radius in [r_in, R]) instead of a
    solid disk. The trunk rises straight through the hole, which is never part-material at
    any height -- so a support column climbing through the hole is NEVER trapped, while
    support placed directly beneath the ring itself always is. This plants a genuine
    trap-vs-untrapped routing choice on top of the sharing/cascading one."""
    T = set()
    for z in range(0, trunk_top):
        T.add((cx, cy, z))
    for z in range(trunk_top, trunk_top + H):
        for dx in range(-R, R + 1):
            rem = R - abs(dx)
            for dy in range(-rem, rem + 1):
                if abs(dx) + abs(dy) >= r_in:
                    T.add((cx + dx, cy + dy, z))
    return T


def bbox_wh(T):
    xs = [c[0] for c in T]; ys = [c[1] for c in T]; zs = [c[2] for c in T]
    return max(xs) + 1, max(ys) + 1, max(zs) + 1


def build(testId):
    # difficulty ladder: solid-disk mushrooms alternate with hollow-ring mushrooms, growing
    # in radius R and trunk height trunk_top. trunk_top is always kept well below R so the
    # naive-per-column cost blows up (cubically) far faster than the reachable, shared,
    # trap-aware support the strong solver finds -- without letting the ratio saturate.
    table = {
        1: ("disk", 3, None, 1),
        2: ("disk", 4, None, 1),
        3: ("ring", 5, 2, 2),
        4: ("disk", 5, None, 2),
        5: ("ring", 6, 3, 2),
        6: ("disk", 6, None, 2),
        7: ("ring", 8, 3, 3),
        8: ("disk", 8, None, 3),
        9: ("ring", 10, 4, 3),
        10: ("disk", 12, None, 4),
    }
    kind, R, r_in, trunk_top = table[testId]
    cx = cy = max(R, trunk_top) + 2
    if kind == "disk":
        T = mushroom(cx, cy, R, trunk_top)
    else:
        T = ring_mushroom(cx, cy, R, r_in, trunk_top)

    Lx, Ly, Lz = bbox_wh(T)
    Lx += 1  # 1 cell of margin so no shape touches the far grid wall
    Ly += 1
    out = [f"{Lx} {Ly} {Lz}", str(len(T))]
    for (x, y, z) in sorted(T):
        out.append(f"{x} {y} {z}")
    return "\n".join(out) + "\n"


def main():
    if len(sys.argv) != 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    testId = int(sys.argv[1])
    sys.stdout.write(build(testId))


if __name__ == "__main__":
    main()
