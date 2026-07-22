#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE instance of the crossed-plywood slab decomposition problem.

Instance = a "sculpture" solid built from a rectangular CORE block plus several thin
rectangular ARMS bolted onto its faces, extending along Y or Z.  Sheet size (SW,SH) and
pin budget P are chosen per test.  Deterministic: fully defined by testId (no RNG needed,
but we still seed one for clarity/future extension).
"""
import sys


def make_arm(direction, core_box, offset1, offset2, length, t1, t2, sign):
    """Attach a thin arm to `core_box` on the +face (sign=+1) or -face (sign=-1) along
    `direction` (0=X,1=Y,2=Z).  offset1/offset2 place the arm's footprint within the face,
    measured from the core's own min corner in the two OTHER dimensions."""
    cx0, cy0, cz0, cx1, cy1, cz1 = core_box
    lohi = [[cx0, cx1], [cy0, cy1], [cz0, cz1]]
    others = [i for i in range(3) if i != direction]
    lo = [0, 0, 0]
    hi = [0, 0, 0]
    for k, i in enumerate(others):
        o = offset1 if k == 0 else offset2
        t = t1 if k == 0 else t2
        lo[i] = lohi[i][0] + o
        hi[i] = lo[i] + t
    if sign > 0:
        lo[direction] = lohi[direction][1]
        hi[direction] = lo[direction] + length
    else:
        hi[direction] = lohi[direction][0]
        lo[direction] = hi[direction] - length
    return (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])


def build_case(core_dims, core_pos, arm_specs):
    """arm_specs: list of (direction, off1, off2, length, t1, t2, sign) -> list of boxes."""
    cx0, cy0, cz0 = core_pos
    Cx, Cy, Cz = core_dims
    core_box = (cx0, cy0, cz0, cx0 + Cx, cy0 + Cy, cz0 + Cz)
    boxes = [core_box]
    for (d, o1, o2, length, t1, t2, sign) in arm_specs:
        boxes.append(make_arm(d, core_box, o1, o2, length, t1, t2, sign))
    return boxes


def arm(d, o1, o2, sign, t1=3, t2=3, length=13):
    return (d, o1, o2, length, t1, t2, sign)


CORE = (11, 7, 6)
SW11, SH11 = 11, 11
POS = (20, 20, 20)

TINY_CORE = (7, 4, 4)
POS_TINY = (5, 5, 5)


def case_table():
    cases = {}
    # 1: tiny sanity
    cases[1] = dict(core=TINY_CORE, pos=POS_TINY,
                     arms=[(1, 0, 0, 9, 2, 2, +1)], sw=7, sh=7, p=4)
    # 2: single arm along Z (matches core's true-best axis) -- greedy finds it, trivial doesn't
    cases[2] = dict(core=CORE, pos=POS, arms=[arm(2, 0, 0, +1)], sw=SW11, sh=SH11, p=9)
    # 3: single arm along Y (core's middle axis)
    cases[3] = dict(core=CORE, pos=POS, arms=[arm(1, 0, 0, +1)], sw=SW11, sh=SH11, p=9)
    # 4: single arm along Z, different footprint offset
    cases[4] = dict(core=CORE, pos=POS, arms=[arm(2, 1, 1, +1)], sw=SW11, sh=SH11, p=9)
    # 5: two arms both along Z (pool nicely, both match core's best axis)
    cases[5] = dict(core=CORE, pos=POS,
                     arms=[arm(2, 0, 0, +1), arm(2, 3, 0, -1)], sw=SW11, sh=SH11, p=18)
    # 6: two arms both along Y
    cases[6] = dict(core=CORE, pos=POS,
                     arms=[arm(1, 0, 0, +1), arm(1, 3, 0, -1)], sw=SW11, sh=SH11, p=18)
    # 7: TRAP -- one Y arm + one Z arm: no single global axis can satisfy both
    cases[7] = dict(core=CORE, pos=POS,
                     arms=[arm(1, 0, 0, +1), arm(2, 3, 0, +1)], sw=SW11, sh=SH11, p=9)
    # 8: TRAP -- two Y arms + one Z arm
    cases[8] = dict(core=CORE, pos=POS,
                     arms=[arm(1, 0, 0, +1), arm(1, 3, 0, -1), arm(2, 0, 1, +1)],
                     sw=SW11, sh=SH11, p=18)
    # 9: TRAP, larger -- one Y arm + two Z arms (longer arms)
    cases[9] = dict(core=CORE, pos=POS,
                     arms=[arm(1, 0, 0, +1, 3, 3, 15), arm(2, 3, 0, +1, 3, 3, 15),
                           arm(2, 0, 1, +1, 3, 2, 15)],
                     sw=SW11, sh=SH11, p=9)
    # 10: TRAP, largest/adversarial -- two Y arms + two Z arms (longest)
    cases[10] = dict(core=CORE, pos=POS,
                      arms=[arm(1, 0, 0, +1, 3, 3, 15), arm(1, 3, 0, -1, 3, 3, 15),
                            arm(2, 0, 1, +1, 3, 2, 15), arm(2, 3, 3, -1, 2, 2, 15)],
                      sw=SW11, sh=SH11, p=18)
    return cases


def main():
    testId = int(sys.argv[1])
    cases = case_table()
    c = cases[testId]
    boxes = build_case(c['core'], c['pos'], c['arms'])
    X = max(b[3] for b in boxes)
    Y = max(b[4] for b in boxes)
    Z = max(b[5] for b in boxes)
    out = []
    out.append(f"{X} {Y} {Z}")
    out.append(f"{len(boxes)}")
    for b in boxes:
        out.append(" ".join(str(v) for v in b))
    out.append(f"{c['sw']} {c['sh']}")
    out.append(f"{c['p']}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
