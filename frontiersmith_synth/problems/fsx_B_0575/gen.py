#!/usr/bin/env python3
"""gen.py <testId> -> one instance on stdout.

Instance = a voxel solid ("3D print") plus a penalty weight LAM.  We plant:
  * CAGES: hollow cubic shells (all 6 walls solid) with a single full-height 1-wide
    SLOT carved into one wall, and a single floating solid "blob" near the interior top.
    The blob overhangs and needs support down to the shell floor; that support can only
    be *extracted* by lining it up with the slot (line-of-sight) -- and the slot is only
    lateral (usable) for some build orientations.  The blob is deliberately OFF the slot
    line, so a straight vertical support column is trapped; an insightful solver slopes
    the column over to the slot line to free it.
  * A tall open PILLAR (a small floating plate over open ground): its support volume is
    large in the given orientation and shrinks when the print is laid on its side.  This
    is the bait that makes the raw-material-minimising heuristic reorient -- into a build
    direction whose slots point the wrong way, trapping the cage supports.
Deterministic: everything is a pure function of testId.
"""
import sys

def add_box_shell(solid, ox, oy, oz, c):
    """Hollow cube: outer side c+2 at origin (ox,oy,oz); all 6 faces solid, interior empty."""
    s = c + 2
    for dz in range(s):
        for dy in range(s):
            for dx in range(s):
                if dx in (0, s - 1) or dy in (0, s - 1) or dz in (0, s - 1):
                    solid.add((ox + dx, oy + dy, oz + dz))

def carve_slot(solid, ox, oy, oz, c, wall, pos):
    """Remove a full-height (along z) 1-wide slot from one lateral wall of the shell."""
    s = c + 2
    if wall == '-x':
        for dz in range(1, s - 1):
            solid.discard((ox + 0, oy + 1 + pos, oz + dz))
    elif wall == '+x':
        for dz in range(1, s - 1):
            solid.discard((ox + s - 1, oy + 1 + pos, oz + dz))
    elif wall == '-y':
        for dz in range(1, s - 1):
            solid.discard((ox + 1 + pos, oy + 0, oz + dz))
    elif wall == '+y':
        for dz in range(1, s - 1):
            solid.discard((ox + 1 + pos, oy + s - 1, oz + dz))

def add_blob(solid, ox, oy, oz, c, bx, by):
    """Single floating solid voxel near interior top (overhang needing support)."""
    solid.add((ox + 1 + bx, oy + 1 + by, oz + c))  # top interior layer

def gen(t):
    # ---- difficulty ladder ----
    if t <= 2:
        c, ncages, H = 3, 1, 3
    elif t <= 4:
        c, ncages, H = 3, 2, 4
    elif t <= 6:
        c, ncages, H = 4, 2, 5
    elif t <= 8:
        c, ncages, H = 4, 3, 6
    else:
        c, ncages, H = 5, 4, 7
    LAM = 6
    s = c + 2

    walls = ['-x', '+y', '+x', '-y']
    solid = set()

    # cages laid in a row along x, sharing y,z=0 origin
    gap = 1
    x_cursor = 0
    for j in range(ncages):
        ox, oy, oz = x_cursor, 0, 0
        wall = walls[j % len(walls)]
        # slot position and blob offset chosen so the straight column is OFF the slot line
        slot_pos = (j * 2 + 1) % c
        if wall in ('-x', '+x'):
            # slot varies along y; blob offset in y set 1 or 2 cells off the slot line
            by = (slot_pos + 2) % c
            bx = c // 2
        else:
            bx = (slot_pos + 2) % c
            by = c // 2
        add_box_shell(solid, ox, oy, oz, c)
        carve_slot(solid, ox, oy, oz, c, wall, slot_pos)
        add_blob(solid, ox, oy, oz, c, bx, by)
        x_cursor += s + gap

    # ---- pillar zone: an open tall plate whose support is bulky in the given orientation ----
    # placed after the cages along x, in its own open lane.
    px0 = x_cursor + 1
    plate_w = 2
    Z = max(s, s + H)
    plate_z = Z - 1
    # a plate_w x plate_w plate floating at z=plate_z over open ground (empty below to plate)
    for ax in range(plate_w):
        for ay in range(plate_w):
            solid.add((px0 + ax, 1 + ay, plate_z))

    # overall dims
    X = px0 + plate_w + 1
    Y = max(s, plate_w + 2)
    Zdim = Z

    # Present the print in its WORST build orientation as the default frame (mirror z): the
    # given plate direction forces the shell's roof + blob to overhang all the way down, so a
    # naive "just print it as given" baseline is expensive.  The insight is to lay the part on
    # a different face; the raw-material heuristic reorients too, but into a face whose slots
    # point away from any extraction ray.
    solid = set((x, y, Zdim - 1 - z) for (x, y, z) in solid)

    # ---- render ----
    out = ["%d %d %d %d" % (X, Y, Zdim, LAM)]
    for z in range(Zdim):
        for y in range(Y):
            row = "".join("1" if (x, y, z) in solid else "0" for x in range(X))
            out.append(row)
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    gen(t)
