#!/usr/bin/env python3
"""gen.py <testId> -> one instance on stdout.

Winterizing a drafty monastery. The building is a thermal resistor network:
rooms are nodes, walls are edges with a base resistance R0. Adding k insulation
layers to a wall raises its resistance to R0 + k*rho (series thermal resistance).
Some walls are stained-glass windows that CANNOT be insulated (kmax = 0).

Occupied rooms (the friars' cells / the scriptorium) are held at a fixed target
temperature by thermostats; unheated buffer rooms (cloisters, the nave) float.
Outside is fixed cold. Heat leaks from occupied rooms out to the cold, possibly
THROUGH a leaky buffer room whose stained-glass window shorts it to outside.

The instances plant the trap of the family: the buffer rooms carry an
uninsulatable window, so pouring insulation onto exterior walls (what an energy
audit flags) leaves the interior doorway as the open short-circuit.
"""
import sys, random

def emit(testId):
    rng = random.Random(90594 * 131 + testId * 977)
    u = (testId + 1) // 2                 # number of occupied/buffer units: 1..5
    use_corridor = (u >= 3)               # shared drafty corridor couples units

    Tstar = 20
    Tout = -20

    rooms = []           # room ids are assigned 1..N; 0 is the outside
    occ = []             # occupied flag per room id (index by id-1)
    walls = []           # (a, b, R0, rho, kmax)

    def new_room(is_occ):
        rid = len(rooms) + 1
        rooms.append(rid)
        occ.append(1 if is_occ else 0)
        return rid

    R0d_choices  = ["0.40", "0.50", "0.60"]      # interior door base resistance (low = leaky)
    rhod_choices = ["0.40", "0.50"]
    R0oe_choices = ["0.80", "1.00", "1.20"]      # occupied exterior wall
    R0w_choices  = ["0.30", "0.35", "0.45", "0.70"]  # window (uninsulatable); low = big short
    R0be_choices = ["0.90", "1.10"]              # buffer exterior wall

    buffers = []
    for _ in range(u):
        O = new_room(True)
        B = new_room(False)
        buffers.append(B)
        R0d = rng.choice(R0d_choices)
        rhod = rng.choice(rhod_choices)
        R0oe = rng.choice(R0oe_choices)
        R0w = rng.choice(R0w_choices)
        R0be = rng.choice(R0be_choices)
        # interior doorway  O <-> B  (insulatable) -- the wall the audit never flags
        walls.append((O, B, R0d, rhod, 8))
        # occupied room's own exterior wall  O <-> outside (insulatable) -- the audit's pick
        walls.append((O, 0, R0oe, "0.50", 8))
        # buffer's stained-glass window  B <-> outside (UNINSULATABLE short-circuit)
        walls.append((B, 0, R0w, "0.00", 0))
        # buffer's exterior wall  B <-> outside (insulatable)
        walls.append((B, 0, R0be, "0.50", 8))

    if use_corridor:
        C = new_room(False)
        # corridor's own drafty window to outside (uninsulatable parallel leak)
        walls.append((C, 0, rng.choice(["0.50", "0.60"]), "0.00", 0))
        # each buffer opens onto the shared corridor via an interior arch (insulatable)
        for B in buffers:
            walls.append((B, C, rng.choice(["0.60", "0.80"]), "0.50", 8))

    N = len(rooms)
    M = len(walls)
    K = 5 * u                              # layer budget (contested)

    out = []
    out.append("%d %d %d %d %d" % (N, M, K, Tstar, Tout))
    out.append(" ".join(str(x) for x in occ))
    for (a, b, R0, rho, kmax) in walls:
        out.append("%d %d %s %s %d" % (a, b, R0, rho, kmax))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    emit(int(sys.argv[1]))
