# TIER: strong
"""The insight: once a module can no longer improve on the target-attractant field A (it
has already arrived at, or is boxed in among, target cells) it should NOT stop -- it
should keep advancing along the broadcast SEED gradient G, which strictly decreases as
you go deeper into a single-file target region toward the seed placed at its far end.

Two-phase rule, evaluated per local pattern (same universal 6561-entry table as greedy,
so this is still a pure local rule table with no global controller):
  Phase 1 (approach): if any of the 4 neighbours is FREE and its A is strictly lower
    (digit 7 or 8), move there (first match in fixed N,E,S,W order) -- identical to
    greedy while a module is still outside the target region.
  Phase 2 (advance-in-place / conveyor): otherwise, if any neighbour is FREE and its G
    is strictly lower (digit 6 or 8), move there.
  Otherwise STAY.

Because a move into an occupied cell is always blocked, this two-phase rule makes the
FIRST module to reach a single-file target region walk all the way to its deepest still
reachable point (the true dead end, where no neighbour has lower G), then STOP there.
The second module then advances only as far as the first module's current position
minus one, and so on: cells fill from the seed end backward, so nobody ever plugs the
entrance before the interior is reached -- a propagating coordinating signal (G) that
sequences everyone's moves without any module knowing about any other module's plan."""
import sys, json

json.load(sys.stdin)   # this table does not depend on instance geometry

DIRS = ["N", "E", "S", "W"]
table = {}
for a in range(9):
    for b in range(9):
        for c in range(9):
            for d in range(9):
                digits = (a, b, c, d)
                act = "STAY"
                moved = False
                for i, dname in enumerate(DIRS):
                    if digits[i] in (7, 8):     # FREE and A strictly improves
                        act = dname
                        moved = True
                        break
                if not moved:
                    for i, dname in enumerate(DIRS):
                        if digits[i] in (6, 8):  # FREE and G strictly improves
                            act = dname
                            break
                table["".join(str(x) for x in digits)] = act

print(json.dumps({"table": table, "default": "STAY"}))
