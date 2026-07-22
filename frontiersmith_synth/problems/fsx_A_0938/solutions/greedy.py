# TIER: greedy
"""The obvious first idea: pure local gradient descent on the target-attractant field A
(distance to the nearest target cell). For every one of the 6561 possible 4-digit local
patterns (N,E,S,W order), move toward the first direction (fixed priority N,E,S,W) whose
neighbour is FREE and strictly reduces A; otherwise STAY. The seed field G (the
"broadcast gradient" of the family) is read from the instance but never used -- this is
exactly the recipe an average strong coder writes first: "always step toward the target,
ignore any other signal".

This recipe works fine when a module's own approach lane never interacts with another
module's. It jams badly whenever the target region is more than one cell deep in the
approach direction (a corridor, or the interior of any solid block): once ANY module
reaches ANY target cell its own A is already 0 (the global minimum), so no neighbour can
ever look "better" under this rule, and the module freezes there forever -- permanently
blocking every module behind it in a single-file region."""
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
                for i, dname in enumerate(DIRS):
                    if digits[i] in (7, 8):   # FREE and A strictly improves
                        act = dname
                        break
                table["".join(str(x) for x in digits)] = act

print(json.dumps({"table": table, "default": "STAY"}))
