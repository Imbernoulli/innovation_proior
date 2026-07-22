# TIER: greedy
"""Obvious first idea: the historical hot zone is the flagged risk, and the
prevailing wind tells you which way a fire that starts there will run -- so
wall off that single most-likely downwind path. Build a straight firebreak
line starting just outside the hot zone in the wind direction, and if budget
remains after the line reaches the map edge, thicken the same frontier
perpendicular to the wind. This defends exactly the corridor a fire that (a)
starts in the hot zone and (b) blows with the prevailing wind would take --
and does nothing for ignitions elsewhere in the still fully-connected
flammable cluster, or for the (very common) days the wind blows some other
way.
"""
import sys, json


def solve(inst):
    R, C = inst["R"], inst["C"]
    grid = inst["flammable"]
    budget = inst["budget"]
    hz = inst["hint_zone"]
    wind = tuple(inst["wind_bias"])
    cr = (hz["r0"] + hz["r1"]) // 2
    cc = (hz["c0"] + hz["c1"]) // 2

    cells = []
    seen = set()

    def add(r, c):
        if 0 <= r < R and 0 <= c < C and grid[r][c] == 1 and (r, c) not in seen and len(cells) < budget:
            cells.append([r, c])
            seen.add((r, c))
            return True
        return False

    # walk straight downwind from the hot zone boundary
    r, c = cr + wind[0], cc + wind[1]
    while len(cells) < budget and 0 <= r < R and 0 <= c < C:
        add(r, c)
        r += wind[0]
        c += wind[1]

    # thicken the same frontier perpendicular to the wind if budget remains
    if len(cells) < budget:
        perp = [(0, 1), (0, -1)] if wind[0] != 0 else [(1, 0), (-1, 0)]
        base_r, base_c = cr + wind[0], cc + wind[1]
        offset = 1
        stall = 0
        while len(cells) < budget and stall < 4:
            progressed = False
            for pr, pc in perp:
                if len(cells) >= budget:
                    break
                if add(base_r + pr * offset, base_c + pc * offset):
                    progressed = True
            offset += 1
            stall = 0 if progressed else stall + 1

    return cells


inst = json.load(sys.stdin)
print(json.dumps({"cells": solve(inst)}))
