# TIER: greedy
"""The obvious first-attempt policy: a FIXED HP-THRESHOLD reactive rule. Grind (heal) whenever
current HP drops below a fixed panic fraction of hp_max; otherwise keep descending. It even
"defends" with a ward, but the trigger is its OWN current HP fraction, not the magnitude of the
hazard it's about to walk into -- it never reads the hazard table ahead. So on any instance with
a hazard spike that arrives while HP happens to be comfortably high, it strides straight into a
hit that exceeds hp_max, dies, and forfeits the entire run's banked reward. It also never
front-loads consumables, so on attrition instances it simply runs its stock dry before the
hazard curve gets survivable again. This is a coherent, reasonable-looking heuristic -- it is
just blind to the planted structure (spike floors, drying loot) that the strong solution reads
directly from the input."""
import sys
import json

LOW = 0.45          # grind/heal below this HP fraction
WARD_PANIC = 0.30    # use a ward reactively only when THIS low, regardless of incoming hazard


def main():
    inst = json.load(sys.stdin)
    F = inst["F"]; T = inst["T"]
    hp_max = inst["hp_max"]
    hp = inst["hp_start"]
    potions = inst["potions_start"]
    wards = inst["wards_start"]
    hazard = inst["hazard"]
    grind_loot = inst["grind_loot"]

    depth = 0
    grinded = set()
    actions = []

    for _ in range(T):
        if depth >= F:
            actions.append({"action": "grind"})
            continue
        if hp < LOW * hp_max and potions > 0:
            actions.append({"action": "drink_potion"})
            potions -= 1
            hp = min(hp_max, hp + inst["potion_heal"])
            continue
        if hp < LOW * hp_max:
            actions.append({"action": "grind"})
            hp = min(hp_max, hp + inst["grind_regen"])
            if depth not in grinded:
                grinded.add(depth)
                pg, wg = grind_loot[depth]
                potions += pg
                wards += wg
            continue
        use_ward = (hp < WARD_PANIC * hp_max) and wards > 0
        actions.append({"action": "descend", "use_ward": use_ward})
        dmg = hazard[depth]
        if use_ward:
            wards -= 1
        else:
            hp -= dmg
            if hp <= 0:
                break
        depth += 1

    print(json.dumps(actions))


main()
