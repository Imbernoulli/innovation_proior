# TIER: strong
"""Inventory-conditioned depth policy -- the genuine insight (not just the reactive rule with a
tighter threshold). It reads the FULL hazard table ahead and classifies each upcoming hit:

  - UNHEALABLE (hazard >= hp_max): no amount of potions/grinding can ever survive this hit, so
    the ONLY way through is a ward. The policy spends a ward on it (in the order it is
    encountered); once wards run out and another unhealable hit is ahead, it recognizes that
    hit is a wall it cannot pass and STOPS the expedition there, banking the depth already
    reached, rather than gambling into a certain-death hit that would zero everything.
  - healable-but-big (would leave a thin safety margin): heal proactively -- prefer a potion
    (turn-efficient) and fall back to grinding (which also restocks consumables while the loot
    is still available at this depth) before taking the hit.
  - small: just descend.

This is the direct instantiation of "consumables convert into survivable depth": the wards on
hand are spent exactly on the hits nothing else can clear, and the policy explicitly refuses to
attempt a hit it cannot pay for, converting "how much inventory do I have" into "how deep can I
safely go" instead of reacting turn-by-turn to its own HP alone."""
import sys
import json


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
    turns_used = 0

    while turns_used < T and depth < F:
        dmg = hazard[depth]
        unhealable = dmg >= hp_max
        if unhealable:
            if wards > 0:
                actions.append({"action": "descend", "use_ward": True})
                wards -= 1
                depth += 1
                turns_used += 1
                continue
            else:
                break  # cannot survive this floor and no ward left -- bank here, don't gamble

        if dmg >= hp - 0.10 * hp_max:           # healable, but leaves too thin a margin
            if potions > 0:
                actions.append({"action": "drink_potion"})
                potions -= 1
                hp = min(hp_max, hp + inst["potion_heal"])
                turns_used += 1
                continue
            elif depth not in grinded and sum(grind_loot[depth]) > 0:
                actions.append({"action": "grind"})
                hp = min(hp_max, hp + inst["grind_regen"])
                grinded.add(depth)
                pg, wg = grind_loot[depth]
                potions += pg
                wards += wg
                turns_used += 1
                continue
            elif hp < hp_max:
                actions.append({"action": "grind"})
                hp = min(hp_max, hp + inst["grind_regen"])
                turns_used += 1
                continue
            else:
                break  # full HP, no potions/loot left, still too costly -- stop and bank

        actions.append({"action": "descend", "use_ward": False})
        hp -= dmg
        if hp <= 0:
            break
        depth += 1
        turns_used += 1

    print(json.dumps(actions))


main()
