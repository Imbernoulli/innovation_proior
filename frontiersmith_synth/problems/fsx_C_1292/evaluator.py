#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_1292 -- "Dungeon Descent: Consumables Buy Depth"
(family: dungeon-descent-policy; eval_form: quality-metric; wave3-lens:policy-simulator shape).

A deterministic roguelike dungeon has floors 1..F of escalating threat. A candidate submits
a single JSON ACTION SEQUENCE (its full expedition plan, computed once from the fully-visible
instance data) rather than being driven turn-by-turn -- but the plan is only good if it reacts
correctly to the state it will pass through, i.e. it must effectively encode a DEPTH POLICY.

Mechanics composed into one objective:
  - risk-of-ruin: dying (HP <= 0) at any point zeroes the ENTIRE run's banked reward, no matter
    how deep you got first. There is no partial credit for an unsurvived expedition.
  - resource-attrition: potions/wards are a finite starting stock, replenished only by GRINDING
    (and only the FIRST grind at each floor yields loot -- loot dries up with repetition).
  - depth-reward-scaling: the reward banked for surviving to depth d grows CONVEX-superlinearly
    in d, so depth is worth disproportionately more the deeper you (safely) bank it.

The INSIGHT the strong solution exploits: your current stock of potions+wards, converted
correctly (wards absorb hits no amount of healing can survive; potions/grinding smooth out the
rest), directly determines how deep you can go and SURVIVE to bank the reward. Racing to the
deepest floor as fast as possible walks into an unmitigated lethal hit (banks ZERO). Grinding
defensively and never risking a descent banks almost nothing either (reward(0) or reward(small)
is tiny). The correct policy paces itself against the hazard curve using the resources on hand.

Candidate contract (isolated stdin -> stdout program):
  stdin:  the PUBLIC instance (all floor data -- nothing is hidden from the planner; the
          challenge is planning quality, not missing information).
  stdout: a JSON list of actions, each one of
            {"action": "descend", "use_ward": true|false}
            {"action": "drink_potion"}
            {"action": "grind"}
          applied in order, one per turn, until HP <= 0 (ruin), the list is exhausted, or the
          instance's turn budget T is reached.

Score of one instance: if the run ends alive, banked = reward[depth_reached]; if it ends dead,
banked = 0 (total ruin). This is affinely anchored against the evaluator's OWN "always descend,
never use an item" reference run (base) and a reward-horizon cap that lies STRICTLY beyond the
deepest floor that exists (so no run can ever bank the cap -- that's the score headroom):

    r = clamp( 0.1 + 0.9 * (banked - base) / max(cap - base, 1.0), 0, 1 )

so exactly reproducing the evaluator's own reckless-descend baseline maps to 0.1. Any
structurally invalid / infeasible action (unknown action name, drinking a potion you don't have,
warding with no wards, non-bool use_ward, non-list answer) makes the WHOLE run rejected -> 0.0
for that instance. The final Ratio is the ARITHMETIC MEAN of the per-instance r's.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean of per-instance r, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys
import json

import isorun

CAND_TIMEOUT = 20
MAX_ACTIONS = 400          # generous vs any T used below; guards against payload abuse


# ============================== instance construction =======================
def _build_hazard(F, base, alpha, spikes):
    hz = [round(base * (1 + alpha * i)) for i in range(F)]
    for idx, val in spikes:
        hz[idx] = val
    return hz


def _build_reward(horizon, rbase, rpow):
    """reward[0] = 0; reward[d] = round(rbase * d**rpow) for d = 1..horizon."""
    return [0] + [round(rbase * (d ** rpow)) for d in range(1, horizon + 1)]


def _build_loot(F, loot_hi, pg, wg):
    """Loot available on the FIRST grind at each floor; zero from loot_hi onward (attrition)."""
    return [[pg, wg] if i < loot_hi else [0, 0] for i in range(F)]


HP_MAX = 100
POTION_HEAL = 34
GRIND_REGEN = 16

# 10 archetypes. >=3 are deliberately engineered "spike"/"attrition" traps where a fixed
# HP-threshold reactive policy (our `greedy`) strolls in at high HP and is instantly killed
# by a hit that exceeds hp_max regardless of HP fraction, OR runs dry of consumables because it
# never front-loaded resource stock before the loot dried up.
_ARCHETYPES = [
    dict(name="steady_climb", F=26, T=30, potions=3, wards=1,
         hz_base=6, hz_alpha=0.10, spikes=[],
         loot_hi=26, pg=1, wg=0, rbase=6, rpow=1.55, rhorizon_extra_pct=0.30, seed=910001),
    dict(name="single_spike_early", F=26, T=28, potions=3, wards=1,
         hz_base=7, hz_alpha=0.06, spikes=[(7, 140)],
         loot_hi=26, pg=1, wg=0, rbase=6, rpow=1.55, rhorizon_extra_pct=0.30, seed=910002),
    dict(name="double_spike_deep", F=30, T=32, potions=3, wards=2,
         hz_base=7, hz_alpha=0.05, spikes=[(11, 130), (23, 150)],
         loot_hi=30, pg=1, wg=0, rbase=5, rpow=1.55, rhorizon_extra_pct=0.25, seed=910003),
    dict(name="loot_drought", F=26, T=30, potions=2, wards=1,
         hz_base=8, hz_alpha=0.10, spikes=[(18, 120)],
         loot_hi=6, pg=1, wg=1, rbase=6, rpow=1.55, rhorizon_extra_pct=0.30, seed=910004),
    dict(name="reward_backloaded", F=28, T=26, potions=3, wards=1,
         hz_base=7, hz_alpha=0.07, spikes=[(16, 135)],
         loot_hi=28, pg=1, wg=0, rbase=1, rpow=2.2, rhorizon_extra_pct=0.30, seed=910005),
    dict(name="triple_spike_scarce_ward", F=30, T=30, potions=3, wards=1,
         hz_base=6, hz_alpha=0.05, spikes=[(9, 110), (17, 145), (25, 160)],
         loot_hi=30, pg=1, wg=0, rbase=5, rpow=1.6, rhorizon_extra_pct=0.25, seed=910006),
    dict(name="turn_tight", F=24, T=18, potions=3, wards=1,
         hz_base=8, hz_alpha=0.08, spikes=[(12, 120)],
         loot_hi=24, pg=1, wg=0, rbase=7, rpow=1.5, rhorizon_extra_pct=0.30, seed=910007),
    dict(name="ward_scarce_potion_rich", F=27, T=30, potions=5, wards=1,
         hz_base=7, hz_alpha=0.06, spikes=[(14, 150)],
         loot_hi=27, pg=1, wg=0, rbase=6, rpow=1.55, rhorizon_extra_pct=0.30, seed=910008),
    dict(name="easy_gentle", F=22, T=26, potions=3, wards=1,
         hz_base=5, hz_alpha=0.08, spikes=[],
         loot_hi=22, pg=1, wg=0, rbase=8, rpow=1.5, rhorizon_extra_pct=0.30, seed=910009),
    dict(name="hard_combo", F=32, T=30, potions=2, wards=2,
         hz_base=7, hz_alpha=0.06, spikes=[(10, 120), (20, 140), (28, 155)],
         loot_hi=8, pg=1, wg=1, rbase=4, rpow=1.65, rhorizon_extra_pct=0.25, seed=910010),
]


def make_instances():
    out = []
    for sp in _ARCHETYPES:
        F = sp["F"]
        horizon = F + max(2, round(F * sp["rhorizon_extra_pct"]))
        hazard = _build_hazard(F, sp["hz_base"], sp["hz_alpha"], sp["spikes"])
        reward_full = _build_reward(horizon, sp["rbase"], sp["rpow"])   # index 0..horizon
        grind_loot = _build_loot(F, sp["loot_hi"], sp["pg"], sp["wg"])
        public = {
            "F": F,
            "T": sp["T"],
            "hp_max": HP_MAX,
            "hp_start": HP_MAX,
            "potions_start": sp["potions"],
            "wards_start": sp["wards"],
            "potion_heal": POTION_HEAL,
            "grind_regen": GRIND_REGEN,
            "hazard": hazard,
            "reward": reward_full[:F + 1],          # only the REACHABLE part is public
            "grind_loot": grind_loot,
            "seed": sp["seed"],
        }
        hidden = {"cap": reward_full[horizon]}
        out.append({"name": sp["name"], "public": public, "hidden": hidden})
    return out


# ================================ simulation ==================================
def simulate(inst_public, actions):
    """Replay `actions` deterministically against the instance. Returns (died, depth) or
    None if the action stream is structurally/semantically invalid (any violation -> reject)."""
    if not isinstance(actions, list) or len(actions) > MAX_ACTIONS:
        return None
    F = inst_public["F"]; T = inst_public["T"]
    hp = inst_public["hp_start"]; hp_max = inst_public["hp_max"]
    potions = inst_public["potions_start"]; wards = inst_public["wards_start"]
    hazard = inst_public["hazard"]; grind_loot = inst_public["grind_loot"]
    depth = 0; turns = 0; grinded = set()
    for act in actions:
        if turns >= T:
            break
        if not isinstance(act, dict):
            return None
        a = act.get("action")
        if a == "descend":
            uw = act.get("use_ward", False)
            if not isinstance(uw, bool):
                return None
            if depth >= F:
                turns += 1
                continue
            dmg = hazard[depth]
            if uw:
                if wards <= 0:
                    return None
                wards -= 1
                dmg = 0
            hp -= dmg
            turns += 1
            if hp <= 0:
                return {"died": True, "depth": depth}
            depth += 1
        elif a == "drink_potion":
            if potions <= 0:
                return None
            potions -= 1
            hp = min(hp_max, hp + inst_public["potion_heal"])
            turns += 1
        elif a == "grind":
            hp = min(hp_max, hp + inst_public["grind_regen"])
            if depth < F and depth not in grinded:
                grinded.add(depth)
                pg, wg = grind_loot[depth]
                potions += pg
                wards += wg
            turns += 1
        else:
            return None
    return {"died": False, "depth": depth}


_RECKLESS_CACHE = {}


def baseline(inst):
    """The evaluator's OWN reference: 'always descend, never grind/heal/ward'. Computed fresh
    from the public instance -- the 0.1 anchor is whatever this reckless run banks (often 0)."""
    key = inst["name"]
    if key not in _RECKLESS_CACHE:
        actions = [{"action": "descend"} for _ in range(inst["public"]["T"])]
        res = simulate(inst["public"], actions)
        banked = 0 if (res is None or res["died"]) else inst["public"]["reward"][res["depth"]]
        _RECKLESS_CACHE[key] = banked
    return _RECKLESS_CACHE[key]


def score(inst, answer):
    """Returns (ok, banked_reward)."""
    res = simulate(inst["public"], answer)
    if res is None:
        return False, 0.0
    banked = 0.0 if res["died"] else float(inst["public"]["reward"][res["depth"]])
    return True, banked


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = make_instances()

    vec = []
    for inst in instances:
        b = baseline(inst)
        cap = inst["hidden"]["cap"]
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=CAND_TIMEOUT)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False; obj = 0.0
        if not ok:
            vec.append(0.0)
            continue
        denom = max(cap - b, 1.0)
        r = 0.1 + 0.9 * (obj - b) / denom
        if not (r == r):     # NaN guard
            r = 0.0
        r = max(0.0, min(1.0, r))
        vec.append(float(r))

    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
