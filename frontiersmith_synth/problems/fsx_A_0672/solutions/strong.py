# TIER: strong
# Insight: only decay-discounted proximity to OPENING NIGHT matters, so build
# the schedule BACKWARD from the deadline. Walk days from opening night
# backward; each day greedily seat the conflict-free scenes with the highest
# decay-discounted marginal value (capped by remaining need) for THAT day.
# This automatically reserves the valuable late days for fast-forgetting /
# high-weight scenes, and lets bottleneck (heavy shared-actor) scenes fall
# back onto the roomier, less-contested early days as leftover capacity --
# exactly the resource-bottleneck-aware backward threading the family wants.
import sys, json, math


def main():
    inst = json.load(sys.stdin)
    S = inst["n_scenes"]; D = inst["n_days"]
    rooms = inst["rooms"]; scene_actors = inst["scene_actors"]
    decay = inst["decay"]; boost = inst["boost"]; cap = inst["cap"]

    recall = [0.0] * S
    schedule = [[] for _ in range(D)]
    actor_sets = [set(a) for a in scene_actors]

    for d in range(D - 1, -1, -1):           # reverse chronological: closest to opening first
        gap = D - d
        used_actors = set()
        room_left = rooms[d]
        remaining = [s for s in range(S) if recall[s] < cap[s] - 1e-12]
        while room_left > 0 and remaining:
            best = None; best_val = -1.0
            for s in remaining:
                if actor_sets[s] & used_actors:
                    continue
                val = min(cap[s] - recall[s], boost[s] * math.exp(-decay[s] * gap))
                if val > best_val:
                    best_val = val; best = s
            if best is None:
                break
            schedule[d].append(best)
            used_actors |= actor_sets[best]
            recall[best] += best_val
            remaining.remove(best)
            room_left -= 1

    print(json.dumps({"schedule": schedule}))


if __name__ == "__main__":
    main()
