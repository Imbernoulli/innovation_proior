# TIER: trivial
# One rehearsal per scene: earliest day that still has a free, conflict-free
# room, scene-index order, no lookahead at all (no decay/weight awareness).
import sys, json


def main():
    inst = json.load(sys.stdin)
    S = inst["n_scenes"]; D = inst["n_days"]
    rooms = inst["rooms"]; scene_actors = inst["scene_actors"]

    room_used = [0] * D
    actor_used = [set() for _ in range(D)]
    schedule = [[] for _ in range(D)]

    for s in range(S):
        aset = set(scene_actors[s])
        for d in range(D):
            if room_used[d] < rooms[d] and not (aset & actor_used[d]):
                schedule[d].append(s)
                room_used[d] += 1
                actor_used[d] |= aset
                break

    print(json.dumps({"schedule": schedule}))


if __name__ == "__main__":
    main()
