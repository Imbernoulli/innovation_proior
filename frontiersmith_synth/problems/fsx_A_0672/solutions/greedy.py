# TIER: greedy
# The "obvious" recipe: be fair, rehearse whoever has had the fewest
# rehearsals so far. Forward day-by-day, entirely decay/weight-agnostic.
# TRAP: this spreads practice uniformly across the whole calendar, so a
# fast-forgetting scene's early rehearsals decay to near-nothing by opening,
# and the scarce tech-week rooms near opening are not specifically reserved
# for the scenes that most need them.
import sys, json


def main():
    inst = json.load(sys.stdin)
    S = inst["n_scenes"]; D = inst["n_days"]
    rooms = inst["rooms"]; scene_actors = inst["scene_actors"]

    count = [0] * S
    schedule = [[] for _ in range(D)]

    for d in range(D):
        used_actors = set()
        placed = 0
        order = sorted(range(S), key=lambda s: (count[s], s))
        for s in order:
            if placed >= rooms[d]:
                break
            aset = set(scene_actors[s])
            if aset & used_actors:
                continue
            schedule[d].append(s)
            used_actors |= aset
            count[s] += 1
            placed += 1

    print(json.dumps({"schedule": schedule}))


if __name__ == "__main__":
    main()
