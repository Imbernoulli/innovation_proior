# TIER: invalid
# Schedules every scene in every day, ignoring room caps and actor conflicts
# entirely -> infeasible on every real instance.
import sys, json


def main():
    inst = json.load(sys.stdin)
    S = inst["n_scenes"]; D = inst["n_days"]
    schedule = [list(range(S)) for _ in range(D)]
    print(json.dumps({"schedule": schedule}))


if __name__ == "__main__":
    main()
