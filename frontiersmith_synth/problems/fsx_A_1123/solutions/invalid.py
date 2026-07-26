# TIER: invalid
# Deliberately broken: emits duplicate anchor indices (fails the distinctness
# check) so every instance must score 0.
import sys, json


def main():
    inst = json.load(sys.stdin)
    k = inst["k"]
    print(json.dumps({"anchors": [0] * k}))


if __name__ == "__main__":
    main()
