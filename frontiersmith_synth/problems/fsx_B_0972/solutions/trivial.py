# TIER: trivial
# Never coin any shorthand -- transcribe every line exactly as dictated.
# Always valid (cost == the raw baseline), never exploits any structure.
import sys, json


def main():
    inst = json.load(sys.stdin)
    lines = inst["lines"]
    rewrites = [list(line) for line in lines]
    print(json.dumps({"new_macros": [], "rewrites": rewrites}))


main()
