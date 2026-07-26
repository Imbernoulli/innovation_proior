# TIER: invalid
# Deliberately broken: every vertical interval is a scale-degree second
# ((cp - cf) mod 7 == 1), which is never in the consonant class set, so this
# fails the very first hard-rule check on every instance.
import sys, json


def main():
    inst = json.load(sys.stdin)
    cantus = inst["cantus"]
    cp = [c + 1 for c in cantus]
    print(json.dumps({"cp": cp}))


main()
