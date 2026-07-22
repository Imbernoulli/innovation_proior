# TIER: invalid
import sys, json


def main():
    inst = json.load(sys.stdin)
    # Deliberately malformed: wrong length AND non-integer entries.
    print(json.dumps({"order": [0.5, 1.5, "3"]}))


main()
