# TIER: invalid
# Deliberately malformed answer: "table" is not a dict and "default" is out of range,
# so strict validation must reject this on every instance -> score 0.0 everywhere.
import sys, json


def main():
    json.load(sys.stdin)
    print(json.dumps({"table": "not-a-dict", "default": 999}))


main()
