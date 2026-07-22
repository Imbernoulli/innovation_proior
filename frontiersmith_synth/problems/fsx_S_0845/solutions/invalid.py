# TIER: invalid
# Reads the instance but returns an out-of-range winner -- must score 0 on every instance.
import sys
import json


def main():
    json.load(sys.stdin)
    print(json.dumps({"winner": 999}))


if __name__ == "__main__":
    main()
