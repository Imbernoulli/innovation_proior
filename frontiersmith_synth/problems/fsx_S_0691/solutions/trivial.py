# TIER: trivial
# Do nothing: identity rule everywhere. The seed cell stays put; nothing ever grows.
import sys, json


def main():
    json.load(sys.stdin)  # read (and ignore) the public instance
    print(json.dumps({"table": {}, "default": "stay"}))


main()
