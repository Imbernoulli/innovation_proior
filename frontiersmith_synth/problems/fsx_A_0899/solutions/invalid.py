# TIER: invalid
import sys, json


def main():
    json.load(sys.stdin)
    # wrong shape on purpose: "accept" is not a boolean/0/1
    print(json.dumps({"accept": "buy-it", "state": None}))


if __name__ == "__main__":
    main()
