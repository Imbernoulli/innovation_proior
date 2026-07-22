# TIER: trivial
import sys, json


def main():
    inst = json.load(sys.stdin)
    S = inst["S"]; K = inst["K"]
    share = K / S
    alloc = [share] * S
    print(json.dumps({"alloc": alloc}))


if __name__ == "__main__":
    main()
