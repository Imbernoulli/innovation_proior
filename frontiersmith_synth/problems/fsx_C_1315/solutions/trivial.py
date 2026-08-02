# TIER: trivial
# "Just point everyone at the exit closest to them, forever." The most obvious
# possible answer to "where should people go" -- never reroutes, so guided ==
# default and the compliance/credibility machinery is moot (nothing ever
# changes, nothing is ever contradicted).
import sys, json


def main():
    inst = json.load(sys.stdin)
    Z, T = inst["n_zones"], inst["T"]
    de = inst["default_exit"]
    guidance = [[de[i]] * T for i in range(Z)]
    print(json.dumps({"guidance": guidance}))


main()
