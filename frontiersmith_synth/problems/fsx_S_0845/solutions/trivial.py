# TIER: trivial
# Ignores almost all structure: folds the declared first- and second-choice votes into a
# checksum and reduces it mod the candidate count. Does not aim at plurality, Borda, or
# Condorcet criteria at all -- a stand-in for "some unprincipled use of the ballots".
import sys
import json


def main():
    inst = json.load(sys.stdin)
    ballots = inst["ballots"]
    m = inst["num_candidates"]
    h = 0
    for bal in ballots:
        h = (h * 131 + bal[0] * 7 + bal[1]) % 999983
    print(json.dumps({"winner": h % m}))


if __name__ == "__main__":
    main()
