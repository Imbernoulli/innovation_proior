# TIER: greedy
# The obvious first attempt: plain plurality. Elect whoever has the most first-place votes,
# ties broken by lowest index. Textbook, and exactly what the COMPROMISE bloc recipe is built
# to exploit (it only has to move first-place votes).
import sys
import json


def main():
    inst = json.load(sys.stdin)
    ballots = inst["ballots"]
    m = inst["num_candidates"]
    tally = [0] * m
    for bal in ballots:
        tally[bal[0]] += 1
    winner = max(range(m), key=lambda c: (tally[c], -c))
    print(json.dumps({"winner": winner}))


if __name__ == "__main__":
    main()
