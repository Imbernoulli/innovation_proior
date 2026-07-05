# TIER: invalid
# A self-intersecting "bowtie" loop over the four corner stations (0,0)->(S,S)->
# (S,0)->(0,S): the two diagonals cross, so the polygon is NOT simple and every
# instance is rejected -> 0.0.
import sys, json


def main():
    json.load(sys.stdin)
    print(json.dumps({"tour": [0, 2, 1, 3]}))


main()
