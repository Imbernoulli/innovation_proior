# TIER: greedy
# The obvious first recipe: "the objective is captured light, so weight
# gravitropism and phototropism equally and don't bother with touch" --
# thigmotropism weight is left at 0. Branch budget is spent "where the
# light is": the K brightest sources, branching AT their exact cell, with
# no regard for whether they are actually visible/reachable from there.
# This looks reasonable and does make real progress -- but whenever the
# nearest visible source sits on the far side of a room from the gap it
# needs, photo pulls the tip AWAY from the gap, and with thigmo silent
# there is nothing to correct the detour: the tip wastes its budget
# wandering a room it can't fully solve and never reaches the deeper, more
# valuable rooms strong integration finds.
import sys


def main():
    data = sys.stdin.read().split("\n")
    R, C = map(int, data[0].split())
    STEPS, K = map(int, data[1].split())
    M = int(data[2 + R])
    sources = []
    for i in range(M):
        r, c, b = data[2 + R + 1 + i].split()
        sources.append((int(b), int(r), int(c)))
    sources.sort(reverse=True)  # brightest first

    print("1.0 1.0 0.0")
    picks = sources[:K]
    print(len(picks))
    for (b, r, c) in picks:
        print(r, c)


if __name__ == "__main__":
    main()
