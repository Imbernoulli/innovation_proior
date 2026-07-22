# TIER: invalid
import sys


def main():
    data = sys.stdin.read().split("\n")
    _n = int(data[0])
    # rule 0 illegally references rule 1, which does not exist yet (forward reference) ->
    # must be rejected by the acyclicity check regardless of the actual string.
    print(1)
    print("r1")


if __name__ == "__main__":
    main()
