# TIER: trivial
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it)); KINDS = int(next(it)); budget = int(next(it))
    price = []
    for _ in range(T):
        for _ in range(KINDS):
            next(it)
        price.append(int(next(it)))

    # blind, data-independent recipe: spend the whole budget on whichever machine
    # type is cheapest (ties broken by lowest index) -- never looks at the scenarios.
    cheapest = min(range(T), key=lambda t: (price[t], t))
    counts = [0] * T
    counts[cheapest] = max(1, budget // price[cheapest])
    print(" ".join(map(str, counts)))


if __name__ == "__main__":
    main()
