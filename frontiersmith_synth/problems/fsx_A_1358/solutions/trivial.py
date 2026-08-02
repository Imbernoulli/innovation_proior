# TIER: trivial
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    m = int(next(it)); n = int(next(it)); rule_type = int(next(it)); target = int(next(it))
    if rule_type == 0:
        for _ in range(m):
            next(it)
    for _ in range(n):
        for _ in range(m):
            next(it)

    # Reproduce the checker's own baseline: EVERYONE in the coalition, all rewritten
    # to the identical ballot [target, then every other candidate in ascending index
    # order]. This always makes target the unique winner, but uses the whole
    # electorate -- the "do nothing clever" reference point.
    others = [c for c in range(m) if c != target]
    ballot = [target] + others

    lines = [str(n)]
    for i in range(n):
        lines.append(str(i) + " " + " ".join(map(str, ballot)))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
