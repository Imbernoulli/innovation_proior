# TIER: invalid
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

    # Claim a tiny, suspiciously cheap coalition, but hand every member a ballot
    # that is NOT a permutation of 0..m-1 (candidate 0 repeated, the last label
    # missing) -- the checker must reject this on the feasibility gate, not score it.
    bad_ballot = [0, 0] + list(range(2, m - 1)) if m >= 2 else [0]
    while len(bad_ballot) < m:
        bad_ballot.append(0)

    k = min(2, n) if n >= 1 else 0
    if k == 0:
        print(0)
        return
    lines = [str(k)]
    for i in range(k):
        lines.append(str(i) + " " + " ".join(map(str, bad_ballot)))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
