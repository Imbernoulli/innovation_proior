import sys

MOD = 998244353

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    L = int(data[idx]); idx += 1
    R = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    feud = set()
    for _ in range(m):
        u = int(data[idx]); idx += 1
        v = int(data[idx]); idx += 1
        feud.add((u, v))
        feud.add((v, u))

    elems = list(range(n))

    # A team is legal if size in [L,R] and no feuding pair inside.
    def team_ok(team):
        s = len(team)
        if s < L or s > R:
            return False
        for i in range(len(team)):
            for j in range(i + 1, len(team)):
                if (team[i], team[j]) in feud:
                    return False
        return True

    # Enumerate all set partitions of {0..n-1}; count those where every block is a legal team.
    # Canonical recursive partition enumeration (each partition produced exactly once).
    count = 0

    def gen(items):
        # yields all set partitions of the list `items` as a list of blocks (frozensets)
        if not items:
            yield []
            return
        first = items[0]
        rest = items[1:]
        # choose which of the rest join `first`'s block
        k = len(rest)
        for sub in range(1 << k):
            block = [first] + [rest[t] for t in range(k) if (sub >> t) & 1]
            remaining = [rest[t] for t in range(k) if not ((sub >> t) & 1)]
            for tail in gen(remaining):
                yield [block] + tail

    for partition in gen(elems):
        if all(team_ok(block) for block in partition):
            count += 1

    print(count % MOD)

main()
