# TIER: trivial
# A timid underwriter: rank by isolated margin and write only the top ceil(C/3) of
# them, leaving the rest of the capacity idle. Storms are ignored entirely (the small
# batch is unlikely to matter). Reproduces the checker's own internal baseline exactly.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); C = int(next(it)); K = int(next(it)); OVER_MULT = int(next(it))
    for _ in range(K):
        for _ in range(5):
            next(it)
    policies = []
    for _ in range(N):
        x = int(next(it)); y = int(next(it)); e = int(next(it)); p = int(next(it)); tech = int(next(it))
        policies.append((x, y, e, p, tech))

    order = sorted(range(N), key=lambda i: (-(policies[i][3] - policies[i][4]), i))
    k = max(1, -(-C // 3))
    book = order[:k]
    print(len(book))
    print(*book)


if __name__ == "__main__":
    main()
