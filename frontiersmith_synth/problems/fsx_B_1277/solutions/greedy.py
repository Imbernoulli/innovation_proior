# TIER: greedy
# The obvious "write every adequately-priced policy" approach: rank candidates by
# isolated margin (premium minus technical price) and fill underwriting capacity from
# the top, WITHOUT looking at storm footprints or accumulation limits at all. This is
# the natural first instinct -- and it is exactly what walks the whole book into a
# single storm's footprint on the planted trap cases.
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
    book = [i for i in order if policies[i][3] - policies[i][4] > 0][:C]

    print(len(book))
    print(*book)


if __name__ == "__main__":
    main()
