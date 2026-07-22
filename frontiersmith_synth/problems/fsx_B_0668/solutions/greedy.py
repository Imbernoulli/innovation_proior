# TIER: greedy
# Textbook instinct: sell the highest-demand lots first (lock in the biggest tickets
# while everyone's still bidding), and set a simple rule-of-thumb reserve (a flat
# fraction of the top bid) as a token nod to "don't let it go too cheap". This
# ignores that the SALE ORDER reshapes remaining budgets for later lots (a bidder
# who is a crucial underbidder for one lot may already be broke by the time that lot
# comes up if pricier lots were sold first), and a flat-fraction reserve leaves most
# of the extractable value on the table for genuinely uncontested lots.
import sys

RESERVE_FRAC = 0.2


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it)); m = int(next(it))
    values = [[int(next(it)) for _ in range(m)] for _ in range(n)]
    _budgets = [int(next(it)) for _ in range(m)]

    maxval = [max(row) if row else 0 for row in values]
    order = sorted(range(n), key=lambda i: (-maxval[i], i))

    out = []
    for lot0 in order:
        r = int(RESERVE_FRAC * maxval[lot0])
        out.append(f"{lot0 + 1} {r}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
