# TIER: greedy
import sys


def read_ints(tokens, k):
    return [int(next(tokens)) for _ in range(k)]


def main():
    toks = iter(sys.stdin.read().split())
    T, N, G = read_ints(toks, 3)
    V, P = read_ints(toks, 2)
    D = read_ints(toks, T)
    suppliers = []
    for _ in range(N):
        group, qualified, lead, qualcost, ntiers = read_ints(toks, 5)
        tiers = [tuple(read_ints(toks, 2)) for _ in range(ntiers)]
        suppliers.append(dict(group=group, qualified=qualified, lead=lead,
                               qualcost=qualcost, tiers=tiers))
    E = int(next(toks))
    for _ in range(E):
        read_ints(toks, 2)

    # The obvious first-write recipe: find the globally cheapest (already
    # qualified) supplier and single-source the ENTIRE order every period, so
    # every order clears the deepest volume-discount tier. Cost-minimal, and
    # correct -- as long as nothing ever goes wrong. It never looks at the
    # disruption calendar and never spends a cent on a second qualified source,
    # so a correlated failure at the sole supplier wipes out every period it hits.
    primary = 0
    best_base = suppliers[0]["tiers"][0][1]
    for i in range(N):
        if suppliers[i]["qualified"] and suppliers[i]["tiers"][0][1] < best_base:
            primary = i
            best_base = suppliers[i]["tiers"][0][1]

    orders = []
    for t in range(1, T + 1):
        Dt = D[t - 1]
        if Dt > 0:
            orders.append((t, primary, Dt))

    out = ["0"]
    out.append(str(len(orders)))
    for t, i, q in orders:
        out.append(f"{t} {i} {q}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
