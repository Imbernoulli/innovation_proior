# TIER: greedy
"""The obvious first attempt: weld struts in input order, and at each strut
pick the side that balances a RUNNING, UNDISCOUNTED ledger per joint -- i.e.
it reasons as if 'equal and opposite ratings cancel', and never looks at the
actual 1/(1+stiffness) divisor the checker applies. On components built from
several equal-rating struts (the checker's planted trap gadgets) this leaves
a large, systematic residual, because the ledger this greedy trusts silently
diverges from the true stiffness-discounted physics."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it))
    edges = []
    for _ in range(m):
        u = int(next(it)); w = int(next(it)); eff = int(next(it))
        edges.append((u, w, eff))

    ledger = [0.0] * n
    sides = [0] * m
    for i in range(m):
        u, w, eff = edges[i]
        cu_p, cw_p = ledger[u] + eff, ledger[w] - eff
        cu_m, cw_m = ledger[u] - eff, ledger[w] + eff
        worst_plus = max(abs(cu_p), abs(cw_p))
        worst_minus = max(abs(cu_m), abs(cw_m))
        if worst_plus <= worst_minus:
            sides[i] = 1
            ledger[u], ledger[w] = cu_p, cw_p
        else:
            sides[i] = -1
            ledger[u], ledger[w] = cu_m, cw_m

    out = [str(m)]
    for i in range(m):
        out.append(f"{i} {sides[i]}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
