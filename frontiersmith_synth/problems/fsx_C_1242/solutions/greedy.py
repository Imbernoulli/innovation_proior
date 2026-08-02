# TIER: greedy
"""The obvious approach: maximize reuse for the BIGGEST matrix in the mix, then
use that one dataflow everywhere. Concretely: find the layer with the largest
element count M*K*N, evaluate the 3 canonical (row/col-symmetric-agnostic)
codes on that single layer, keep the cheapest one, and apply it unchanged to
every layer in the sequence -- no per-layer adaptation, no switching, no
awareness that later layers might have a totally different shape."""
import sys

CANON = {"M": "KNM", "K": "MNK", "N": "MKN"}  # stream-dim -> canonical code


def read_instance():
    toks = sys.stdin.read().split()
    i = 0

    def nxt():
        nonlocal i
        v = toks[i]
        i += 1
        return v

    P = int(nxt()); Q = int(nxt()); L = int(nxt())
    RELOAD = int(nxt()); SWITCH = int(nxt())
    layers = []
    for _ in range(L):
        m = int(nxt()); k = int(nxt()); n = int(nxt())
        layers.append({'M': m, 'K': k, 'N': n})
    return P, Q, L, RELOAD, SWITCH, layers


def layer_cost(P, Q, RELOAD, dims, code):
    d1, d2, s = code[0], code[1], code[2]
    D1, D2, S = dims[d1], dims[d2], dims[s]
    tp = -(-D1 // P)
    tq = -(-D2 // Q)
    pipe = P + Q - 1
    return tp * tq * P * Q * (RELOAD + S + pipe)


def main():
    P, Q, L, RELOAD, SWITCH, layers = read_instance()

    dom = max(range(L), key=lambda i: layers[i]['M'] * layers[i]['K'] * layers[i]['N'])

    best_code, best_cost = None, None
    for s in ("M", "K", "N"):
        code = CANON[s]
        c = layer_cost(P, Q, RELOAD, layers[dom], code)
        if best_cost is None or c < best_cost:
            best_cost, best_code = c, code

    out = [str(L)] + [best_code] * L
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
