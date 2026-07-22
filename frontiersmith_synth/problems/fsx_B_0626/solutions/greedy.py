# TIER: greedy
import sys

# The "obvious" competent-coder recipe: textbook cycle decomposition of pi, then realize
# each cycle as a STAR of transpositions (center, pi(center)), (center, pi^2(center)), ...
# from whichever cycle element minimizes the total Hamming distance to the rest of the
# cycle. Each individual transposition only pays for the bits that actually differ (a
# proven palindrome gadget of length 2*|D|-1, all (n-1)-control gates), instead of the
# trivial tier's wasteful fixed-length 2n-1. It also tries the cheapest structural
# shortcut a coder would think of first: check whether any SINGLE input bit happens to
# pass straight through pi unchanged (pi(x^e) == pi(x)^e for that axis e alone) -- i.e.
# "conjugation-detection" restricted to the n coordinate axes only, never a general
# XOR-combination of bits as a candidate change of basis. On a genuinely conjugated
# instance (planted so no single axis is invariant) this always comes up empty, and the
# solution falls back to the star synthesis above -- a real, distinct algorithm from the
# trivial tier, just not the insight.


def transposition_gates_shortest(v1, v2, n):
    diff = [b for b in range(n) if ((v1 >> b) & 1) != ((v2 >> b) & 1)]
    fwd = []
    cur = v1
    for b in diff:
        w_prev = cur
        controls = [(bit, (w_prev >> bit) & 1) for bit in range(n) if bit != b]
        fwd.append((controls, b))
        cur ^= (1 << b)
    return fwd + list(reversed(fwd[:-1])) if fwd else []


def hamdist(a, b):
    return bin(a ^ b).count("1")


def find_cycles(pi, n):
    N = 1 << n
    seen = [False] * N
    cycles = []
    for x in range(N):
        if seen[x]:
            continue
        if pi[x] == x:
            seen[x] = True
            continue
        c = [x]
        seen[x] = True
        y = pi[x]
        while y != x:
            c.append(y)
            seen[y] = True
            y = pi[y]
        cycles.append(c)
    return cycles


def star_decompose(cycle, pi, center):
    """(center, pi(center)), (center, pi^2(center)), ... applied left-to-right realizes
    the whole cycle exactly."""
    seq = [center]
    y = pi[center]
    while y != center:
        seq.append(y)
        y = pi[y]
    return [(center, seq[i]) for i in range(1, len(seq))]


def best_center(cycle):
    best_c, best_cost = cycle[0], None
    for c in cycle:
        tot = sum(hamdist(c, v) for v in cycle if v != c)
        if best_cost is None or tot < best_cost:
            best_cost, best_c = tot, c
    return best_c


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    N = 1 << n
    pi = [int(next(it)) for _ in range(N)]

    # cheap "obvious" structural check: any single axis passes through unchanged?
    _ = [e for e in range(n) if all(pi[x ^ (1 << e)] == (pi[x] ^ (1 << e)) for x in range(N))]
    # (on this family's instances this is always empty by construction -- included so the
    # method genuinely tries the easy thing first, as a real "greedy" coder would)

    cycles = find_cycles(pi, n)
    gates = []
    for c in cycles:
        center = best_center(c)
        for u, v in star_decompose(c, pi, center):
            gates.extend(transposition_gates_shortest(u, v, n))

    out = [str(len(gates))]
    for controls, t in gates:
        parts = [str(len(controls))]
        for cc, p in controls:
            parts.append(str(cc)); parts.append(str(p))
        parts.append(str(t))
        out.append(" ".join(parts))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
