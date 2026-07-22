# TIER: trivial
import sys

# Canonical "structure-blind" reference construction (matches the checker's own baseline
# formula exactly -- this reproduces the checker's baseline cost B/Q, i.e. ratio ~= 0.1*Q).
#
# For each input x whose output isn't already correct, find where the needed target value
# currently sits (cycle-following via a position index) and swap the two register VALUES
# with a fixed-length transposition gadget: walk ALL n bit positions in the canonical order
# 0,1,...,n-1 -- the |D| positions where the two values actually differ realize the move
# (a proven palindrome of adjacent-transposition gates), and the remaining n-|D| positions
# are visited anyway via a self-canceling "there and back" detour (pure no-op, deliberately
# wasteful). Every gate ends up with exactly n-1 controls, and every transposition costs a
# FIXED 2n-1 gates regardless of how close v1, v2 actually are -- no attempt is made to
# notice that most of the state is untouched, which is exactly what the naive approach
# misses when the target was secretly built as a conjugated, mostly-local permutation.


def transposition_gates_fixed(v1, v2, n):
    diff = [b for b in range(n) if ((v1 >> b) & 1) != ((v2 >> b) & 1)]
    pad = [b for b in range(n) if b not in diff]

    fwd = []
    cur = v1
    for b in diff:
        w_prev = cur
        controls = [(bit, (w_prev >> bit) & 1) for bit in range(n) if bit != b]
        fwd.append((controls, b))
        cur ^= (1 << b)
    main_seq = fwd + list(reversed(fwd[:-1])) if fwd else []

    pfwd = []
    cur2 = v1
    for b in pad:
        w_prev = cur2
        controls = [(bit, (w_prev >> bit) & 1) for bit in range(n) if bit != b]
        pfwd.append((controls, b))
        cur2 ^= (1 << b)
    pad_seq = pfwd + list(reversed(pfwd)) if pfwd else []

    return main_seq + pad_seq


def canonical_gates(pi, n):
    N = 1 << n
    cur = list(range(N))
    pos = list(range(N))
    gates = []
    for x in range(N):
        target = pi[x]
        if cur[x] == target:
            continue
        y = pos[target]
        gates.extend(transposition_gates_fixed(cur[x], cur[y], n))
        vx, vy = cur[x], cur[y]
        cur[x], cur[y] = cur[y], cur[x]
        pos[vx], pos[vy] = y, x
    return gates


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    N = 1 << n
    pi = [int(next(it)) for _ in range(N)]

    gates = canonical_gates(pi, n)

    out = [str(len(gates))]
    for controls, t in gates:
        parts = [str(len(controls))]
        for c, p in controls:
            parts.append(str(c)); parts.append(str(p))
        parts.append(str(t))
        out.append(" ".join(parts))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
