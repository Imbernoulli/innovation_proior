import sys

# Format D checker -- VCG payments for a land-plot auction, computed by an
# explicit straight-line arithmetic circuit (wire-indexed, like a tiny netlist).
#
# Instance (<in>):   n T
#                     T lines, each n bid values v_1..v_n (1-indexed bidders)
#   Adjacent plots (i,i+1) conflict; the welfare-maximizing allocation is the
#   maximum-weight independent set on the path 1..n. The checker computes each
#   trial's n VCG (Clarke-pivot) payments directly with its own O(n)
#   forward+backward DP -- that is the ONLY ground truth.
#
#   The T trials share the same n (same circuit topology, same input-wire
#   count) but carry INDEPENDENTLY random bid values. The submission emits
#   ONE fixed circuit; the checker re-evaluates that SAME circuit once per
#   trial (only the values on input wires 0..n-1 change) and requires every
#   trial's outputs to match exactly. This is the anti-memorization guard: a
#   circuit that just hardcodes one instance's numeric answer via bare
#   literals cannot also reproduce the other, independently-random trials.
#
# Participant (<out>): a straight-line integer circuit
#   G
#   G gate lines, each one of:
#       CONST c        (introduces literal integer c, no operand)
#       ADD a b | SUB a b | MUL a b | MAX a b | GT a b
#   OUT w_1 ... w_n
#   Wires: inputs are wires 0..n-1 (wire i-1 = v_i for the trial being
#   evaluated); the g-th gate line (0-indexed) creates wire n+g and may
#   reference only STRICTLY earlier wires. GT a b = 1 if value(a) > value(b)
#   else 0. `OUT` names the n wires holding pay_1..pay_n, each 0<=w<n+G.
#
# Scoring:
#   1) EXACT equivalence gate: for EVERY trial, every OUT wire must equal
#      that trial's reference VCG payment EXACTLY (integer arithmetic
#      throughout). Any mismatch on any trial, malformed token, non-finite
#      value, out-of-range wire/literal, or trailing tokens -> Ratio 0.0.
#   2) Objective (minimize) = G, the gate count (the circuit's SHAPE is
#      fixed across trials, so G does not depend on T).
#      Baseline B = 4*n^2 + 1 -- the exact gate count of the straightforward
#      construction that reruns the O(n) welfare recurrence fresh for every
#      one of the n leave-one-out economies (no sharing across economies).
#      ratio = min(1.0, 0.1 * B / G)

MAXGATES = 200000
VALCAP = 10 ** 12
BIDCAP = 10 ** 6


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def read_instance(path):
    toks = open(path).read().split()
    it = iter(toks)
    try:
        n = int(next(it))
        T = int(next(it))
    except Exception:
        fail("bad instance header")
    if not (1 <= n <= 2000 and 1 <= T <= 20):
        fail("instance n/T out of range")
    trials = []
    try:
        for _ in range(T):
            v = [int(next(it)) for _ in range(n)]
            trials.append(v)
    except Exception:
        fail("truncated instance")
    for v in trials:
        for val in v:
            if not (1 <= val <= BIDCAP):
                fail("bid value out of range")
    return n, T, trials


def reference_payments(n, v):
    # forward DP: F[j] = best welfare using bidders 1..j (path independent set)
    F = [0] * (n + 1)
    if n >= 1:
        F[1] = v[0]
    for j in range(2, n + 1):
        F[j] = max(F[j - 1], F[j - 2] + v[j - 1])
    # backward DP: Bwd[j] = best welfare using bidders j..n
    Bwd = [0] * (n + 2)
    if n >= 1:
        Bwd[n] = v[n - 1]
    for j in range(n - 1, 0, -1):
        Bwd[j] = max(Bwd[j + 1], Bwd[j + 2] + v[j - 1])
    OPT = F[n]
    pay = []
    for i in range(1, n + 1):
        V0 = F[i - 1] + Bwd[i + 1]          # OPT(economy without bidder i) -- exact, since
        x = 1 if OPT > V0 else 0            # non-adjacent removal never unlocks new pairs
        pay.append(V0 - OPT + v[i - 1] * x)
    return pay


def parse_program(path, n):
    """Parse+structurally validate the circuit ONCE (independent of any
    trial's concrete bid values): opcodes, wire-reference ranges, literal
    ranges, OUT arity/range, and exact token consumption. Returns
    (gates, outs) or calls fail()."""
    out = open(path).read().split()
    if not out:
        fail("empty output")
    idx = 0

    def nxt():
        nonlocal idx
        if idx >= len(out):
            raise IndexError
        t = out[idx]
        idx += 1
        return t

    try:
        G = int(nxt())
    except Exception:
        fail("bad G")
    if G < 0 or G > MAXGATES:
        fail("G out of range")

    gates = []
    try:
        for g in range(G):
            cur = n + g
            op = nxt()
            if op == "CONST":
                c = int(nxt())
                if not (-VALCAP <= c <= VALCAP):
                    fail("literal out of range")
                gates.append(("CONST", c, None))
            elif op in ("ADD", "SUB", "MUL", "MAX", "GT"):
                a = int(nxt())
                b = int(nxt())
                if not (0 <= a < cur and 0 <= b < cur):
                    fail("wire reference out of range (not strictly earlier)")
                gates.append((op, a, b))
            else:
                fail("unknown opcode '%s'" % op)
        tok = nxt()
        if tok != "OUT":
            fail("missing OUT")
        outs = []
        for _ in range(n):
            w = int(nxt())
            if not (0 <= w < n + G):
                fail("OUT wire out of range")
            outs.append(w)
    except (IndexError, ValueError):
        fail("malformed / non-finite token")
    if idx != len(out):
        fail("trailing tokens")
    return G, gates, outs


def evaluate(v, gates, outs):
    """Evaluate the FIXED gate list against one trial's input wires."""
    wires = list(v)
    for op, a, b in gates:
        if op == "CONST":
            wires.append(a)
            continue
        va, vb = wires[a], wires[b]
        if op == "ADD":
            r = va + vb
        elif op == "SUB":
            r = va - vb
        elif op == "MUL":
            r = va * vb
        elif op == "MAX":
            r = va if va >= vb else vb
        else:  # GT
            r = 1 if va > vb else 0
        if not (-VALCAP <= r <= VALCAP):
            fail("intermediate value out of bounds")
        wires.append(r)
    return [wires[w] for w in outs]


def main():
    n, T, trials = read_instance(sys.argv[1])
    G, gates, outs = parse_program(sys.argv[2], n)

    for v in trials:
        ref_pay = reference_payments(n, v)
        got_pay = evaluate(v, gates, outs)
        if got_pay != ref_pay:
            fail("payments do not match the reference VCG payments on a trial")

    B = 4 * n * n + 1
    F = G if G > 0 else 1
    ratio = min(1.0, 0.1 * B / max(1e-9, F))
    print("G=%d B=%d Ratio: %.6f" % (G, B, ratio))


if __name__ == "__main__":
    main()
