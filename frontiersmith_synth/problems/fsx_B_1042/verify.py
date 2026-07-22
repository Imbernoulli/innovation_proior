#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the algae-fuel-pond-fleet
problem. Prints 'Ratio: <float in [0,1]>' on the last line; exits 0 always."""
import sys
import math


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    P = int(next(it))
    T = int(next(it))
    C = float(next(it))
    ponds = []
    for _ in range(P):
        a = float(next(it))
        b0 = float(next(it))
        e0 = float(next(it))
        decay = float(next(it))
        tau = float(next(it))
        ponds.append((a, b0, e0, decay, tau))
    return P, T, C, ponds


def fail(msg):
    sys.stdout.write("INFEASIBLE: %s\n" % msg)
    sys.stdout.write("Ratio: 0.0\n")
    sys.exit(0)


def simulate(P, T, C, ponds, starts, switches, feeds, strict=True):
    """Replays the fleet trajectory and returns total fuel F. If strict, validates
    feasibility along the way and calls fail() (exit 0, Ratio 0.0) on any violation."""
    feed_at = [0.0] * T
    fuels = [0.0] * P
    for p in range(P):
        a, b0, e0, decay, tau = ponds[p]
        s, e = starts[p], switches[p]
        w = e - s
        row = feeds[p]
        if strict and len(row) != w:
            fail("pond %d feed row length %d != window %d" % (p, len(row), w))
        B = b0
        for k in range(w):
            f = row[k]
            if strict and (not math.isfinite(f) or f < -1e-9):
                fail("pond %d has a non-finite/negative feed value" % p)
            f = max(0.0, f)
            t = s + k
            feed_at[t] += f
            # Activation threshold: below tau the flow keeps the culture alive
            # but does not trigger cell division -- zero biomass gain that step.
            if f >= tau - 1e-9:
                B += a * math.sqrt(f)
        fuels[p] = e0 * (decay ** e) * B
    if strict:
        for t in range(T):
            if feed_at[t] > C + 1e-4:
                fail("shared feed cap exceeded at step %d: %.6f > %.6f" % (t, feed_at[t], C))
    return sum(fuels)


def main():
    if len(sys.argv) < 4:
        fail("bad invocation")
    inp, outp = sys.argv[1], sys.argv[2]
    P, T, C, ponds = read_instance(inp)

    with open(outp) as f:
        toks = f.read().split()
    it = iter(toks)

    try:
        Pout = int(next(it))
    except StopIteration:
        fail("empty output")
    except ValueError:
        fail("non-integer pond count")
    if Pout != P:
        fail("pond count mismatch: declared %d, expected %d" % (Pout, P))

    starts, switches, feeds = [], [], []
    try:
        for p in range(P):
            s_tok = next(it)
            e_tok = next(it)
            s = int(s_tok)
            e = int(e_tok)
            if not (0 <= s <= e <= T):
                fail("pond %d start/switch out of range: start=%d switch=%d T=%d" % (p, s, e, T))
            starts.append(s)
            switches.append(e)
            row = []
            for _ in range(e - s):
                v = float(next(it))
                if not math.isfinite(v):
                    fail("pond %d feed value not finite" % p)
                row.append(v)
            feeds.append(row)
    except StopIteration:
        fail("truncated output")
    except ValueError:
        fail("non-numeric token where an integer/feed value was expected")

    # Reject trailing garbage after the expected P blocks (e.g. an appended
    # nan/inf/huge token tacked onto an otherwise-valid submission): the output
    # must contain EXACTLY the declared tokens, nothing more.
    leftover = list(it)
    if leftover:
        fail("trailing tokens after the expected output: %r" % (leftover[:5],))

    F = simulate(P, T, C, ponds, starts, switches, feeds, strict=True)
    if not math.isfinite(F) or F < 0:
        fail("non-finite or negative objective")

    # Internal baseline B: the "single-pipe" trivial construction -- feed the
    # WHOLE line to pond 0 for the whole horizon and ignore the rest of the
    # fleet entirely (every other pond harvests immediately, no growth). This
    # is always feasible and always non-degenerate (pond 0 always clears its
    # own threshold, since tau < C for every pond).
    base_feeds = [[0.0] * T for _ in range(P)]
    base_feeds[0] = [C] * T
    base_starts = [0] * P
    base_switches = [T] + [0] * (P - 1)
    B = simulate(P, T, C, ponds, base_starts, base_switches, base_feeds, strict=False)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    sys.stdout.write("F=%.6f B=%.6f\n" % (F, B))
    sys.stdout.write("Ratio: %.6f\n" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
