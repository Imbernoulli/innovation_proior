import sys
import model


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def main():
    try:
        in_toks = open(sys.argv[1]).read().split()
        tiers = model.parse_instance(in_toks)
    except Exception:
        fail("bad input")
        return
    K = len(tiers)

    try:
        out_toks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
        return

    if len(out_toks) != K:
        fail("expected %d prices, got %d tokens" % (K, len(out_toks)))
        return

    prices = []
    for tok in out_toks:
        # strict integer parse: rejects "nan"/"inf"/floats/garbage outright
        body = tok[1:] if tok.startswith("-") else tok
        if not body.isdigit():
            fail("non-integer price token %r" % tok)
            return
        try:
            v = int(tok)
        except Exception:
            fail("bad price token %r" % tok)
            return
        prices.append(v)

    for i, (t, p) in enumerate(zip(tiers, prices)):
        lo, hi = model.tier_band(t)
        if p < lo or p > hi:
            fail("tier %d price %d outside rate-change-cap band [%d,%d]" % (i, p, lo, hi))
            return

    F = model.total_profit(tiers, prices)
    B = model.baseline_profit(tiers)
    B = max(1e-6, B)

    sc = 100.0 * F / B
    sc = max(0.0, min(1000.0, sc))
    print("F=%.4f B=%.4f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
