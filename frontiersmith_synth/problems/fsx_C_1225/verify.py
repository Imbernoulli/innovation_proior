#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the hierarchical
rate-limiter design problem. Prints exactly one line ending 'Ratio: <float>'.

Feasibility: H+G <= M (memory budget on tracked bucket counters), G>=1,
all integers, 0<=key<=1e12, 0<=cap,rate<=RATE_MAX, no duplicate explicit keys,
EXACT token count in the artifact (no trailing/garbage tokens).

Objective: replay the trace through the declared bucket hierarchy with an
integer token-bucket policy; F = max(0, served_good - served_abusive).
Baseline B = served_good under the checker's own single non-isolating shared
bucket (H=0, G=1, fixed small cap/rate) -- always positive (some legitimate
traffic gets through any minimally-alive bucket), and stable even when the
trace is abuse-dominated (unlike a net-based baseline, which can flip sign).
sc = min(1000, 100*F/max(1e-9,B));  print Ratio: sc/1000.
"""
import sys

RATE_MAX = 5000
KEY_MAX = 10 ** 12
BASE_CAP = 20
BASE_RATE = 3


def fail(msg):
    print("INVALID: %s Ratio: 0.0" % msg)
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    try:
        T = int(next(it)); M = int(next(it)); A = int(next(it)); B = int(next(it)); P = int(next(it))
        R = int(next(it))
        events = []
        for _ in range(R):
            t = int(next(it)); key = int(next(it)); lab = int(next(it))
            events.append((t, key, lab))
    except (StopIteration, ValueError):
        # malformed instance file -- should never happen (we control gen.py), but
        # fail closed rather than crash.
        print("BAD_INSTANCE Ratio: 0.0")
        sys.exit(0)
    return T, M, A, B, P, events


def parse_artifact(path, M):
    """Strict parse: exact token count, integers only, all bounds checked.
    Returns (H, G, explicit_dict, gcap, grate) or calls fail()."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except Exception:
        fail("cannot read output")

    if len(toks) < 2:
        fail("too few tokens")
    try:
        H = int(toks[0]); G = int(toks[1])
    except ValueError:
        fail("H/G not integers")

    if H < 0 or G < 1:
        fail("H must be >=0 and G must be >=1")
    if H + G > M:
        fail("memory budget exceeded: H+G=%d > M=%d" % (H + G, M))

    expected = 2 + 3 * H + 2
    if len(toks) != expected:
        fail("token count mismatch: got %d expected %d" % (len(toks), expected))

    explicit = {}
    idx = 2
    for _ in range(H):
        try:
            k = int(toks[idx]); cap = int(toks[idx + 1]); rate = int(toks[idx + 2])
        except ValueError:
            fail("non-integer explicit bucket entry")
        idx += 3
        if not (0 <= k <= KEY_MAX):
            fail("explicit key out of range")
        if not (0 <= cap <= RATE_MAX) or not (0 <= rate <= RATE_MAX):
            fail("explicit cap/rate out of range")
        if k in explicit:
            fail("duplicate explicit key %d" % k)
        explicit[k] = (cap, rate)

    try:
        gcap = int(toks[idx]); grate = int(toks[idx + 1])
    except ValueError:
        fail("non-integer group template")
    if not (0 <= gcap <= RATE_MAX) or not (0 <= grate <= RATE_MAX):
        fail("group cap/rate out of range")

    return H, G, explicit, gcap, grate


def replay(events, A, B, P, G, explicit, gcap, grate):
    """Deterministic integer token-bucket replay. events already sorted by t
    (checker trusts and reuses the instance's own emission order)."""
    exp_state = {k: [cap, 0] for k, (cap, rate) in explicit.items()}
    grp_state = [[gcap, 0] for _ in range(G)]
    good = 0
    abusive = 0
    for t, key, lab in events:
        if key in explicit:
            st = exp_state[key]
            cap, rate = explicit[key]
        else:
            g = ((key * A + B) % P) % G
            st = grp_state[g]
            cap, rate = gcap, grate
        dt = t - st[1]
        if dt < 0:
            dt = 0
        tokens = st[0] + rate * dt
        if tokens > cap:
            tokens = cap
        st[1] = t
        if tokens >= 1:
            tokens -= 1
            if lab == 1:
                good += 1
            else:
                abusive += 1
        st[0] = tokens
    return good, abusive


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        return 0
    in_path, out_path = sys.argv[1], sys.argv[2]
    T, M, A, B, P, events = read_instance(in_path)

    H, G, explicit, gcap, grate = parse_artifact(out_path, M)

    good, abusive = replay(events, A, B, P, G, explicit, gcap, grate)
    F = good - abusive
    if F < 0:
        F = 0.0

    base_good, base_abusive = replay(events, A, B, P, 1, {}, BASE_CAP, BASE_RATE)
    Bval = base_good if base_good >= 1 else 1.0

    sc = min(1000.0, 100.0 * F / max(1e-9, Bval))
    print("good=%d abusive=%d F=%.1f base_good=%d base_abusive=%d B=%.1f Ratio: %.6f" %
          (good, abusive, F, base_good, base_abusive, Bval, sc / 1000.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
