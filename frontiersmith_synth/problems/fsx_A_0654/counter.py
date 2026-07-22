import sys

# Format D checker -- Greenhouse Ledger plan verifier + fewest-multiplier scorer.
#
# Input <in>:  S K T p m ; x0 ; K*(A_k banded matrix + b_k) ; pattern[0..p-1] ;
#              m * (t ; G dense matrix ; h)
# Output <out>: a PLAN = a sequence of actions from a FIXED menu, each with a
#   CHECKER-COMPUTED (never solver-supplied) multiplier cost:
#     STEP P k     apply recipe block k densely for one day      cost = S*S
#     STEP O idx   apply recalibration graft idx for one day     cost = S*S
#     BAND k       apply recipe block k using its banded entries cost = nnz(A_k)
#     BUILD        cache the composed one-period map              cost = p*S^3
#     PERIOD n     apply the cached period map n times (n>=1)     cost = 2*bits(n)*S^3
#                  (requires a prior BUILD; only valid when the current day is
#                   phase-aligned (day % p == 0) and no recalibration day falls
#                   inside the n*p-day span it covers)
# The plan must exactly cover all T days (final day counter == T) and its
# replayed final state (exact arithmetic mod MOD) must equal the reference
# simulation's final state. Objective (minimize) = total charged multiplier
# cost F. Baseline B = T*S*S (an all-dense STEP-only plan). Ratio = min(1, 0.1*B/F).

MOD = 1_000_000_007
MAXN = 200_000  # cap on number of plan actions (bounds checker time)


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read input")
    try:
        out_text = open(sys.argv[2]).read()
    except Exception:
        fail("cannot read output")
    out = out_text.split()

    it = iter(inp)

    def nxt_int():
        return int(next(it))

    try:
        S = nxt_int(); K = nxt_int(); T = nxt_int(); p = nxt_int(); m = nxt_int()
    except Exception:
        fail("bad header")
    if not (1 <= S <= 32 and 1 <= K <= 16 and 1 <= T <= 100000 and 1 <= p <= 64 and 0 <= m <= 10000):
        fail("bad dims")

    try:
        x0 = [nxt_int() % MOD for _ in range(S)]
        A = []
        B_ = []
        nnz_list = []
        for _k in range(K):
            mat = [[nxt_int() % MOD for _ in range(S)] for _i in range(S)]
            bias = [nxt_int() % MOD for _ in range(S)]
            A.append(mat)
            B_.append(bias)
            nnz_list.append(sum(1 for row in mat for v in row if v != 0))
        pattern = [nxt_int() for _ in range(p)]
        for kk in pattern:
            if not (0 <= kk < K):
                fail("bad pattern entry")
        overrides = []  # list of (t, G, h) in file order (sorted by t, per generator)
        for _o in range(m):
            t = nxt_int()
            if not (0 <= t < T):
                fail("bad override day")
            G = [[nxt_int() % MOD for _ in range(S)] for _i in range(S)]
            h = [nxt_int() % MOD for _ in range(S)]
            overrides.append((t, G, h))
    except Exception:
        fail("bad instance body")

    # instance must be fully consumed (sanity on our own generator, not required
    # of the solver, but guards a corrupted <in> file)
    try:
        next(it)
        fail("trailing input tokens")
    except StopIteration:
        pass

    for i in range(1, len(overrides)):
        if overrides[i][0] <= overrides[i - 1][0]:
            fail("overrides not strictly increasing (bad instance)")
    override_at = {t: idx for idx, (t, _G, _h) in enumerate(overrides)}

    S2 = S * S
    S3 = S ** 3

    def apply_map(state, mat, bias):
        return [
            (sum(mat[i][j] * state[j] for j in range(S)) + bias[i]) % MOD
            for i in range(S)
        ]

    # ---- reference simulation (independent ground truth) ----
    ref_state = x0[:]
    for t in range(T):
        if t in override_at:
            _tt, G, h = overrides[override_at[t]]
            ref_state = apply_map(ref_state, G, h)
        else:
            k = pattern[t % p]
            ref_state = apply_map(ref_state, A[k], B_[k])

    # ---- parse participant plan ----
    if not out:
        fail("empty output")
    pos = 0

    def take():
        nonlocal pos
        if pos >= len(out):
            raise IndexError
        v = out[pos]
        pos += 1
        return v

    try:
        N = int(take())
    except Exception:
        fail("bad action count")
    if N < 1 or N > MAXN:
        fail("action count out of range")

    total_cost = 0
    built = False
    t_cur = 0
    state = x0[:]

    try:
        for _ in range(N):
            typ = take()
            if typ == "STEP":
                src = take()
                if src == "P":
                    k = int(take())
                    if not (0 <= k < K):
                        fail("STEP P: bad k")
                    if t_cur >= T:
                        fail("STEP P: past horizon")
                    state = apply_map(state, A[k], B_[k])
                    total_cost += S2
                    t_cur += 1
                elif src == "O":
                    idx = int(take())
                    if not (0 <= idx < m):
                        fail("STEP O: bad idx")
                    if overrides[idx][0] != t_cur:
                        fail("STEP O: idx not scheduled for current day")
                    _tt, G, h = overrides[idx]
                    state = apply_map(state, G, h)
                    total_cost += S2
                    t_cur += 1
                else:
                    fail("STEP: bad source tag")
            elif typ == "BAND":
                k = int(take())
                if not (0 <= k < K):
                    fail("BAND: bad k")
                if t_cur >= T:
                    fail("BAND: past horizon")
                if t_cur in override_at:
                    fail("BAND: cannot use banded step on a recalibration day")
                state = apply_map(state, A[k], B_[k])
                total_cost += nnz_list[k]
                t_cur += 1
            elif typ == "BUILD":
                total_cost += p * S3
                built = True
            elif typ == "PERIOD":
                n = int(take())
                if n < 1:
                    fail("PERIOD: n must be >= 1")
                if not built:
                    fail("PERIOD: no prior BUILD")
                if t_cur % p != 0:
                    fail("PERIOD: not phase-aligned")
                span = p * n
                if t_cur + span > T:
                    fail("PERIOD: overruns horizon")
                lo, hi = t_cur, t_cur + span
                for (ot, _G, _h) in overrides:
                    if lo <= ot < hi:
                        fail("PERIOD: span crosses a recalibration day")
                for tt in range(lo, hi):
                    k = pattern[tt % p]
                    state = apply_map(state, A[k], B_[k])
                total_cost += 2 * n.bit_length() * S3
                t_cur += span
            else:
                fail("unknown action tag %r" % typ)
    except IndexError:
        fail("truncated / wrong token count")
    except (ValueError, TypeError):
        fail("non-integer parameter")

    if pos != len(out):
        fail("trailing plan tokens")
    if t_cur != T:
        fail("plan does not cover exactly T days (t_cur=%d, T=%d)" % (t_cur, T))
    if state != ref_state:
        fail("final state mismatch")

    Bline = T * S2
    F = total_cost
    if F <= 0:
        fail("nonpositive cost")
    ratio = min(1.0, 0.1 * Bline / max(1e-9, F))
    print("F=%d B=%d Ratio: %.6f" % (F, Bline, ratio))


if __name__ == "__main__":
    main()
