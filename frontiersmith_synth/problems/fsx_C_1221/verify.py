#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans is unused, per format-C contract)

Scores a per-request retry POLICY (max_attempts + a backoff-delta list) on
the "retrying without making the outage worse" objective. Simulates all N
requests, tick by tick, over a shared backend with:

  - a shared capacity pool (retry-storm-feedback): if the number of attempts
    DUE at a tick exceeds the current effective capacity, the excess is
    dropped (must retry later per the policy's own backoff), AND the
    overload itself multiplicatively shrinks the effective capacity for
    subsequent ticks (a collapsing backend); capacity recovers additively
    while demand stays within budget.
  - per-endpoint outage / ambiguous-ack-loss ticks (idempotency-classification):
    an admitted attempt against a DOWN endpoint cleanly fails; against a
    HEALTHY endpoint it server-side succeeds -- but if that tick is also
    flagged "ambiguous" the client still perceives a failure and retries a
    request the server already completed. A later attempt that ALSO
    succeeds server-side for an already-succeeded NON-idempotent request is
    a harmful DUPLICATE (idempotent endpoints never incur this penalty).

This is a MAXIMIZATION objective. F = confirmed_successes - DUP_PENALTY *
duplicate_count (floored at 0). The checker's own trivial baseline B is F
under "one attempt only, never retry" (max_attempts=1 for every request).
Ratio = min(1000, 100*F/B) / 1000.
"""
import sys
import math

MAX_ATTEMPTS_CAP = 6
MAX_BACKOFF_CAP = 500

COLLAPSE_RATE = 0.45
RECOVERY_STEP = 0.2
MIN_LOAD_FACTOR = 0.12
MIN_CAP = 1
DUP_PENALTY = 3.0


def fail(msg):
    print(f"INVALID: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_input(path):
    with open(path, "r") as f:
        toks = f.read().split()
    it = iter(toks)
    try:
        T = int(next(it)); N = int(next(it)); E = int(next(it)); C = int(next(it))
        idem = [int(next(it)) for _ in range(E)]
        outage = [[int(next(it)) for _ in range(T)] for _ in range(E)]
        ambiguous = [[int(next(it)) for _ in range(T)] for _ in range(E)]
        arrivals = []
        for _ in range(N):
            t0 = int(next(it)); e0 = int(next(it))
            arrivals.append((t0, e0))
    except StopIteration:
        raise ValueError("truncated input")
    return T, N, E, C, idem, outage, ambiguous, arrivals


def parse_output(path, N):
    """Strict line-based parse: exactly N non-blank lines, line i is
    `max_attempts b_1 ... b_{max_attempts-1}` with all integers in range.
    Returns list of (max_attempts, [backoffs])."""
    try:
        with open(path, "r") as f:
            raw = f.read()
    except Exception:
        fail("cannot read output")

    lines = [ln for ln in raw.splitlines() if ln.strip() != ""]
    if len(lines) != N:
        fail(f"expected exactly {N} non-blank lines, got {len(lines)}")

    policies = []
    for idx, ln in enumerate(lines):
        toks = ln.split()
        if not toks:
            fail(f"request {idx}: empty line")
        try:
            ma = int(toks[0])
        except ValueError:
            fail(f"request {idx}: non-integer max_attempts {toks[0]!r}")
        if ma < 1 or ma > MAX_ATTEMPTS_CAP:
            fail(f"request {idx}: max_attempts {ma} out of range [1,{MAX_ATTEMPTS_CAP}]")
        if len(toks) != ma:
            fail(f"request {idx}: expected {ma} tokens (max_attempts + {ma-1} backoffs), got {len(toks)}")
        backoffs = []
        for tok in toks[1:]:
            try:
                b = int(tok)
            except ValueError:
                fail(f"request {idx}: non-integer backoff {tok!r}")
            if b < 1 or b > MAX_BACKOFF_CAP:
                fail(f"request {idx}: backoff {b} out of range [1,{MAX_BACKOFF_CAP}]")
            backoffs.append(b)
        policies.append((ma, backoffs))
    return policies


def simulate(T, N, E, C, idem, outage, ambiguous, arrivals, policies):
    attempts_used = [0] * N
    active = [True] * N
    had_success = [False] * N
    confirmed = [False] * N
    duplicate_count = 0
    load_factor = 1.0

    schedule = [[] for _ in range(T)]
    for i in range(N):
        t0 = arrivals[i][0]
        if 0 <= t0 < T:
            schedule[t0].append(i)

    for t in range(T):
        demand_list = sorted(schedule[t])
        cap_eff = max(MIN_CAP, int(math.floor(C * load_factor)))
        admit_n = min(len(demand_list), cap_eff)
        admitted = demand_list[:admit_n]
        dropped = demand_list[admit_n:]
        overloaded = len(demand_list) > cap_eff

        for i in demand_list:
            attempts_used[i] += 1

        def schedule_retry(i, t):
            ma, backoffs = policies[i]
            if attempts_used[i] < ma:
                b = backoffs[attempts_used[i] - 1]
                nt = t + b
                if nt < T:
                    schedule[nt].append(i)
                else:
                    active[i] = False
            else:
                active[i] = False

        for i in admitted:
            e = arrivals[i][1]
            if outage[e][t] == 1:
                outcome = "clean_fail"
            else:
                is_dup = had_success[i] and (idem[e] == 0)
                if is_dup:
                    duplicate_count += 1
                had_success[i] = True
                if ambiguous[e][t] == 1:
                    outcome = "ambiguous"
                else:
                    outcome = "confirmed"
                    confirmed[i] = True
                    active[i] = False
            if active[i] and outcome != "confirmed":
                schedule_retry(i, t)

        for i in dropped:
            if active[i]:
                schedule_retry(i, t)

        if overloaded:
            load_factor = max(MIN_LOAD_FACTOR, load_factor * COLLAPSE_RATE)
        else:
            load_factor = min(1.0, load_factor + RECOVERY_STEP)

    confirmed_count = sum(confirmed)
    F_raw = confirmed_count - DUP_PENALTY * duplicate_count
    return F_raw, confirmed_count, duplicate_count


def main():
    if len(sys.argv) < 3:
        fail("usage: verify.py <in> <out> <ans>")
    in_path, out_path = sys.argv[1], sys.argv[2]

    try:
        T, N, E, C, idem, outage, ambiguous, arrivals = read_input(in_path)
    except Exception as e:
        print(f"BAD_INPUT: {e}")
        print("Ratio: 0.0")
        sys.exit(0)

    policies = parse_output(out_path, N)

    F, confirmed_count, dup = simulate(T, N, E, C, idem, outage, ambiguous, arrivals, policies)

    trivial_policies = [(1, []) for _ in range(N)]
    B, _, _ = simulate(T, N, E, C, idem, outage, ambiguous, arrivals, trivial_policies)

    if B <= 1e-9:
        fail("degenerate instance (zero baseline)")

    Fc = max(0.0, F)
    sc = min(1000.0, 100.0 * Fc / B)
    print("confirmed=%d duplicates=%d F=%.3f baseline_B=%.3f" % (confirmed_count, dup, F, B))
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
