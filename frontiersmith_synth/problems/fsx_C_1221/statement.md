# Shared Fate: Retry Budgeting Under a Correlated Outage

## Problem

`N` client requests must each reach one of `E` backend endpoints through a
**shared** backend over `T` discrete ticks. You must output a **retry
policy** for every request: how many attempts it gets, and how long to wait
after each failure before trying again. There is no interaction — you see
the entire instance (capacity, every endpoint's outage/ack-loss schedule,
every request's arrival) up front and commit to one static policy.

Every endpoint is either **idempotent** (re-executing it is harmless) or
**non-idempotent** (re-executing it after it already succeeded is a harmful
duplicate — e.g. a double charge). A short window right after a shared
outage recovers is flagged **ambiguous**: the server actually completes the
work, but the client's acknowledgement is lost, so the client sees a
(false) failure and — being a client — retries.

All `N` requests also compete for one **shared capacity pool**. If the
number of attempts due at a tick exceeds what the backend can currently
handle, the excess is dropped (and must be retried later, per your own
backoff). Worse: an overloaded tick **shrinks** the backend's effective
capacity for the ticks that follow (a thundering-herd collapse); staying
within budget lets capacity recover. Aggressive retrying that ignores this
shared resource can make the outage last far longer than the outage itself.

## Input (stdin)
```
T N E C
idem_0 ... idem_{E-1}                 # 0/1 per endpoint
outage_0[0..T-1]                      # E lines, 0/1: endpoint e down at tick t
...
ambiguous_0[0..T-1]                   # E lines, 0/1: success at (e,t) is ack-lost
...
arrival_0 endpoint_0                  # N lines: request i's arrival tick + endpoint
...
```

## Output (stdout)
`N` lines. Line `i` is your policy for request `i`:
```
max_attempts b_1 b_2 ... b_{max_attempts-1}
```
`max_attempts` in `[1,6]`; each `b_k` (ticks to wait before the next
attempt, after the k-th attempt fails) in `[1,500]`. If `max_attempts=1`,
just print `1` (no backoffs).

**Feasibility**: exactly `N` non-blank lines; each line's token count must
equal its own `max_attempts`; all tokens integers in range. Any violation
scores that instance `0`.

## Simulation (how your policy is graded)
Ticks `0..T-1`, in order. At tick `t`, gather every request whose next
attempt is due now. Let `cap_eff = max(1, floor(C * load_factor))`
(`load_factor` starts at `1.0`). The lowest-id `cap_eff` of them are
**admitted**; the rest are **dropped** (they retry later, per their own
backoff, exactly like a failure — they never reached the server). For an
admitted request against endpoint `e` at tick `t`:
- `outage_e[t]=1` → clean failure.
- else the server succeeds. If the request already had an earlier
  server-side success **and** `idem_e=0`, this is a **duplicate**. Then: if
  `ambiguous_e[t]=1` the client still perceives failure (retries if
  attempts remain); else the client sees **confirmed success** (value `+1`,
  no more attempts).
- A clean failure or a dropped attempt schedules the next attempt at
  `t + b_k` (the k-th backoff), if attempts remain and `t+b_k < T`.

After admission: if demand this tick exceeded `cap_eff`, `load_factor *=
0.45` (floored at `0.12`); otherwise `load_factor = min(1, load_factor +
0.2)`.

## Objective (maximize)
`F = confirmed_successes − 3 × duplicate_count` (floored at `0`). Let `B`
be `F` under the trivial policy "`max_attempts=1` for everyone" (the
checker's own baseline). Then
```
Ratio = min(1000, 100 * F / B) / 1000
```
so the never-retry baseline always scores `≈0.1`.

## Constraints
`6 ≤ N ≤ 70`, `18 ≤ T ≤ 80`, `E = 4`, `2 ≤ C`. All schedule arrays are
`0/1`. Deterministic exact-integer simulation — no floating-point score
dependence beyond the fixed `load_factor` recurrence above, computed
identically every run.

## Example (illustrative shape only — much smaller than real instances)
`T=6,N=2,E=1,C=1`, endpoint idempotent, no outage, tick `2` ambiguous.
Requests: `(arrival=0)`, `(arrival=0)`. Policy `"1"`, `"1"` (never retry):
tick 0 has demand 2 > `cap_eff=1`; one request (id 0) is admitted and
confirms (`+1`); id 1 is dropped and, with `max_attempts=1`, is abandoned.
`F=1=B`, so `Ratio=0.1`. Policy `"2 3"` for request 1 (retry once, 3 ticks
later) instead reschedules it to tick 3 (clean, since endpoint is
idempotent and healthy there): both requests confirm, `F=2`,
`Ratio=min(1000,200)/1000=0.2`.
