# Counterdesk Commitments: Repair Shop Turnaround Quotes

## Setting

You run the front desk of a repair shop with `C` identical benches. Jobs walk in
one at a time, in a fixed order. Job `i` arrives with a **value** `v_i` (revenue if
delivered on time), a **service time** `s_i` (bench-minutes actually needed), and a
**patience** `p_i` (the longest turnaround the customer will tolerate before going
to a competitor). For every job you must either **reject** it, or post a **binding
delay quote** `d_i`: "ready in `d_i` time units."

A quote is a real commitment, not small talk:

- **Balking best response.** The customer joins iff `d_i <= p_i`; otherwise they
  walk away (no value, no cost, no schedule impact).
- **Binding delay quotes.** If they join, the shop is on the hook to actually
  finish by `arrival_i + d_i`. There is no renegotiating later.
- **Admission control.** The C benches process joined jobs by
  **earliest-promised-deadline-first** (EDF): whichever joined-but-unfinished job
  has the tightest outstanding promise gets the next bench that frees up, no
  matter which order jobs arrived in. A rush of new jobs with tighter promises
  can and will bump an already-promised job out of its slot.

Because dispatch is EDF, a delay quote is really an **option written on your own
future bench capacity**: promising a short turnaround buys that job priority
ahead of time, and every quote you issue changes who wins that priority fight
later. The input traces contain bursts of near-simultaneous, high-value,
impatient jobs — the shop physically cannot honor every promise it could make
during one of these, so which promises you choose to make (and how much slack
you leave yourself) determines your score.

## Candidate program contract

Standalone program: read ONE JSON object (the public instance) from **stdin**,
write ONE JSON object (your answer) to **stdout**.

### Public instance (stdin)

```json
{
  "name": "burst-a",
  "capacity": 2,
  "n": 40,
  "jobs": [
    {"t": 3, "value": 42.0, "service": 4, "patience": 9},
    {"t": 5, "value": 88.0, "service": 3, "patience": 4},
    ...
  ]
}
```
`jobs` is sorted by arrival time `t` ascending (ties allowed); `t`, `service` are
integers, `patience` is an integer `>= service`, `value` is a positive number.

### Answer (stdout)

```json
{
  "decisions": [
    {"action": "reject"},
    {"action": "quote", "delay": 7},
    ...
  ]
}
```
`decisions` must have **exactly** `n` entries, aligned with `jobs`. Each entry is
either `{"action": "reject"}` or `{"action": "quote", "delay": d}` where `d` is a
plain integer with `service_i <= d <= 100000`. Any malformed entry, wrong length,
non-JSON output, a crash, or a timeout scores the whole instance `0.0`.

## Mechanics and scoring (deterministic)

A `"quote"` job **joins** iff `d_i <= patience_i` (otherwise it balks: contributes
`0`). All joined jobs across the whole instance are then dispatched together by
the frozen **earliest-promised-deadline-first** simulation across the `C`
benches (a bench, once free, always takes the joined-and-waiting job with the
tightest `arrival + delay` deadline; ties break by arrival time then job index).
For each joined job, compare its true completion time to its own deadline:

- finished **on or before** `arrival_i + d_i` → contributes `+v_i`
- finished **after** `arrival_i + d_i` (a broken promise) → contributes `-1.5 * v_i`

Rejected and balked jobs contribute `0`. Let `cand` be the total over all jobs.

The evaluator also runs, itself, a weak **mean-field forecast** desk policy: it
tracks a single pooled backlog counter drained at rate `C`, quotes exactly that
forecast (no strategic slack), and admits iff the forecast is within patience.
Let `b` be that policy's realized total, and `U` the sum of every job's value (an
unreachable ceiling — capacity can never serve every job in a burst). Then:

```
r = clamp( 0.1 + 0.9 * (cand - b) / max(1e-9, U - b), 0, 1 )
```

Reproducing the mean-field policy scores `~0.1`; beating it scores higher; doing
worse (e.g. racking up violation fines) can push the score toward `0`. `Ratio`
is the mean of `r` over 10 seeded instances (varying `C`, trace length, and
burst structure, some held out at larger scale); `Vector` lists the per-instance
scores.

## Constraints

`2 <= C <= 4`, `40 <= n <= 80`, time limit 5s, memory 512m.

## Hints

A quote you *know* the customer will accept costs nothing extra to make as loose
as `patience_i` allows — tighter quotes only add violation risk. The harder
question is *which* jobs to promise at all: a burst brings more jobs than `C`
benches can honor within their patience, so admitting jobs greedily in arrival
order can crowd out (or get crowded out by) higher-value jobs nearby in the
trace. The whole trace is given to you up front — you don't have to decide
job-by-job with no look-ahead.
