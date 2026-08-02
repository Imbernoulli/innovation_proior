# Hierarchical Rate Limiting Under a Cardinality-Explosion Attack

## Problem

You operate an API gateway. A historical request trace has already been
audited: every request is labeled *legitimate* or *abusive* (e.g. from a
fraud-scoring feed). You must design a rate-limiter **configuration** that,
when replayed against this exact trace, admits as much legitimate traffic as
possible while suppressing abusive traffic -- subject to a hard **memory
budget**: the gateway can afford to maintain state (an independent
token-bucket counter) for at most `M` logical buckets, no more.

The trace may contain a **cardinality-explosion attack**: thousands of
distinct, mostly one-shot request keys, far more than `M`. Tracking every key
exactly is therefore not always an option -- your design must decide which
keys deserve a dedicated bucket and how to safely pool the rest.

## Input (stdin)

```
T M A B P
R
t_1 key_1 label_1
...
t_R key_R label_R
```
`T` = number of discrete time steps (times satisfy `1<=t<=T`). `M` = memory
budget (max total buckets you may declare). `A,B,P` are fixed integer hash
constants used for group routing (see below). `R` = number of requests.
Each of the next `R` lines gives one request: time `t`, integer key `key`
(a client identifier), and `label` (`1`=legitimate, `0`=abusive). Requests
are given in nondecreasing order of `t`; ties keep the given order.

## Output (stdout) -- your limiter configuration

```
H G
key_1 cap_1 rate_1
...
key_H cap_H rate_H
gcap grate
```
`H` = number of keys you isolate with their own dedicated bucket (0 or
more), `G` = number of shared "pool" buckets (must be >=1). **Feasibility
requires `H+G <= M`.** Each explicit bucket is `key cap rate` (integers,
`0<=cap,rate<=5000`, no duplicate keys). The final line `gcap grate` is the
single template (integers in the same range) applied to every one of the `G`
shared buckets. Any key you did NOT list explicitly is routed to shared
bucket number `((key*A+B) mod P) mod G` and served by that bucket's
`(gcap,grate)` counter (independent per-bucket state, same template).

## Replay / scoring mechanics

Each bucket (explicit or shared) starts with `tokens = cap`. Processing
requests for that bucket in time order, on a request arriving at time `t`
(previous update at `last`): `tokens = min(cap, tokens + rate*(t-last))`;
if `tokens>=1` the request is **admitted** (`tokens -= 1`), else it is
**rejected**. Let `good`/`abusive` be the counts of admitted requests with
each label. Your raw objective is `F = max(0, good - abusive)`.

The checker also replays the same trace through its own trivial baseline
configuration (a single non-isolating shared bucket, `H=0, G=1`, small fixed
`cap,rate`) and counts how much LEGITIMATE traffic that baseline alone
admits -- call it `B` (always positive; some legitimate traffic gets through
any minimally-alive bucket). Your final score is
`Ratio = min(1000, 100*F/B) / 1000`, clipped to `[0,1]`. A design that does
no better than the naive baseline scores low; meaningfully beating it on the
*net* good-minus-abusive measure scores higher, capped below `1.0` so there
is always headroom above any reference solution.

## Feasibility

`H>=0`, `G>=1`, `H+G<=M`, all values integers in range, no duplicate
explicit keys, and the output must contain exactly the expected number of
tokens (no missing or extra data). Any violation scores `0.0`.

## Example (worked, illustrative shape only -- not from an actual test)

Suppose `T=5, M=2` and 4 requests: `(1,k1,1) (2,k1,1) (3,k2,0) (4,k2,0)`,
with `k1` legitimate and `k2` abusive. Output `1 1` / `k1 5 1` / `1 0`
isolates `k1` with a generous bucket (admits both its requests) and routes
`k2` into the single shared bucket with `cap=1,rate=0` (its bucket starts
with 1 token, admits the first `k2` request, then is empty and rejects the
second). Replay: `good=2, abusive=1`, `F=max(0,2-1)=1`. If the checker's own
baseline (one shared bucket for everyone, no isolation) admits, say,
`good=1` legitimate request in total, `B=1`, so the score is
`min(1000,100*1/1)/1000 = 0.1` -- here illustrating the mechanics only, not
the actual baseline parameters used at scale.

## Constraints

`5<=T<=1200`, `25<=M<=40`, `1<=R<=60000`, keys are positive integers,
time limit 5s, memory 512MB.
