# Headroom Gate: Admission Control for a Shared Compute Cluster

You run the admission gate for a shared compute cluster over one arrival stream of `N`
jobs. The cluster is one **work-conserving server** that runs admitted jobs
**non-preemptively in arrival order** (FIFO by arrival time, ties by index). You do not
reorder the server — **admission is your only lever**: for each job you output
`admit (1)` or `drop (0)`.

Job `i` has arrival `a_i`, service size `s_i > 0`, value `v_i > 0`, and an absolute
deadline `d_i` (the time by which it must **finish**). The entire stream is given up
front, so every future arrival — including premium jobs that land soon — is **known**.

## Queue-latency buildup (you inflict it on yourself)

Over the admitted jobs sorted by `(a_i, i)`, the server runs:

```
free = 0
start_i  = max(free, a_i)
finish_i = start_i + s_i
free     = finish_i
```

Admitting a job now therefore **raises the finish time of every admitted job behind it**.
Take on too much and you congest yourself.

## SLA deadline-penalty ramp

- An admitted job that finishes **on time** (`finish_i ≤ d_i`) earns its value `v_i`.
- An admitted job that **misses** earns nothing and pays a penalty that **ramps with
  lateness**: `beta · (finish_i − d_i)`.
- A dropped job earns 0 and costs 0.

## Objective (maximize)

```
J = Σ_{admitted, finish ≤ d} v_i  −  beta · Σ_{admitted, finish > d} (finish_i − d_i)
```

Admitting nothing yields `J = 0`. Admitting a set that misses deadlines is **legal but
penalized** — keeping the set on time is your job, not a validity rule. The ramp slope
`beta` lives in the input; read and exploit it.

## Input (one JSON object on stdin — the public instance)

```
{"name": str, "N": int, "beta": float,
 "a": [a_0 … a_{N-1}],   # arrival times (non-decreasing)
 "s": [s_0 … s_{N-1}],   # service sizes (> 0)
 "v": [v_0 … v_{N-1}],   # values (> 0)
 "d": [d_0 … d_{N-1}]}   # absolute deadlines (finish-by)
```

## Output (one JSON object on stdout)

```
{"admit": [x_0 … x_{N-1}]}   # each x_i in {0, 1}
```

`admit` must have exactly `N` entries, each `0` or `1`. Any violation, crash, timeout,
or non-JSON scores 0 on that instance.

## Scoring (deterministic; no wall-time)

For each of 10 fixed seeded instances your `J` is normalized against a **loose,
unreachable** bound — the total value if *every* job were served on time:

```
hi = GAIN · Σ_i v_i
r  = clamp( 0.1 + 0.9 · J / hi ,  0, 1 )
```

Admitting nothing scores exactly `0.1`; because one server cannot serve every job on
time, even an oracle stays well below `1.0`. Your score is the **mean of `r`** over the
10 instances, which span headroom-trap single-premium cases, mildly congested mixed
cases, uncongested calm cases, and a held-out twin-premium regime.

## What to notice

"Admit anything with positive value" fills the FIFO queue so predictably: a burst of
small low-value jobs busies the server just as a high-value job with a tight deadline
arrives, so that premium job starts late and is **both forfeited and penalized**.
Because the whole stream is public you can instead **reserve headroom** — treat
admission as choosing an on-time subset, let the premium jobs claim server time first,
and refuse a marginal low-value job exactly when its size would push the FIFO tail past
a deadline you care about. Value banked against congestion inflicted: that is the trade.
