# Correlated Fleet Scheduling: Partition for Expected Makespan

A depot runs **N jobs** on **K identical machines** in parallel. Each machine's
finish time is the sum of the processing times of the jobs assigned to it, and the
**makespan** is the finish time of the slowest machine. You must partition the jobs
to make the makespan small.

The twist: a job's processing time is **not fixed**. Every run, all jobs are
perturbed by a few shared hidden factors (ambient load, a shared bus, a common
data feed). Under this hidden **low-rank covariance** some jobs rise and fall
together, while others move in **opposite** directions. You are given a sample of
past runs and are graded on the **expected makespan** over the fluctuation
distribution.

You write a program that reads one instance and outputs an assignment.

## Input (one JSON object on stdin)

```
{"name": str,
 "n": N,                      # number of jobs (12..18)
 "k": K,                      # number of machines (3..4)
 "s": S,                      # number of probe scenarios
 "probe": [[t_00, ..., t_0{N-1}],   # S rows, each a length-N list of
           ...                       #   sampled processing times (floats >= 0)
           [t_{S-1}0, ...]]}
```

`probe[r][i]` is job `i`'s processing time observed in probe run `r`. These probe
runs are a **training sample** of the hidden distribution; the exact means, factor
loadings, and the scenarios you are graded on are hidden.

## Output (one JSON object on stdout)

```
{"assign": [m_0, m_1, ..., m_{N-1}]}
```

`m_i` is the machine (integer in `[0, K-1]`) that job `i` runs on. Machines may be
left empty. The assignment is **valid** iff `assign` is a list of exactly `N`
integers each in `[0, K-1]`; anything else (wrong length, out of range, non-integer,
a crash, a timeout, or non-JSON) scores **0.0** on that instance.

## Objective — MINIMIZE expected makespan

Your assignment is scored on a **held-out sample H** of scenarios drawn from the
same hidden distribution as the probe (disjoint from it, so memorizing the probe
does not help). For a scenario, `makespan = max over machines of that machine's
total processing time`; your objective is the **mean makespan over the held-out
scenarios**, `q_cand`.

The evaluator normalizes with two references it computes on `H`: `q_base`, the
expected makespan of the index round-robin assignment (`i -> i % K`), and `q_lb`,
the mean of `(total time in the scenario) / K` — the perfectly balanced per-scenario
load, an unreachable lower bound:

```
r = clamp( 0.1 + 0.5 * (q_base - q_cand) / (q_base - q_lb) , 0, 1 )
```

Round-robin scores `~0.1`; every reduction in expected makespan raises `r`. The
per-instance scores are averaged across all instances. `q_lb` cannot be reached by
any single assignment (you cannot balance every scenario at once), so scores stay
well below `1.0` — there is always headroom.

## What actually moves the score

Balancing only the **mean** load per machine (classic Longest-Processing-Time) is
the obvious move, but it is blind to correlation. Two **anti-correlated** jobs put
on the **same** machine cancel each other's peaks — when one spikes the other dips,
so that machine's load stays flat and rarely becomes the bottleneck. Split the same
pair across two machines and **both** become volatile; the max-over-machines picks
up whichever spiked, raising the expected makespan. The probe scenarios carry this
second-order signal: the empirical mean of the max-machine-load over the probe rows
is itself a covariance-aware objective you can optimize directly (seed with LPT,
then refine with single-job moves and pairwise swaps). Several instances are built
so jobs come in anti-correlated pairs with near-equal means — there LPT balances
the means perfectly yet lands far from a covariance-aware partition.

Deterministic scoring only. Time limit 2–5 s, memory ≤ 512 MB.
