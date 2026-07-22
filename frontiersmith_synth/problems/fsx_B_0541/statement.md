# Dressing the Wheel

A precision grinding shop has a single wheel and a batch of `N` jobs to run today.
The wheel accumulates **wear** as it grinds. A worn wheel is slow: the more worn it
is, the longer *every* job takes. You choose the order in which the jobs run and may
periodically **re-dress** the wheel (a maintenance stop that restores it to pristine).

The twist: some jobs are **soft dressing materials**. Grinding them barely loads the
machine and actually *cleans the wheel* — running one is maintenance in disguise. The
challenge is to schedule production and this hidden maintenance on one timeline.

## The machine

The wheel has an integer **wear** `w`, starting at `w = 0`. Job `i` has a **size**
`s_i` (positive) and a **hardness** `d_i` (an integer that may be negative).

Processing job `i` while the current wear is `w` takes time

```
    s_i * (1 + w)^2
```

(the wear penalty is superlinear). Afterwards the wear changes:

```
    w  <-  clamp(w + d_i, 0, W_max)
```

Hard jobs (`d_i > 0`) roughen the wheel; **soft dressing jobs (`d_i < 0`) restore it**.
Wear never drops below `0`, and a fully worn wheel is capped at `W_max` (it cannot get
worse than that).

A **re-dress** may be inserted anywhere between jobs. It costs a fixed time `T_r` and
sets `w <- 0`.

## Input (stdin)

```
N T_r W_max
s_1 d_1
s_2 d_2
...
s_N d_N
```

`1 <= N <= 60`; `T_r`, `W_max`, and every `s_i` are positive integers; `d_i` is a
nonzero integer.

## Output (stdout)

A single schedule: a whitespace-separated list of tokens. A token `j` in `1..N` means
"process job `j` now"; a token `0` means "re-dress now". **Every job index `1..N` must
appear exactly once**; re-dress tokens `0` may appear any number of times (including
none), anywhere. Any other token, a missing job, or a repeated job makes the schedule
infeasible.

## Objective

Minimize the **total time** = sum of all processing times plus `T_r` for each re-dress,
under the wear dynamics above.

## Scoring

Let `F` be your schedule's total time and let `B` be the total time of the **baseline
schedule** — the jobs run in the given input order with no re-dress. Since this is a
minimization, your score is

```
    Ratio = min(1000, 100 * B / F) / 1000
```

So the baseline scores `0.1`, a schedule ten times faster caps at `1.0`, and lower total
time always scores higher. An infeasible or non-integer output scores `0`.

## Constraints

* Time limit 5 s, memory 512 MB. Each input is at most a few kilobytes.
* Scoring is exact integer arithmetic and fully deterministic.

## Worked example

Input (`W_max = 4` here, so no clamping happens in this small case):

```
3 20 4
10 3
1 -2
10 1
```

Baseline order `1 2 3` (no re-dress): job 1 at `w=0` costs `10`, then `w=3`; job 2 at
`w=3` costs `1*16=16`, then `w=1`; job 3 at `w=1` costs `10*4=40`. `B = 66`.

Alternative schedule `1 3 2` (still no re-dress): job 1 costs `10`, `w=3`; job 3 at `w=3`
costs `10*16=160`, `w=4`; job 2 at `w=4` costs `1*25=25`. Total `195` — worse.

Schedule `3 2 1`: job 3 at `w=0` costs `10`, `w=1`; job 2 (the dressing job) at `w=1`
costs `1*4=4`, `w=0`; job 1 at `w=0` costs `10`. Total `24`, so `Ratio = 100*66/24/1000
= 0.275`. Notice the soft job was used as maintenance *between* the two hard jobs — no
`T_r` was paid. Whether an explicit re-dress ever helps depends on `T_r` and on how much
restoring capacity the soft jobs give you.
