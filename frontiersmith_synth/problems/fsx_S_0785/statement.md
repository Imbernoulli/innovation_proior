# Ghost Train Timetable

A railway dispatcher keeps seeing a "ghost train" pull into platforms
according to a rule nobody wrote down. You have a logbook of the platform
number `a[t]` it used at each tick `t = 0, 1, 2, ...`, and you must learn a
**prefetch predictor** that names candidate platforms for future ticks so the
signal box can pre-clear them.

The logbook you receive covers only the train's **early service epochs**.
Timetables like this are organized into fixed-length epochs: within an epoch
the train shunts through platforms in a fast, repeating local pattern, while
the epoch **as a whole** is anchored to a platform that drifts from one epoch
to the next by a rule tied to the epoch counter. You will be graded on
**later epochs the logbook never shows you** — the dispatcher must anticipate
epochs the ghost train hasn't run yet, not just replay what it already saw.

## Input (stdin)

```
N P testId
a[0]
a[1]
...
a[N-1]
```

`P` is the epoch length (ticks per epoch); `N` is a multiple of `P`. The
graded epochs are strictly later than anything in this log.

## Output (stdout): a two-slot predictor

Emit exactly two lines:

```
SLOT1 <expr>
SLOT2 <expr>            (or:  SLOT2 NONE)
```

Each `<expr>` is an arithmetic expression using integer constants and the
variables `t` (the query tick), `h1,h2,h3,h4` (the actual platforms used at
ticks `t-1,t-2,t-3,t-4` — the most recent history, handed to you at grading
time), the operators `+ - * // % **` and parentheses. `**` exponents must be
constant integers in `[0,4]`. No other names, calls, or float literals are
allowed. `SLOT2 NONE` means you decline to use the second slot.

**Illustrative FORM only — NOT the hidden timetable:**
```
SLOT1 h1 + 2 * ( h2 - h3 )
SLOT2 t % 17
```
This only shows the syntax; the real rule has a different shape you must
discover from the log.

## Feasibility

Both lines must parse under the grammar above (known names only, integer
constants with `|c| <= 1e6`, bounded expression size). Any violation scores
`0`.

## Objective (maximize)

The grader queries `H` future ticks, each in an epoch your log never
reached. At each query it hands your expressions the TRUE recent history
`h1..h4` and evaluates them at that `t`. Let `true` be the actual platform.
Each slot earns partial credit for being close, not just exact:

```
credit1 = 1 / (1 + |SLOT1 - true| / 2)
credit2 = 0.7 / (1 + |SLOT2 - true| / 2)      (0 if SLOT2 is NONE)
credit  = max(credit1, credit2)
```

(slot 2 costs a budget discount: using it to hedge only pays off if it lands
closer than slot 1 by a wide enough margin). `F` is the mean credit over all
`H` queries. The grader also computes its OWN naive one-slot predictor —
`platform(t-1) + (platform(t-1) - platform(t-2))`, i.e. "assume the most
recent step repeats" — as an internal baseline `B` (it never sees the
timetable law either). The printed score is

```
Ratio = min(1000, 100 * F / B) / 1000
```

A predictor that only looks at recent local deltas tracks the fast local
pattern about as well as this baseline, and stays close to it. A predictor
that recognizes platforms sampled once per epoch isolate the epoch-level
drift rule, fits that rule, and extrapolates it to unseen epochs, scores
well above baseline. Some jitter in the log is built from bit-mixing your
expression grammar cannot represent — nobody's `Ratio` reaches `1.0`.

## Constraints

Time limit 5 s, memory 512 MB. `N` is a few thousand rows across the 10
tests. Scoring is fully deterministic and O(H).
