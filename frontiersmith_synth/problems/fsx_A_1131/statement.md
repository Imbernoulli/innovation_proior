# The School and the Current

A fisherman tracks one school of fish along a long channel. Positions in the
channel are indexed `0 .. N-1` (think: a huge sorted array of buoy markers).
Each day `t = 1, 2, 3, ...` the school occupies some position, and your only
tool to reach it is a **finger** — a pointer you can leave where it is or
relocate. Leaving the finger `d` positions away from the school costs `d`
(that day's fishing is worse the farther you are); relocating the finger is
free up to a budget tied to how much the current genuinely moves the school,
but excess relocations cost real money.

You are given the first `T_train` days of **noisy position fixes** taken
directly on the school. You must hand back a **closed-form formula that
predicts the school's position on any future day `t`**. Your formula is then
rolled forward, day by day, over the days *after* your training window —
days you never see — to actually run the fishing trip and pay its costs.

## Input (stdin)

```
T_train t N W H
i_1 obs_1
i_2 obs_2
...
i_{T_train} obs_{T_train}
```

`t` is the test id. `N` is the channel length. `W` is the **relocation
hysteresis band** (see below). `H` is the number of graded future days
(days `T_train+1 .. T_train+H`); it is **not** shown to you during training.
Each row `i_k obs_k` is a noisy integer position fix taken on day `i_k`
(`i_k = 1..T_train`).

The school's true position follows a hidden law with three ingredients: a
steady current (roughly linear drift), a seasonal eddy (a sinusoid), and a
slowly strengthening rip current that is a minor wobble across the training
days but keeps accelerating — by the graded window it can dominate the
other terms entirely. **Illustrative FORM only — NOT the hidden law:**
`sqrt(abs(t)) + 4*cos(t)` (the real law's shape must be discovered from the
data — do not assume this example's functions or term count).

## Output (stdout): one line

```
EXPR <expression>
```

`<expression>` is an arithmetic expression in the single variable `t`, using
`+ - * / **`, parentheses, numeric constants, and the unary functions `sin`,
`cos`, `sqrt`, `abs`, `exp`. At most 120 expression nodes.

## Grading (rolled forward over days `T_train+1 .. T_train+H`)

Start the finger at your prediction for day `T_train+1`. On every later
graded day `t`, compute your prediction `p = round(expr(t))` (clamped to
`[0,N-1]`). **If `|p - finger| > W`, relocate the finger to `p`** (this is
the only way the finger ever moves — it can only follow your formula, never
the school's true position, which you cannot see). Otherwise the finger
holds. Each day, charge `|finger - true_position|` (the true position is
never revealed; only its noisy training-window fixes are).

Relocations cost nothing up to a **free budget**: the number of times the
school's *true* position itself would cross the band `W` over the graded
window — a fixed property of the hidden law, unrelated to your formula, i.e.
how often relocation is genuinely unavoidable. Every relocation beyond that
costs an extra `3*W`. A forecast tracking the real drift relocates close to
that unavoidable count; one that jitters, or lags and over-corrects,
relocates more and pays for it.

## Objective (minimise)

Let `F` be your total charge above. Let `B` be the same charge for the
trivial policy "freeze the finger at your last training-day fix forever"
(no relocations). The checker prints

```
Ratio = min(1000, 100*B / F) / 1000
```

Matching the frozen-finger baseline scores `Ratio ≈ 0.1`; a materially lower
`F` raises the score toward (but never reaching) `1.0` — irreducible sensor
noise on the graded days keeps a perfect score out of reach.

## Why the training window is a trap

Over `T_train` days the rip current is only a small addition to the steady
current and the eddy, so a plain straight-line (or straight-line + one
sinusoid) fit looks almost as good as anything else. But the graded window
runs on for **twice as many days again**, where that same term dominates. A
predictor that only ever captures average drift under/over-shoots by a
growing margin as the window progresses, driving up both the distance
charge and, once its forecast wanders enough, the relocation count.

## Constraints

Time limit 5 s, memory 512 MB. `T_train` is a few hundred rows. `N` and `H`
fit in 32-bit integers. Scoring is fully deterministic.
