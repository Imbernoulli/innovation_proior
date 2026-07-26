# Biogas Digester Fed from Spoiling Stockpiles

## Problem

You operate an anaerobic digester over `T` days. Each day, `K` feedstock
types (`0..K-1`) may be delivered in some amount; every delivery has a
limited shelf life, and once its window passes, whatever is left unused
physically rots and is gone forever. You choose, for every day, how much
of each *currently unspoiled* type to feed into the digester, subject to
the digester's fixed daily processing capacity.

**Chemistry.** Feeding a daily mix with fractions `x_0..x_{K-1}`
(summing to 1 over whatever total mass you feed that day) has a base
value `c . x + sum_{i<j} s_ij * x_i * x_j` -- a linear value per type
plus pairwise synergy/antagonism terms. This is then hit by **substrate
inhibition**: for every type whose fraction exceeds its own threshold
`thr_k`, the value is further multiplied by `max(0, 1 - pen_k*(x_k -
thr_k))`. Call the result `quality(x)` (never negative).

**Adaptation memory.** The digester's microbial community tracks an
internal state `M` (a distribution over the `K` types) that drifts
toward whatever you actually fed: after every day you feed, `M <-
(1-alpha) M + alpha x`. Switching diets is not free: today's realized
value is `max(0, quality(x) - switch_cost * L1(x, M))`, `L1` being the
L1 distance between today's mix and the CURRENT `M` (before today's
update). Day score = (mass fed) x (realized value); total score sums
this over all `T` days.

## Input (stdin)
```
T K
alpha_milli switch_cost_x100 cap
c_0 ... c_{K-1}
s_01 s_02 ... s_0(K-1) s_12 ... s_(K-2)(K-1)      [K*(K-1)/2 values, i<j row-major]
thr_0 ... thr_{K-1}          [per-mille: threshold = thr_k/1000]
pen_0 ... pen_{K-1}
shelf_0 ... shelf_{K-1}      [days a REGULAR delivery of type k stays usable]
M0_0 ... M0_{K-1}            [per-mille, sums to 1000: initial adaptation state]
arr_1_0 ... arr_1_{K-1}      [T lines: today's regular delivery of each type]
...
arr_T_0 ... arr_T_{K-1}
n_spikes
day_1 type_1 amount_1 shelf_1   [n_spikes lines: a SEPARATE bulk
...                                consignment delivered that day, with
                                   its OWN (usually shorter) shelf life --
                                   on top of that day's regular row]
```
`switch_cost = switch_cost_x100 / 100`. `alpha = alpha_milli / 1000`.

## Output (stdout)
`T` lines, each with `K` nonnegative numbers: `feed_t_0 ... feed_t_{K-1}`,
the amount of each type you feed on day `t` (1-indexed days, in input
order).

## Feasibility
Every `feed_t_k` must be a finite number `>= 0`. Internally, a FIFO
queue per type tracks unspoiled stock (oldest delivery consumed first;
any batch -- regular or spike -- past its OWN shelf life is discarded
before that day's feeding). `feed_t_k` may not exceed that day's
available unspoiled stock, and the day's TOTAL mass fed may not exceed
`cap`. Any violation (including wrong token count, or `nan`/`inf`)
scores `0.0`.

## Scoring
Let `F` be your total score as defined above. The checker also builds a
naive reference plan -- feed each day's freshly available stock in its
own arrival ratio, scaled down only as far as needed to fit `cap` -- and
scores it identically for `B > 0`. Since this is a maximization:
`ratio = min(1000, 100*F/B) / 1000`.

## Constraints
`8 <= T <= 16`, `4 <= K <= 6`, time limit 5s, memory 512MB.

## Example
Suppose today `x = (0.3, 0.7)`, `quality(x) = 5` (no inhibition
triggered), and `M = (0.5, 0.5)`: `L1 = 0.4`. With `switch_cost = 3`,
realized value `= max(0, 5 - 3*0.4) = 3.8`. If your stock and `cap`
allow feeding 10 units total at this ratio, today's score contribution
is `10 * 3.8 = 38`.

## Why this is open-ended
There is no known closed-form optimum: the real decision variable is a
whole `T`-day trajectory, not a per-day best mix, because every day's
choice both burns perishable stock (use-it-or-lose-it) and taxes
tomorrow's value through `M`. A per-day optimizer that chases whatever
looks best today pays that tax whenever the locally-best mix drifts, and
cannot defend against an upcoming spoilage deadline until the last
moment. Planning the trajectory as a whole -- a stable target recipe,
deviating only in bounded, deliberate ways to salvage stock about to
expire -- recovers value a myopic solver structurally cannot see.
