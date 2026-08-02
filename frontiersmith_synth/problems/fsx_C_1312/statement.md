# Asking the Question That Tells You the Most: Exposure-Aware Adaptive Item Selection

## Story

A testing program shares ONE item bank across an entire **exam season**: N
examinees, processed in a fixed order, each answering an adaptively-selected
test of L items out of the bank's M items. Each item has an IRT
discrimination `a` and difficulty `b` (2PL model). After every response, the
engine updates a running ability estimate via a standard Bayesian (EAP)
posterior update over a fixed grid — not yours to control. **Your job is
only to choose, at each step, which item to administer next**, by submitting
a compact prioritization **policy** (four numbers) that this evaluator
applies at every decision point using its own live-tracked state (current
estimate + current per-item exposure count).

The trap: once an item is administered to more than a `cap_frac` fraction of
the season's examinees, it becomes **compromised** — its answer leaks — and
every later administration draws a response from a constant high pass rate
(`leak_pass_rate`), independent of true ability, instead of the item's real
2PL model. The ability update, unaware of the leak, reads a "correct" answer
on a hard item as strong evidence of high ability, so a compromised item
doesn't just stop informing — it **biases** every later examinee who draws
it. Always picking the maximum-information item at the current estimate (the
standard recipe) ignores this: whenever true ability is concentrated rather
than spread out like a generic population, it repeatedly reaches for the same
few locally-best items, burns through their cap early, and corrupts estimates
for the rest of the season. The insight this problem rewards is trading a
little per-decision information for long-run **pool survival**.

## Isolation

Your program runs as an **isolated subprocess**: read one JSON *public
instance* from stdin, write one JSON *policy* to stdout. You never see hidden
true abilities or the evaluator's live state.

## Public instance (stdin)

```json
{
  "M": 40, "L": 9, "N": 80,
  "items": [{"a": float, "b": float, "cap_frac": float}, ...] ,
  "typical_demand_hint": [float, ...],
  "leak_pass_rate": float,
  "prior_mean": float, "prior_sd": float,
  "selection_anchor_jitter_sd": float,
  "seed": int
}
```
`typical_demand_hint[i]` is item i's admin-rate under a **generic
N(0,1)-ability reference population** running pure maximum-information
selection with no exposure control — it can under-predict real demand when
this season's true ability distribution is not that generic one.
`selection_anchor_jitter_sd` is a fixed per-examinee idiosyncratic offset
(does not bias the actual estimate) applied only to which item looks best.

## Answer (stdout)

`{"info_weight": float, "exposure_weight": float, "exposure_shape": float,
"hint_trust": float}` — clipped to `[0,50]`, `[0,50]`, `[0,6]`, `[0,1]`
respectively. Non-finite / missing / wrong type → instance score 0.

## Selection rule (evaluator-applied, at every step, for every not-yet-used item i)

```
p        = 1 / (1 + exp(-a_i * (theta_hat + jitter - b_i)))   # at the selection anchor
info_i   = a_i^2 * p * (1-p)
usage_i  = admins_so_far_i / cap_count_i                       # THIS policy's own live count
hint_i   = typical_demand_hint[i] * N / cap_count_i             # forecast, same units
pred_i   = hint_trust*hint_i + (1-hint_trust)*usage_i
score_i  = info_weight*info_i - exposure_weight * max(pred_i,0)**exposure_shape
```
Administer `argmax_i score_i` (ties → smallest index). Once
`admins_so_far_i > cap_count_i` (`cap_count_i = ceil(cap_frac_i * N)`),
every further administration of item i uses the leaked response model
instead of the real one — silently, from the engine's point of view.

## Objective

Per instance: `obj = -mean_n (theta_hat_final_n - theta_true_n)^2` (negative
mean-squared ability-estimation error over the N examinees). The evaluator
compares your `obj` against two references it computes itself: a
non-adaptive fixed-order baseline (`info_weight=exposure_weight=0`;
`obj_base` → `r=0.1`) and a loose, unreachable upper bound — pure
maximum-information selection with the exposure cap removed entirely
(`obj_upper` → `r` up to ~1.0):

```
r = clamp(0.1 + 0.9*(obj - obj_base) / max(obj_upper - obj_base, 1e-6), 0, 1)
```

`Ratio` is the **arithmetic mean** of `r` over 10 instances: several calm
seasons (true ability spread like the generic forecast assumes) and several
trap seasons — narrow, bimodal, mid-season-shifting, very tight-cap, or
far-from-generic true-ability distributions — where several of the bank's
best-matched items are shared by far more examinees than the generic forecast
predicts. On the sharpest of these, a policy that always chases maximum
information sacrifices most of the season's accuracy once its favorite items
are compromised, while a policy that reads the cap tightness and throttles
popular items pre-emptively (without over-trusting the population-agnostic
forecast) keeps serving accurate estimates throughout. **Maximize `Ratio`.**
