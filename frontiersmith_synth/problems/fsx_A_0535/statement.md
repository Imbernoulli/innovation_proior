# Corridor Tolls: Steering Selfish Drivers to the Social Optimum

A city runs a corridor of **M parallel routes** between one origin and one destination
(highway lanes, a bridge, side streets, a tunnel...). Route `e` has a **superlinear**
travel-time (latency) function

```
l_e(f) = a_e * f**p_e + b_e        (a_e > 0, b_e >= 0, p_e >= 1)
```

where `f` is the flow (vehicles/hour) on route `e`. Drivers are **selfish**: given a fixed
per-route toll `tau_e >= 0`, every driver takes the route of smallest experienced cost
`l_e(f_e) + tau_e`. Traffic settles into a **user (Wardrop) equilibrium**: all used routes
share one common experienced cost and no driver can switch and improve. The total demand `D`
splits across the routes accordingly.

You set **one toll per route**. Tolls are internal transfers — they only re-route drivers,
they do **not** count in the social cost. The **true social cost** of a flow is the toll-free
total travel time

```
Cost(f) = sum_e f_e * l_e(f_e).
```

Your job: choose tolls so that the selfish equilibrium they induce **minimises the social cost**.

## The catch — demand is a distribution

Demand `D` is uncertain, drawn from a distribution. You commit to **one** toll vector and it is
graded on a **held-out set of demand scenarios you never see**. You are given a sample of
`train_demands` from the same distribution to design your tolls. A toll that is perfect for one
demand is wrong for another, so no fixed toll reaches the per-scenario optimum — there is always
headroom.

## Input (public instance, one JSON object on stdin)

```
{"name": str,
 "m": M,                       # number of parallel routes
 "a": [a_0, ..., a_{M-1}],     # latency scale per route (a_e > 0)
 "b": [b_0, ..., b_{M-1}],     # free-flow latency per route (b_e >= 0)
 "p": [p_0, ..., p_{M-1}],     # latency exponent per route (p_e >= 1; MAY differ across routes)
 "train_demands": [D_0, ...]}  # sample demands from the scenario distribution
```

## Output (one JSON object on stdout)

```
{"tolls": [tau_0, ..., tau_{M-1}]}     # tau_e finite, 0 <= tau_e <= 1e9
```

Wrong length, a negative / NaN / infinite entry, a crash, a timeout, or non-JSON output makes
that instance score **0.0**.

## Scoring (deterministic; averaged over the hidden eval demands)

For each hidden demand `D`, the evaluator computes three toll-free total costs: `L_zero` at the
**untolled** equilibrium (do nothing), `L_so` at the **per-scenario social optimum** (the ideal),
and `L_cand` at the equilibrium **induced by your tolls**. Averaging each over the eval demands,

```
r = clamp( 0.1 + 0.9 * (L_zero - L_cand) / (L_zero - L_so),  0, 1 ).
```

Doing nothing scores ~0.1. Reaching the per-scenario ideal would score 1.0, but a single fixed
toll **cannot** be optimal for every demand in the distribution, so even the best principled toll
stays strictly below 1.0. Tolls that steer flow **worse** than doing nothing score below 0.1. Your
final score is the mean of `r` over the instances.

## What matters

A driver joining route `e` suffers their own delay **and** slows everyone already there. The
externality they ignore is the flow times the **marginal** latency,
`f_e * l_e'(f_e) = a_e * p_e * f_e**p_e`. This is **not** the same as the observed congestion level
`a_e * f_e**p_e`: it carries the exponent `p_e`. When routes have different exponents, that factor
is the whole game — and the flow at which you evaluate the externality matters too. The exact
coefficients `a_e, b_e, p_e` and the demand spread live in the input; read them and exploit them.
