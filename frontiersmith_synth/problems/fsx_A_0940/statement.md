# Budgeted Tastings: Mapping a Hidden Phase Plateau

**Family:** calorimeter-mixing-probe · **Format:** B (isolated heuristic evaluation) ·
**Objective:** maximize prediction quality

## Story

A calorimetry bench holds three reagents: hot solvent **A** (initial temperature `TA`), a cold
phase-change reagent **B** (initial temperature `TB0`, below its own melting point), and a
neutral diluent **C** (initial temperature `TC`). Mixing nonnegative masses `mA, mB, mC` lets
heat flow until the mixture reaches one common final temperature.

Ignoring latent heat, energy conservation gives the "naive" linear prediction
`T_lin(m) = (mA*TA + mB*TB0 + mC*TC) / (mA+mB+mC)`
— a weighted average, computable from the public initial temperatures alone, no experiment
needed. **This is exactly right whenever the mixture never gets warm enough to trouble B's
melting point.** But reagent B has a hidden melting point `T*` and a hidden latent heat: if
`T_lin(m)` would fall *inside* a narrow band above `T*`, part of B only partially melts,
absorbing the excess energy, and the true final temperature **locks flat at `T*`** instead of
continuing to rise — a genuine plateau, a low-dimensional manifold in composition space. Once
enough excess energy is present to fully melt B, the temperature resumes rising, but now
systematically *below* `T_lin` (by an amount tied to how much of B's mass had to melt) — so
`T_lin` is exact below the transition, flat-wrong across the narrow plateau, and offset-wrong
everywhere hotter. Precisely:

Let `xB = mB/(mA+mB+mC)` and let `T*`, `ell` (latent-heat scale) be fixed but hidden per
instance.
- if `T_lin(m) <= T*`: final temperature = `T_lin(m)` (cold, no melt)
- elif `T_lin(m) < T* + ell*xB`: final temperature = `T*` (**plateau** — partial melt)
- else: final temperature = `T_lin(m) - ell*xB` (hot, fully melted)

The plateau band's width scales with `xB`, so it is a narrow, easy-to-miss manifold in
composition space unless you deliberately hunt for it.

## Protocol (your program is invoked once per round)

You get a budget of `max_experiments = 18` mixing experiments, spent across `R = 4` adaptive
rounds; each round you see every reading so far and choose the next batch (a round may spend
the whole remaining budget). Read ONE JSON object from stdin and write ONE JSON object to
stdout.

**Query round** (`phase == "query"`): stdin has
`{"phase":"query","materials":{"TA","TB0","TC"},"bounds":{"Tstar_lo","Tstar_hi","ell_lo","ell_hi"},"budget":{"max_experiments","mass_cap_per_experiment"},"budget_left","round","R","max_this_round","history":[{"mA","mB","mC","T_f"},...]}`.
`bounds` is a *hint* window — `T*` lies in `[Tstar_lo, Tstar_hi]` and `ell` in
`[ell_lo, ell_hi]`, but the exact values are secret. Reply
`{"experiments":[{"mA":x,"mB":y,"mC":z}, ...]}` with nonnegative, finite masses; each
experiment's `mA+mB+mC` must be `> 0` and `<= mass_cap_per_experiment` (checked with a tiny
floating-point tolerance, ~1e-6; well outside that band it is silently skipped, still charged
against budget); at most `min(budget_left, max_this_round)` entries per round are honored. A
malformed entry (wrong type, negative, non-finite) scores the **whole
instance 0.0**.

**Predict round** (`phase == "predict"`): stdin additionally has
`"test_mixes":[{"mA","mB","mC"}, ...]` (a frozen, unseen suite). Reply
`{"predictions":[p_0, ..., p_{K-1}]}`, one finite real number per test mix, same order. Wrong
length / type / non-finite scores **0.0**.

Any crash, timeout, non-JSON, or wrong shape on any round scores that instance **0.0**.

## Scoring (deterministic)

Per instance, with true final temperature `T_f`:
```
err_ref = mean_j |T_lin(test_mixes[j]) - T_f(test_mixes[j])|   # zero-experiment guess
err     = mean_j |pred_j - T_f(test_mixes[j])|
quality = clip(1 - err/err_ref, 0, 1)
r       = 0.10 + 0.82 * quality                                 # cap 0.92, floor 0.10
```
Predicting `T_lin` everywhere (no experiments) scores exactly `0.10`. Your total score is the
mean of `r` over 10 fixed instances (some concentrate most of their test mixes near the hidden
transition — the manifold *dominates* their test error).

## Isolation

Your program runs in a fresh sandboxed subprocess via `isorun.run_candidate` and sees only the
public fields above. `T*`, `ell`, and the true `T_f` function live only in the evaluator's
parent process.
