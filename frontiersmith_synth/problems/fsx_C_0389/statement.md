# Solar-Farm Inverter Bench — Design a Drop-in Activation for the Inverter-Response Surrogate

## Setting

A solar-farm operator runs cheap CPU **surrogate models** that map a handful of inverter
telemetry channels (irradiance, cell temperature, DC voltage/current, phase, ...) onto a
scalar **response** of the inverter. Different responses obey different nonlinear physical
laws — efficiency curves, AC-power clipping, thermal derating, MPP-tracking loss, harmonic
distortion — so a single linear model is hopeless.

Every surrogate shares the same fixed pipeline, an "extreme learning machine":

1. standardize the `D` raw channels (train statistics),
2. apply a **frozen random projection** to `H` hidden units,
3. pass the pre-activations through **one elementwise activation `g(·)`**,
4. standardize the `H` feature columns (train statistics), append an intercept,
5. solve a ridge head in closed form.

The **only** component the operator designs is the activation `g` — the atomic, transferable
piece reused by every surrogate on the farm. Your job is to invent a good `g`.

## What you design

One drop-in scalar activation, written as a small sum of whitelisted basis functions:

```
g(x) = sum_k  w_k * base_k( a_k * x + b_k )
```

`base_k` is chosen from `activation_bases` (see below); `a_k, b_k, w_k` are real coefficients.
You design `g` **blind**: you never see any telemetry, any target, the random projection, or
the held-out setting — exactly like inventing ReLU / GELU / Swish rather than tuning to a
dataset. Feature columns are standardized after `g`, so the overall scale of `g` is irrelevant;
what matters is its **shape** and the relative weighting of your components.

## Public instance (stdin, one JSON object)

```json
{
  "name": "farm03",
  "input_dim": 6,
  "hidden_units": 50,
  "ridge_lambda": 0.5,
  "n_public_settings": 3,
  "n_hidden_settings": 1,
  "activation_bases": ["identity","relu","tanh","sigmoid","gauss","sin","abs","swish","softplus","cube"],
  "spec": "g(x)=sum_k w_k*base_k(a_k*x+b_k); return {'components':[{base,a,b,w},...]}, 1..8 entries."
}
```

Basis definitions (scalar `u`, internally clipped to `[-20,20]`):
`identity`=u, `relu`=max(0,u), `tanh`, `sigmoid`=1/(1+e^-u), `gauss`=e^(-u^2), `sin`,
`abs`=|u|, `swish`=u·sigmoid(u), `softplus`=ln(1+e^u), `cube`=u^3.

## Answer (stdout, one JSON object)

```json
{"components": [{"base": "abs", "a": 1.0, "b": 0.0, "w": 1.0},
                {"base": "gauss", "a": 1.2, "b": 0.0, "w": 1.0}]}
```

`components` must be a list of **1 to 8** dicts. Each needs a `base` in the whitelist and finite
floats `a`, `b`, `w` with `|a| ≤ 10`, `|b| ≤ 10`, `|w| ≤ 10`. Any wrong shape, unknown base,
out-of-range or non-finite number, a crash, a timeout, or non-JSON ⇒ the run scores **0.0**.

## Objective & scoring (deterministic)

Each instance holds **3 public inverter settings plus 1 held-out setting** (its physics is never
revealed). For each setting the evaluator builds the surrogate above with **your** `g`, and
computes the normalized test error `nmse = mean((y − ŷ)²) / var(y)` (lower is better). The
per-instance objective is the **geometric mean of `nmse` over all four settings** — so you must
do well *across* settings, including the one you can't see, not just tune one.

Normalization against the evaluator's own trivial component (the identity activation, `g(x)=x`,
which reduces the surrogate to a linear ridge):

```
b   = per-instance gmean-nmse with g(x)=x        (the linear baseline)
obj = per-instance gmean-nmse with your g
r   = min(1, 0.1 * b / max(obj, 1e-12))
```

Reproducing the linear baseline scores ≈ 0.1; a worse-than-linear `g` scores below 0.1; a good
nonlinearity scores higher. The irreducible observation-noise floor keeps even excellent designs
below 1.0. The reported score is the mean of `r` over all instances.

## Notes

- The candidate runs **OS-sandboxed in its own subprocess** and only ever receives the public
  instance. Targets, projections, the held-out setting and all scoring live in the evaluator
  process — introspection buys you nothing.
- There is **no single dominant activation**: saturating, rectifying, even/localized and periodic
  shapes each help some settings and hurt others, and the held-out setting rewards generalization
  over overfitting the visible mix.
