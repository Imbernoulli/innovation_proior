# Rooftop Garden Equilibrium: Blind Saddle-Point Schedule

## Story

A city operates a network of interconnected **rooftop gardens**. A horticulture
controller sets continuous action levels `x` (irrigation, shading, nutrient dosing per
plot) to keep the gardens healthy, while an adversarial environment `y` (heat waves,
pests, wind stress) constantly pushes them out of balance. Their conflict is a
**convex-concave saddle game**

```
min_x  max_y   f(x, y) = 1/2 x^T P x  +  x^T A y  -  1/2 y^T Q y  +  b^T x  -  c^T y
```

with `P`, `Q` symmetric positive semidefinite (the controller's cost is convex in `x`,
the environment's payoff concave in `y`) and `A` the cross-coupling between the two.

The network is **at equilibrium** (a saddle point) exactly when the joint
*disequilibrium* vector vanishes:

```
F(z) = [ grad_x f ;  -grad_y f ] = M z + q ,          z = (x, y) ∈ R^d ,  d = dx + dy
M = [[ P ,  A ],
     [ -A^T, Q ]] ,        q = [ b ; c ]
```

`||F(z)||` is the residual imbalance of the whole rooftop network. The symmetric part of
`M` is positive semidefinite (monotone operator); the antisymmetric part `[[0,A],[-A^T,0]]`
is the rotational stress the coupling injects.

## Your task (op-budgeted, blind schedule design)

Starting from a fixed point `z0`, the controller may apply exactly **`T` update steps**.
Each step spends its budget on **one** disequilibrium probe `g_k = F(z_k)` (one operator
application) and then moves the iterate:

```
z_{k+1} = z_k  -  a_k * g_k  +  m_k * (z_k - z_{k-1})  -  o_k * (g_k - g_{k-1})
          (with z_{-1} := z0 ,  g_{-1} := g_0)
```

You must **commit the entire scalar schedule up front**: three length-`T` vectors

* `a_k` — step size,
* `m_k` — heavy-ball momentum,
* `o_k` — optimistic / negative-probe correction.

Special cases: `m=o=0` is plain gradient descent-ascent; `o=a, m=0` is optimistic GDA;
`m>0` adds heavy-ball acceleration. Because the coefficients are fixed before any probe
is seen, the final residual is exactly a degree-≤`T` matrix polynomial in `M` applied to
`g_0` — so this is a pure **convergence-schedule-design** problem.

**Objective (minimize):** drive the final imbalance `||F(z_T)||` as low as possible
within the `T`-step budget. Lower is better.

## Public instance (stdin, one JSON object)

```json
{
  "name": "rooftop101",
  "dx": 8, "dy": 8, "d": 16, "T": 6,
  "M":  [[...], ...],   // d x d saddle operator (authoritative)
  "q":  [...],          // length d
  "z0": [...],          // length d start iterate
  "P": [[...]], "Q": [[...]], "A": [[...]], "b": [...], "c": [...]  // theme blocks
}
```

`M` and `q` are authoritative; the `P,Q,A,b,c` blocks are provided for interpretability
(`M`, `q` are assembled from them as above). The budget always satisfies `T < d`.

## Answer (stdout, one JSON object)

```json
{"a": [a_0, ..., a_{T-1}], "m": [m_0, ...], "o": [o_0, ...]}
```

* `"a"` is **required**: a list of exactly `T` finite numbers.
* `"m"` and `"o"` are optional; if present each must be a list of exactly `T` finite
  numbers (defaulting to all-zeros when omitted).

Wrong length, non-finite coefficients, a crash, a timeout, non-JSON, or a schedule that
makes the iterate blow up to `inf`/`nan` → that instance scores **0.0**.

## Scoring (deterministic, no wall-time)

For each instance the evaluator computes, in its own process:

* `base_res = ||F(z0)||` — the "do nothing" reference,
* `cand_res = ||F(z_T)||` from your committed schedule,
* `ref_res` — the **optimal degree-`T` residual**, i.e.
  `min over degree-≤T polynomials p with p(0)=1 of ||p(M) g_0||`, obtained by truncated
  GMRES(`T`). This is the information-theoretic floor for **any** committed `T`-step
  schedule, so it can be approached but not beaten.

The normalized per-instance score (affine in log-residual, since residuals decay
geometrically) is

```
r = clamp( 0.1 + 0.9 * (ln base_res - ln cand_res) / (ln base_res - ln ref_res), 0, 1 )
```

* doing nothing → `0.1`;
* matching the (unreachable) GMRES optimum → `1.0`;
* an unstable/divergent schedule that ends up worse than doing nothing → below `0.1`,
  floored at `0`.

Because the budget is strictly smaller than the dimension (`T < d`), a degree-`T`
polynomial cannot annihilate all `d` modes of `M`, so even the optimal schedule leaves a
substantial residual — leaving genuine headroom above any hand-tuned schedule.

The final reported **Ratio** is the mean of `r` over all instances; **Vector** lists the
per-instance `r`.

## Isolation

Your program is run untrusted in a fresh sandboxed subprocess and only ever receives the
**public instance** above. `base_res`, `ref_res` and the scoring are computed by the
evaluator process, which your program cannot reach.
