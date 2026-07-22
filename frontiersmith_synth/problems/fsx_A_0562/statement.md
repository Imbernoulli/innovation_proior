# Wind-Tunnel Drag Law — the units are the constraint

A wind-tunnel notebook records the **drag force** `F` on a bluff body against four
measured quantities. Every column carries a declared **physical unit** (with
`M`=mass, `L`=length, `T`=time):

| symbol | quantity            | unit                    | dimension  |
|--------|---------------------|-------------------------|------------|
| `rho`  | fluid density       | kg·m⁻³                  | M L⁻³      |
| `V`    | free-stream speed   | m·s⁻¹                   | L T⁻¹      |
| `D`    | body length scale   | m                       | L          |
| `mu`   | dynamic viscosity   | kg·m⁻¹·s⁻¹              | M L⁻¹ T⁻¹  |
| `F`    | **drag force**      | kg·m·s⁻² (N)            | M L T⁻²    |

The hidden law is a single smooth ground truth of the form
`F = (a dimensionally admissible expression in rho, V, D, mu)`, corrupted by
multiplicative sensor noise. Recover a closed-form expression for `F`.

**The declared units are not decoration — they are hard algebraic constraints.**
Any formula whose units do not reduce to `M L T⁻²` is physically impossible and
will be punished, because it is graded where extrapolation exposes it.

## Input (stdin)
- Line 1: two integers `n` and a case id.
- Next `n` lines: `rho V D mu F`, one wind-tunnel measurement each (floats).

The recorded campaign is a **single facility with a single working fluid**: the
speed `V` is swept widely and the body size `D` moderately, but `rho` and `mu`
barely move across the notebook.

## Output (stdout)
One line: a closed-form Python expression for `F` in the variables
`rho`, `V`, `D`, `mu`. Allowed: `+ - * / **`, unary `-`, numeric constants, and
the functions `sqrt log exp sig tanh absv`. Example (illustrative **form only —
NOT the hidden law**): `12.0 * V**2 * D / sqrt(rho)`. No other names are accepted.

## Scoring (deterministic, minimization)
Your expression is evaluated on a **held-out extrapolation grid**, regenerated
inside the grader, on which all four quantities lie **far outside their training
ranges** (a different fluid, a different scale). Let `p_i` be your prediction and
`t_i` the true (noisy) drag at held-out point `i`:

```
metric   = mean_i  min(1, |p_i - t_i| / (|p_i| + |t_i|))     # bounded rel. error
O        = metric * (1 + LAMBDA * nodes)                     # nodes = expr size
baseline = the same metric for the constant predictor geomean(train F)
Ratio    = min(1000, 100 * baseline / O) / 1000
```

Lower held-out error ⇒ higher `Ratio` (capped at `1.0`). A constant scores about
`0.1`. `LAMBDA` is a small parsimony weight, so an overgrown expression is
penalized. Non-finite or complex-valued predictions score `0`.

## Why the obvious fit is a trap
A generic symbolic-regression fit of a free power law
`F = k·rho^p1·V^p2·D^p3·mu^p4` matches the notebook, but because `rho` and `mu`
are nearly constant in training their exponents are **not identifiable** — the fit
pins them to noise. On the held-out grid `rho` and `mu` move by orders of
magnitude and those exponents send the prediction to nonsense.

Dimensional homogeneity removes the guesswork: the units alone force the force
scale to the monomial `rho·V²·D²` and leave exactly **one** dimensionless group,
the Reynolds number `Re = rho·V·D/mu`. Hence `F = rho·V²·D² · g(Re)` — a **4-D
fitting problem collapses to fitting one 1-D curve** `g(Re)`. Nail the prefactor
from the units, fit only `g`, and your formula survives extrapolation.

## Constraints
- Time limit 5 s, memory 512 MB; `n` up to a few hundred rows.
- Held-out noise leaves irreducible error, so even a correct law does not reach
  `Ratio = 1.0` — there is room above the reference solutions.
