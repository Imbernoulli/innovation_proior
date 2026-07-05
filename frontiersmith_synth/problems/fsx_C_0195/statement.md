# Interstellar Relay: Beam-Alignment Saddle Optimizer

## Story

A chain of deep-space relay stations must jointly steer their phased-array beams.
The transmit side (variables `x`) tries to **minimise** a coupling cost while the
adversarial channel / interference side (variables `y`) tries to **maximise** it.
The operating point is the **saddle point** of a convex-concave objective

```
f(x, y) = 1/2 xᵀ A x  +  xᵀ B y  −  1/2 yᵀ C y  +  bᵀ x  +  cᵀ y
```

with `A, C` symmetric positive semidefinite (convex in `x`, concave in `y`) and `B`
the cross-coupling ("rotation") between the two sides. The natural first-order
quantity is the **monotone operator**

```
F(z) = ( ∇ₓ f , −∇_y f ) = M z + q ,     z = (x, y),
M = [[ A ,  B ],
     [−Bᵀ,  C ]]  =  S + K      (S = blockdiag(A,C) symmetric PSD, K skew-symmetric).
```

At the equilibrium `z*`, `F(z*) = 0`. So after a **fixed iteration budget** of `T`
steps, the quality of a run is its **residual gradient norm** `‖F(z_T)‖` — smaller is
better. The budget is an op-count, **not** wall-time: scoring is fully deterministic.

## Your task — design the update schedule (you do not run the optimiser)

You never see the matrices `M, q` and you never run the optimiser yourself. Instead
you **design the coefficient schedule** of a **frozen** generalised first-order
saddle-point method, and submit that schedule as data. The evaluator then executes
your schedule **itself** on several **hidden** relay problems drawn from the same
class and recomputes `‖F(z_T)‖`.

The frozen update template, for `t = 0 … T−1`, is:

```
m = 0
for t in range(T):
    g      = F(z)                       # gradient at the current iterate
    z_look = z − alpha_t * eta_t * g    # extrapolation / look-ahead
    g_look = F(z_look)                   # gradient at the look-ahead point
    m      = beta_t * m + g_look          # heavy-ball momentum buffer
    z      = z − eta_t * m
# objective = ‖ F(z_T) ‖
```

Special cases: `alpha=1, beta=0` is classic **extragradient (EG)**; `alpha=0, beta=0`
is plain **gradient descent-ascent (GDA)**, which *diverges* on rotation-dominated
relays; intermediate settings give optimistic / momentum / accelerated variants.

Because the matrices are **hidden**, you must choose a schedule that **generalises**
from the class descriptors — this is convergence-rate analysis, not offline solving.

## Isolation (how your program is run)

Your program runs as an **isolated subprocess**. It reads exactly one JSON object (the
*public* view of one relay class) from **stdin** and writes exactly one JSON object
(your schedule) to **stdout**. You never see the hidden matrices, the equilibria, or
the evaluator's memory.

```python
import sys, json
inst = json.load(sys.stdin)          # public class descriptors ONLY
L, mu, typ, T = inst["L"], inst["mu"], inst["type"], inst["T"]
# ... design a schedule from the descriptors ...
print(json.dumps({"eta": 0.5 / L, "alpha": 1.0, "beta": 0.0}))
```

## Public instance (stdin)

```json
{
  "type":       "strong" | "mixed" | "bilinear" | "illcond",  // relay class
  "n":          int,      // block dimension (dim x = dim y = n)
  "dim":        int,      // total dimension 2n
  "T":          int,      // iteration budget (number of update steps)
  "L":          float,    // Lipschitz upper bound on ‖M‖ (a step ~1/L is stable)
  "mu":         float,    // strong-monotonicity lower bound (min eig of S); 0 if none
  "num_hidden": int,      // how many hidden problems your schedule is scored on
  "seed":       int       // advisory
}
```

## Answer (stdout)

```json
{"eta": <float | [T floats]>,     // step sizes (REQUIRED)
 "alpha": <float | [T floats]>,    // look-ahead fraction (optional, default 1.0)
 "beta":  <float | [T floats]>}    // momentum coefficient (optional, default 0.0)
```

A scalar is broadcast to all `T` steps; a list must have length exactly `T`. Every
value must be finite. Any of the following makes an instance score **0**: a missing
`eta`, a non-list/non-scalar field, a wrong-length list, a non-finite value, a crash,
or a timeout. A schedule whose iterates blow up simply earns a large residual (and so
a score near 0) on that instance.

## Scoring

For each relay class the evaluator runs your schedule on `num_hidden` hidden problems
through the frozen template and takes the **mean final residual gradient norm** `obj`.
It compares against its own **reference method** — extragradient with the constant step
`1/(2L)` (`alpha=1, beta=0`), which is stable on every class — whose mean final norm is
`B`:

```
r = min( 1.0,  0.1 * B / max(obj, 1e-12) )
```

so reproducing the reference maps to `≈ 0.1` and being **10× better** maps to `1.0`.
The reported score is the arithmetic mean of the per-class `r`:

```
Ratio:  <mean of per-class r, in [0,1]>
Vector: [r_1, r_2, ..., r_8]
```

## Objective

**Minimise the residual norm ⇒ maximise `Ratio`.** There is no easy optimum. A single
constant step cannot win the whole family: aggressive extragradient rules the
well-conditioned relays but wastes the budget on ill-conditioned ones, heavy-ball
momentum accelerates the symmetric ill-conditioned classes but **diverges** on the
rotation-dominated `bilinear`/`mixed` classes, and plain GDA diverges outright. A
schedule that reasons from `(L, mu, type)` — the essence of convergence-rate analysis —
is required to score well across all eight relay classes.
