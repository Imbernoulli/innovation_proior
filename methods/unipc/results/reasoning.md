I have a fast guided diffusion sampler that already works — the data-prediction multistep
exponential-integrator, DPM-Solver++(kM) — and it gets me to maybe fifteen or twenty network calls
with stable images under large guidance. But I want the extreme few-step regime, five to ten calls,
and there the wheels come off: every solver I have leaves a visible per-step truncation error because
at five steps the half-log-SNR interval `h` is large and the `O(h^k)` term is not small. The reflex is
"go to higher order," but I have already learned that just raising the order of a *predictor* solver
buys less and less — and under guidance, where the model's derivatives are amplified, can even hurt.
So before I reach for order, let me ask what *kind* of error I am leaving uncorrected, because the
answer might be a different lever than order.

Here is the structural observation. Every solver I use — DDIM, DPM-Solver, DPM-Solver++, DEIS — is a
*predictor*: at each step it forms an estimate of `x_t` from information it already has (past model
outputs, or fresh intermediate ones) and then *commits* to that estimate and marches on. The leading
truncation error of that step is baked in; nothing ever looks back and refines it. In classical ODE
numerics this is exactly the situation a *predictor-corrector* pair addresses: an Adams–Bashforth
predictor takes the step, then you evaluate the right-hand side *at the predicted point*, and an
Adams–Moulton corrector uses that new evaluation to refine the step, picking up an extra order. The
corrector is the cheapest order I can buy — it needs one new function evaluation at the predicted
point. In a generic ODE that extra evaluation is real cost. But look at what a *multistep diffusion*
loop actually does: at the start of step `i+1`, the predictor's base evaluation is the network applied
at the point that step `i` just predicted. That is *precisely* the evaluation a corrector for step `i`
would need. So in a multistep sampler the corrector is **free** — I reuse the next step's predictor
evaluation to correct the previous step, and the network-call count is unchanged. That is the opening.
A predictor-corrector that costs the same NFE as the predictor alone, but with the leading truncation
error of each step corrected.

Now I have to ask why correctors are not already standard in fast diffusion solvers, because if it were
this easy someone would do it. The reason is derivation cost: a corrector of order `p` is a *different*
formula from the predictor of order `p`, and from the corrector of order `p+1`, and each is a separate
hand-derivation of exponential-integrator coefficients. People derive a second-order solver, a
third-order solver, and stop, because each new order is bespoke algebra. So the real obstacle is not
the corrector idea — it is the lack of a *unified* form that produces predictor and corrector at
*arbitrary* order from one template. If I can write both as the same update, parameterized by order,
then "add a corrector" and "go one order higher" become the same one-line change, and the corrector
becomes trivially available at whatever order I am running. So the problem reduces to: find one
analytical update that is the predictor at order `p-1` and the corrector at order `p`, for any `p`.

Start from the exact data-prediction step between noise levels `s_0` (current) and `t` (next):
`x_t = (sigma_t/sigma_{s_0}) x_{s_0} + sigma_t integral_{lambda_{s_0}}^{lambda_t} e^{lambda}
x_theta(lambda) d lambda`. Everything high-order lives in approximating that integral. I expand
`x_theta(lambda)` in a Taylor series around the left endpoint `lambda_{s_0}`. The zeroth-order term
(hold `x_theta` constant at its current value `m_0 = x_theta(x_{s_0}, s_0)`) integrates to the
first-order base step; writing `h = lambda_t - lambda_{s_0}` and using `sigma_t e^{lambda_t} = alpha_t`,
the base term is `x_t^{(base)} = (sigma_t/sigma_{s_0}) x_{s_0} - alpha_t h_phi_1 m_0`, where
`h_phi_1 = e^{hh} - 1` with `hh = -h` in the data-prediction convention (I keep the implementation's
sign: `predict_x0` flips `h`). The higher-order terms involve the derivatives `x_theta^{(n)}` at
`lambda_{s_0}`, each weighted by an integral of `e^{lambda}(lambda-lambda_{s_0})^n`, i.e. by a `phi`
function evaluated at `hh`.

The derivatives I do not have in closed form; I estimate them from finite differences of past model
outputs. Say I keep the last few data predictions `m_0, m_1, m_2, ...` at past noise levels with
half-log-SNRs `lambda_{s_0}, lambda_{s_1}, ...`. Define the ratios `r_k = (lambda_{s_k} -
lambda_{s_0})/h` (so `r_0 = 0`, and the past points have `r_k != 0`) and the scaled differences
`D1_k = (m_k - m_0)/r_k`. A linear combination `sum_k rho_k D1_k` of these differences, with the right
coefficients `rho_k`, reproduces the Taylor correction to whatever order I have points for. The
question is which `rho_k`. Match the method's update to the exact integral's Taylor expansion term by
term: that is a *linear system*. Collect the powers of the ratios into a matrix `R` whose `i`-th row is
`r_k^{i-1}` (a Vandermonde-like structure in the ratios), and a right-hand side `b` whose `i`-th entry
is the `phi`-derived coefficient that the exact integral assigns to the `i`-th Taylor term, divided by
a chosen scalar `B(h)`. Then `R rho = b` gives exactly the `rho_k` that make the finite-difference
combination match the exact correction to the available order. That is the unification: one linear
solve produces the high-order coefficients at *any* order, no per-order hand derivation.

Let me pin the right-hand side. The exact integral's `phi` coefficients follow the recurrence
`phi_{k+1}(z) = (phi_k(z) - 1/k!)/z` starting from `phi_1`. In the code this appears as building
`h_phi_k` iteratively: start `h_phi_k = h_phi_1/hh - 1` (that is `phi_2`-ish), and for `i = 1..order`
append `b_i = h_phi_k * i! / B_h`, then advance `h_phi_k = h_phi_k/hh - 1/((i+1)!)` and grow the
factorial. The matrix rows are `R_i = r_k^{i-1}` for `i = 1..order`. So `R` and `b` are built together
in one loop, and the order is just how many rows/points I include.

Now the scalar `B(h)`. It multiplies the whole high-order correction, and it is genuinely free — any
nonzero choice gives a consistent method; it only changes the error *constant* and the conditioning.
Two natural choices: `B(h) = hh` ("bh1"), the simplest, and `B(h) = e^{hh} - 1 = h_phi_1` ("bh2"),
which matches the exponential weight of the integral more closely. At a tight budget where `h` is large
the `bh2` choice tracks the true integrand better, so I default to `bh2 = expm1(hh)`. I keep `B(h)` as
a named knob because it is exactly the kind of free constant that a robustness sweep later might want to
touch.

With `R`, `b`, `B(h)` in hand, predictor and corrector are the *same* formula evaluated with different
amounts of information. The **predictor** (UniP) has the past points but not yet an evaluation at the
new point `t`, so it solves the *reduced* system: for order `p` it uses the `(p-1)`-dimensional solve
`rho_p = solve(R[:-1,:-1], b[:-1])` (the special case order 2 just gives `rho_p = 0.5`), and the
update is
`x_t = x_t^{(base)} - alpha_t B_h (sum_k rho_{p,k} D1_k)`.
The **corrector** (UniC) is applied *after* I have taken the predictor step and then evaluated the
network at the predicted point to get `m_t = x_theta(x_t^{pred}, t)`. Now I have one *more* usable
evaluation, so the corrector solves the *full* `p`-dimensional system `rho_c = solve(R, b)` (order 1
gives `rho_c = 0.5`), and refines the *same* base step with the extra difference `D1_t = m_t - m_0`:
`x_t = x_t^{(base)} - alpha_t B_h (sum_k rho_{c,k} D1_k + rho_{c,last} D1_t)`,
where the base `x_t^{(base)}` and `m_0` here are taken from the *previous* step's quantities — the
corrector is refining the step that *produced* the point at which `m_t` was just evaluated. So the loop
is: at step `i`, first run the corrector on step `i-1` using the model output I just computed (free),
then run the predictor for step `i`. The corrector raises the realized order by one over the predictor
at the same NFE. This is precisely the "reuse the next step's evaluation" structure I spotted, made
exact.

A few realities I keep from the reference implementation. The `phi`/`expm1` factors are small-argument
exponentials, so `expm1` is used throughout to avoid cancellation. The order ramps up as history
accumulates — first step order 1, then 2, then up to the configured max — and there is a
`lower_order_final` option that drops the order on the last step(s) where there is no future evaluation
to correct with (and where the trajectory is nearly straight at low noise anyway). The time grid is the
EDM/Karras power schedule (`rho=7`) or uniform-`lambda`; either spends the budget where truncation
error is largest. For latent-space text-to-image there is no `[-1,1]` bound, so thresholding is off and
only the numerics matter. And the data prediction itself folds in classifier-free guidance — the
combined conditional/unconditional prediction — inside the model wrapper, so the solver sees a clean
`x_theta` oracle.

Let me write the per-step `bh` update — predictor and corrector sharing the `R`, `b`, `B_h` machinery —
so it drops into the multistep loop, each block tied to the step that motivated it.

```python
import torch


def expm1(x):
    return torch.expm1(x)                                  # accurate e^x - 1 for small x


class UniPCBH:
    """Unified predictor-corrector (bh variant) on the data-prediction diffusion ODE."""

    def __init__(self, ns, predict, solver_type="bh2", predict_x0=True):
        self.ns = ns                                       # schedule: alpha, sigma, lam
        self.predict = predict                             # (x, sigma) -> x_theta (guided, data pred)
        self.solver_type = solver_type                     # "bh1" (B=h) or "bh2" (B=e^h-1)
        self.predict_x0 = predict_x0

    def _R_b(self, rks, hh, B_h, order):
        # R rho = b : R_i = r^{i-1}, b_i = phi-derived coeff / B_h  (shared by P and C)
        h_phi_1 = expm1(hh)
        R, b = [], []
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        return torch.stack(R), torch.stack(b), h_phi_1

    def _bh(self, hh):
        return hh if self.solver_type == "bh1" else expm1(hh)   # B(h)

    def predictor(self, x, s0, t, m_list, lam_list, order):
        """UniP: extrapolate the history to advance from s0 to t (no new evaluation yet)."""
        ns = self.ns
        lam_s0, lam_t = ns.lam(s0), ns.lam(t)
        h = lam_t - lam_s0
        hh = -h if self.predict_x0 else h
        B_h = self._bh(hh)
        m0 = m_list[-1]

        rks, D1s = [], []
        for i in range(1, order):                          # past points -> ratios + differences
            rk = (lam_list[-(i + 1)] - lam_s0) / h
            rks.append(rk)
            D1s.append((m_list[-(i + 1)] - m0) / rk)
        rks.append(torch.ones((), device=x.device))
        rks = torch.stack(rks)

        R, b, h_phi_1 = self._R_b(rks, hh, B_h, order)
        x_base = (ns.sigma(t) / ns.sigma(s0)) * x - ns.alpha(t) * h_phi_1 * m0
        if D1s:
            D1s = torch.stack(D1s, dim=1)
            rhos_p = (torch.tensor([0.5], dtype=x.dtype, device=x.device) if order == 2
                      else torch.linalg.solve(R[:-1, :-1], b[:-1]))
            pred_res = torch.einsum("k,bk...->b...", rhos_p, D1s)
        else:
            pred_res = 0
        return x_base - ns.alpha(t) * B_h * pred_res

    def corrector(self, x_prev, s0, t, m_list, lam_list, model_t, order):
        """UniC: refine the previous step (s0 -> t) using model_t at the predicted point (free)."""
        ns = self.ns
        lam_s0, lam_t = ns.lam(s0), ns.lam(t)
        h = lam_t - lam_s0
        hh = -h if self.predict_x0 else h
        B_h = self._bh(hh)
        m0 = m_list[-1]

        rks, D1s = [], []
        for i in range(1, order):
            rk = (lam_list[-(i + 1)] - lam_s0) / h
            rks.append(rk)
            D1s.append((m_list[-(i + 1)] - m0) / rk)
        rks.append(torch.ones((), device=x_prev.device))
        rks = torch.stack(rks)

        R, b, h_phi_1 = self._R_b(rks, hh, B_h, order)
        x_base = (ns.sigma(t) / ns.sigma(s0)) * x_prev - ns.alpha(t) * h_phi_1 * m0
        rhos_c = (torch.tensor([0.5], dtype=x_prev.dtype, device=x_prev.device) if order == 1
                  else torch.linalg.solve(R, b))
        D1_t = model_t - m0
        if D1s:
            D1s = torch.stack(D1s, dim=1)
            corr_res = torch.einsum("k,bk...->b...", rhos_c[:-1], D1s)
        else:
            corr_res = 0
        return x_base - ns.alpha(t) * B_h * (corr_res + rhos_c[-1] * D1_t)

    @torch.no_grad()
    def sample(self, x, sigmas, max_order=3, lower_order_final=True):
        """One model evaluation per step; the corrector reuses it -> no extra NFE."""
        ns = self.ns
        N = len(sigmas) - 1
        m_list, lam_list, x_prev, s_prev = [], [], None, None
        for i in range(N):
            sigma = sigmas[i]
            model_t = self.predict(x, sigma)               # the ONLY network call this step
            if x_prev is not None:                         # correct the previous predictor step
                order_c = min(max_order, len(m_list))      # past points available for the corrector
                x = self.corrector(x_prev, s_prev, sigma, m_list, lam_list, model_t, order_c)
            m_list.append(model_t)
            lam_list.append(ns.lam(sigma))
            if i == N - 1:
                x = m_list[-1]                             # final step: return clean prediction
                break
            order_p = min(max_order, len(m_list))
            if lower_order_final:
                order_p = min(order_p, N - i)
            x_prev, s_prev = x, sigma
            x = self.predictor(x, sigma, sigmas[i + 1], m_list, lam_list, order_p)
        return x
```

The causal chain end to end: I want guided sampling in the extreme few-step regime, where every
*predictor-only* solver (DDIM, DPM-Solver++, DEIS) leaves an uncorrected leading truncation error
because order alone gives diminishing returns and is amplified by guidance. The cheapest extra order is
a *corrector* — evaluate at the predicted point and refine — and in a multistep diffusion loop that
evaluation is *already taken* at the start of the next step, so the corrector is free in NFE. Correctors
were absent only because each order needs bespoke algebra; the fix is to write predictor and corrector
as one update whose high-order coefficients come from solving a small linear system `R rho = b` in the
`lambda`-ratios, with the predictor solving the reduced `(p-1)`-system and the corrector the full
`p`-system using the extra evaluation. A free scalar `B(h)` (default `bh2 = e^{hh}-1`) sets the error
constant; `expm1` keeps the small-`h` factors honest; the order ramps with history and drops on the
final step; and the whole thing runs at one network call per step — the same NFE as DPM-Solver++(kM),
but with each step's leading error corrected, which is what wins in the 5–10 step regime.
