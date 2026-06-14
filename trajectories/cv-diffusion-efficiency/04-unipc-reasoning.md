The third-order multistep SDE paid off across the board and it landed where I called it. DPM++(3M) SDE
pulled every variant below 2S: SD v1.5 from 29.01 to 27.98, SD v2.0 from 23.89 to 23.45, and SDXL —
again the one I said had the most to gain — from 42.83 to 39.66, the largest absolute drop and the
first time any sampler put SDXL under 40. So both of my diagnoses from the 2S rung were right: keeping
all twenty steps fine (multistep, one call each) beat the halved ten-step singlestep grid, and the
Langevin re-noising at $\eta=1.2$ bought back the global structure a deterministic march was drifting
away from. Three rungs in, the ladder has done what a ladder should — each step's *measured* failure
named the next move, and each move improved every variant. But now I look hard at *what kind* of solver
3M still is, and I see a single structural gap that all three rungs share and none has touched, and it
is the gap that is keeping SDXL at 39.66 instead of lower.

Every sampler I have run — DDIM, 2S, 3M — is a **predictor**. At each step it forms an estimate of the
next latent from information it already has (the current prediction, plus, for 2S and 3M, an
intermediate or past one) and then *commits*. The leading truncation error of that step is baked in;
nothing ever looks back at a step and refines it. 3M's third-order curvature term reduces that error,
and the SDE noise washes out some of what is left — but notice what the noise is doing: it is a
*statistical* correction for accumulated drift, not a correction of the *individual step's*
deterministic error. In fact the $\eta=1.2$ stochasticity is partly compensating for the very thing a
better-corrected step would not produce. So the question is whether I can correct each step's leading
error *directly*, within the loop, rather than letting it accumulate and then dousing it with noise.

The classical answer in ODE numerics is a predictor-corrector pair. You take the predictor step, then
you *evaluate the right-hand side at the predicted point*, and a corrector uses that fresh evaluation to
refine the step — gaining an order. The catch in a generic ODE is that the corrector's evaluation is
extra cost, and at NFE = 20 I cannot afford a single extra model call: 3M already spends all twenty on
one call per step. So a corrector looks disqualified on budget. But look closely at what a *multistep*
loop already does. At the start of step $i+1$, the predictor's base evaluation is the network applied at
the latent that step $i$ just predicted. That is *exactly* the evaluation a corrector for step $i$ would
need. The corrector's "extra" call is *already being taken* by the next step's predictor. So in a
multistep diffusion sampler the corrector is **free** — I reuse step $i+1$'s evaluation to correct step
$i$, and the total call count stays at twenty. This is the lever none of the three rungs pulled: each
spent its evaluations purely on prediction; I can spend the same evaluations on prediction *and*
correction, raising the realized order by one at no budget cost. That is the move past 3M.

I have to be honest about why correctors are not already in the fast solvers I have been climbing, since
if it were this easy 3M would already be a predictor-corrector. The reason is derivation cost: a
corrector of order $p$ is a *different* formula from the predictor of order $p$, and each order is a
separate hand-derivation of exponential-integrator coefficients. People derive a second- and
third-order predictor and stop. So the real obstacle is the lack of a *unified* form that yields both
predictor and corrector at *arbitrary* order from one template. If I can write both as the same update,
parameterized by order, then "add a corrector" is the same one-line change as "go one order higher,"
and the corrector becomes available at whatever order the history supports.

Here is the unification, and I derive it in the data-prediction face because that is the one that stays
stable under the large CFG++ guidance the substrate uses, and because — like 3M — it lets me work in the
clean-image estimate the harness's Tweedie step already produces. The exact data-prediction step from
the current level $s_0$ to the next $t$ approximates an integral of $x_\theta(\lambda)$ in half-log-SNR.
Hold $x_\theta$ at its current value $m_0$ and I get the first-order base step; the higher-order terms
are the Taylor derivatives of $x_\theta$, which I estimate from finite differences of past data
predictions. Keep the recent predictions $m_0, m_1, m_2,\dots$ at $\lambda$-spacings, define the ratios
$r_k=(\lambda_{s_k}-\lambda_{s_0})/h$ with $h=\lambda_t-\lambda_{s_0}$ (and $hh=-h$ in this convention,
the sign the implementation uses for $\mathrm{predict\_x0}$), and the scaled differences
$D1_k=(m_k-m_0)/r_k$. A linear combination $\sum_k\rho_k D1_k$ reproduces the Taylor correction — and
the coefficients $\rho_k$ are exactly the solution of a small linear system $R\rho=b$, where $R$ has rows
$r_k^{\,i-1}$ (a Vandermonde structure in the ratios) and $b$ is the $\phi$-derived sequence
$h\phi_k\cdot i!/B(h)$, advanced by $h\phi_k\leftarrow h\phi_k/hh - 1/(i+1)!$. Matching the system to the
exact integral's Taylor expansion to order $p$ makes the method order-$p$ for *any* $p$. That single
linear solve is the whole unification — no per-order algebra.

Predictor and corrector are then the *same* update with different amounts of information. The predictor
(UniP) does not yet have an evaluation at the new point $t$, so it solves the *reduced* system: for
order $p$ it uses $\rho_p=\mathrm{solve}(R[:\!-1,:\!-1],b[:\!-1])$ (order 2 collapses to $\rho_p=0.5$),
and the step is $x_{\text{base}}-\alpha_t B(h)\,(\sum_k\rho_{p,k}D1_k)$, where
$x_{\text{base}}=(\sigma_t/\sigma_{s_0})x-\alpha_t\,h\phi_1\,m_0$. The corrector (UniC) runs *after* the
predictor has stepped and the next iteration has evaluated the network at the predicted point, giving
$m_t=x_\theta$ there. Now there is one *more* usable evaluation, so the corrector solves the *full*
$p$-dimensional system $\rho_c=\mathrm{solve}(R,b)$ (order 1 gives $\rho_c=0.5$) and refines the same
base step with the extra difference $D1_t=m_t-m_0$:
$x_{\text{base}}-\alpha_t B(h)\,(\sum_k\rho_{c,k}D1_k+\rho_{c,\text{last}}D1_t)$. The free scalar $B(h)$
sets the error constant; I take $B(h)=e^{hh}-1$ (the "bh2" choice), which tracks the exponential weight
better at the large $h$ a twenty-step budget forces, over the simpler $B(h)=hh$ ("bh1"). The loop is
therefore: at each step make the one model call, first run the corrector on the *previous* step using
that call, then run the predictor for the current step — one evaluation doing double duty.

Now make it concrete in this task's edit surface, because the substrate is the same `sample` body I
have been filling and I want UniPC expressed in its vocabulary, not a generic harness. The loop walks
`self.scheduler.timesteps` exactly as DDIM did (twenty steps, one `predict_noise` per step, so NFE
stays at twenty). At each timestep I form the guided prediction
$\tilde\epsilon=\epsilon_{uc}+s(\epsilon_c-\epsilon_{uc})$ and the data prediction by Tweedie,
$m_t=z_0=(z_t-\sqrt{1-\bar\alpha_t}\,\tilde\epsilon)/\sqrt{\bar\alpha_t}$ — that single evaluation is
both the corrector input for the step I took last iteration and the predictor base for this one. I keep
$\bar\alpha$ via `self.alpha(t)` and compute $\lambda=\tfrac12(\log\bar\alpha-\log(1-\bar\alpha))$ from
it, so I never leave the substrate's $(\bar\alpha,z_0)$ language for a separate $\sigma$ schedule. The
$R\rho=b$ machinery, the reduced-vs-full solves, the `expm1` for the $h\phi$ factors, and the
order ramp (1 → 2 → 3 as history fills, dropping on the final step where there is no future evaluation
to correct with) all go inside the body. The final step returns the clean data prediction $z_0$ itself,
exactly as 3M's $\sigma=0$ step does. Two things I am deliberately *not* importing: there is no
thresholding (this is latent space, no $[-1,1]$ bound), and there is no SDE noise term — UniPC is a
deterministic predictor-corrector, and the whole point is that *correcting* each step's leading error is
a more direct fix than 3M's stochastic compensation for uncorrected error. I keep the CFG++ convention
in that the data prediction uses the guided $\tilde\epsilon$ while the unconditional stream is available
for the base; the renoising that 2S/DDIM did with $\epsilon_{uc}$ is subsumed here into the
data-prediction step.

The bar this has to clear is 3M's measured numbers, and I will state the falsifiable expectation against
them rather than invent a result. The claim is that correcting each step's leading truncation error
within the loop — at the same twenty calls, with no extra evaluation — beats predicting-and-renoising.
So I expect UniPC to land *below* 27.98 on SD v1.5, below 23.45 on SD v2.0, and below 39.66 on SDXL,
with the largest absolute improvement again on SDXL, where uncorrected per-step error has been the
dominant cost at every rung. The specific mechanism I am betting on is that the corrector buys a genuine
order over 3M's predictor at equal NFE, and that a deterministic corrected step does not need the
$\eta=1.2$ noise 3M spent to mask its drift. The way this is falsified is clean: if UniPC fails to beat
39.66 on SDXL, then the corrector's extra order is not worth more than the stochastic correction it
replaces at this budget — in which case the right reading is that at twenty steps 3M's noise was doing
real work that a deterministic corrector cannot replicate, and the next move would be a *stochastic*
predictor-corrector rather than abandoning the corrector. But the expectation I commit to is that the
unified predictor-corrector, correcting where every previous rung only predicted, is the strongest
sampler of the four, and the place it shows most is SDXL.
