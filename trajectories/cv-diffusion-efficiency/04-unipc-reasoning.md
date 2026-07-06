The third-order multistep SDE paid off across the board and it landed where I called it. DPM++(3M) SDE
pulled every variant below 2S: SD v1.5 from 29.01 to 27.98, SD v2.0 from 23.89 to 23.45, and SDXL —
again the one I said had the most to gain — from 42.83 to 39.66, the first time any sampler put SDXL
under 40. So both of my diagnoses from the 2S rung were right: keeping all twenty steps fine (multistep,
one call each) beat the halved ten-step singlestep grid, and the Langevin re-noising at $\eta=1.2$ bought
back the global structure a deterministic march was drifting away from. But now I read the *sizes* of the
drops, and they carry a warning. The DDIM$\to$2S step bought $-5.22/-4.52/-8.69$; the 2S$\to$3M step bought
only $-1.03/-0.44/-3.18$ — that is $20\%$, $10\%$, and $37\%$ of the previous gain on the three variants.
The improvements are shrinking fast even as I climb from second to third order. That is the classic
diminishing-returns signature of stacking *predictor* order: each extra order corrects a higher derivative,
but at twenty steps the low derivatives already dominate the error, so the marginal order buys less and
less, and under CFG++ guidance the amplified high derivatives make it even less safe to keep climbing. SDXL
is still worst — $39.66$ is $1.69\times$ SD v2.0's $23.45$, a $16.2$-point gap that has narrowed at every
rung but never collapsed. Three rungs in, the ladder has done what a ladder should — each step's *measured*
failure named the next move — but the shrinking drops tell me that "add one more order" is no longer the
lever. I need a different one.

So before I reach for order again, let me ask what *kind* of error I am leaving uncorrected, because the
answer might be a different axis entirely. Here is the structural observation. Every sampler I have run —
DDIM, 2S, 3M — is a **predictor**. At each step it forms an estimate of the next latent from information it
already has (the current prediction, plus, for 2S and 3M, an intermediate or past one) and then *commits*.
The leading truncation error of that step is baked in; nothing ever looks back at a step and refines it.
3M's third-order curvature term reduces that error, and the SDE noise washes out some of what is left — but
notice what the noise is doing: it is a *statistical* correction for accumulated drift, not a correction of
the *individual step's* deterministic error. In fact the $\eta=1.2$ stochasticity is partly compensating
for the very thing a better-corrected step would not produce; I spent a knob masking uncorrected error
rather than removing it. So the question is whether I can correct each step's leading error *directly*,
within the loop, rather than letting it accumulate and then dousing it with noise.

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
correction, raising the realized order by one at no budget cost. And crucially it is a *different* order
than the diminishing predictor order — it is the order the predictor left on the table, recovered from a
call I am already paying for.

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
linear solve is the whole unification — no per-order algebra. And it is genuinely free in the sense that
matters here: the system is at most $3\times3$ (order capped at three), so each `torch.linalg.solve` is a
handful of floating-point operations on tiny matrices, utterly negligible next to a full UNet forward pass.
The only expensive thing in the loop is `predict_noise`, and the corrector adds none of those — so "the
corrector is free" is free in NFE *and* in wall-clock, not just in call count.

That recurrence for $b$ is where a dropped factorial would silently wreck the order, so I check it rather
than trust it. The claim is that at step $i$ the built quantity $h\phi_k$ equals $\phi_{i+1}(hh)\cdot hh$,
the exact $\phi$-ladder coefficient. Run the recurrence at a representative $hh=-0.3$: it produces
$-0.13606074,\,-0.04646421,\,-0.01178595,\,-0.00238016$ for $i=1,2,3,4$, and computing
$\phi_2(hh)\cdot hh,\,\phi_3(hh)\cdot hh,\,\phi_4(hh)\cdot hh,\,\phi_5(hh)\cdot hh$ directly from the
closed-form $\phi$ definitions gives exactly the same four numbers. So the little iterative build *is* the
$\phi$ ladder and nothing is off by a factorial; the right-hand side $b$ is correct at every order I will
use.

Predictor and corrector are then the *same* update with different amounts of information. The predictor
(UniP) does not yet have an evaluation at the new point $t$, so it solves the *reduced* system: for
order $p$ it uses $\rho_p=\mathrm{solve}(R[:\!-1,:\!-1],b[:\!-1])$ (order 2 collapses to $\rho_p=0.5$),
and the step is $x_{\text{base}}-\alpha_t B(h)\,(\sum_k\rho_{p,k}D1_k)$, where
$x_{\text{base}}=(\sigma_t/\sigma_{s_0})x-\alpha_t\,h\phi_1\,m_0$. That base term is worth recognizing:
$h\phi_1=e^{hh}-1=e^{-h}-1$, so $x_{\text{base}}=(\sigma_t/\sigma_{s_0})x-\alpha_t(e^{-h}-1)m_0
=(\sigma_t/\sigma_{s_0})x+\alpha_t(1-e^{-h})m_0$ — exactly the first-order data-prediction step, which is
DDIM. So UniPC, like every rung before it, is DDIM plus a correction; here the correction is
$-\alpha_t B(h)\sum\rho D1$, and the corrector's contribution is one more term in that same sum. The whole
ladder has been the same base step wearing progressively better corrections. The corrector (UniC) runs *after* the
predictor has stepped and the next iteration has evaluated the network at the predicted point, giving
$m_t=x_\theta$ there. Now there is one *more* usable evaluation, so the corrector solves the *full*
$p$-dimensional system $\rho_c=\mathrm{solve}(R,b)$ (order 1 gives $\rho_c=0.5$) and refines the same
base step with the extra difference $D1_t=m_t-m_0$:
$x_{\text{base}}-\alpha_t B(h)\,(\sum_k\rho_{c,k}D1_k+\rho_{c,\text{last}}D1_t)$. The dimension bookkeeping
is the cleanest way to see that predictor and corrector are one formula: at a step where two past points
are available (order 3), the predictor solves the *reduced* $2\times2$ system on the two past ratios
$R[:\!-1,:\!-1]$ — it has no value at $t$ yet, so the endpoint slot (the $r=1$ row/column) is dropped —
while the corrector, once $m_t$ exists, solves the *full* $3\times3$ system including that endpoint slot.
The extra row-and-column that turns the $2\times2$ into the $3\times3$ *is* the free correction: it is the
$D1_t$ difference built from the evaluation the next step was going to make anyway. So "add a corrector" is
literally "include one more row," the same operation as "go one order higher," which is the unification I
was after. The free scalar $B(h)$
sets the error constant; any nonzero choice solves the same matching conditions (it rescales $b$ and the
solve rescales $\rho$ inversely, so the update is invariant), but it changes the error constant and the
conditioning. Two natural choices: $B(h)=hh$ ("bh1"), the simplest, and $B(h)=e^{hh}-1$ ("bh2"), which
matches the integral's exponential weight. The difference is negligible when $h$ is small but not when it
is large, and a twenty-step budget forces large $h$: at $hh=-0.1$ the two agree to $5\%$ ($-0.1$ vs
$-0.0952$), but at $hh=-1.5$ they differ by a factor of two ($-1.5$ vs $-0.777$) and at $hh=-2.9$ — the
scale of the coarsest steps on the SD noise grid — by a factor of three ($-2.9$ vs $-0.945$). Since $B(h)$
is exactly the weight the exponential integrand carries, bh2 tracks it where bh1 overshoots, so I default
to bh2, keeping it a named knob a later robustness sweep could touch.

This is the moment the whole rung stands or falls on, so I test the "+1 order at no extra NFE" claim on an
integrand whose exact integral I know, rather than trust the construction. Work in a face where the target
is $\sigma_t\int_{\lambda_{s_0}}^{\lambda_t}e^\lambda x_\theta\,d\lambda$ with a VP schedule setting
$\alpha_t,\sigma_t$ from $\lambda_t$; take a single step $\lambda_{s_0}=0\to\lambda_t=-0.4$ with one past
point at $\lambda=0.5$, bh2. On a *linear* $x_\theta$, the order-2 corrector (full $2\times2$ solve,
reusing the endpoint evaluation) reproduces the integral to $4.6\times10^{-14}$ — machine zero, as a method
built to match the Taylor terms must be exact on an integrand with no terms beyond what it matches. Now push
the integrand to *quadratic* and watch predictor and corrector diverge, because that is where the claim
either holds or does not. Same step, same NFE: the order-2 *predictor* (reduced $1\times1$ solve, past
point only, no endpoint evaluation) lands with error $2.5\times10^{-2}$ — genuinely second order, wrong on
quadratics — while the order-2 *corrector*, reusing the endpoint $m_t$ the next step supplies for free,
returns the quadratic integral to $2.6\times10^{-14}$. And bumping the corrector to order 3 (two past
points) reproduces the quadratic at the same machine-zero. So at equal cost the corrector is exact one
polynomial degree higher than the predictor: the "+1 realized order, +0 NFE" claim is not a slogan, it is
what the numbers do. That is precisely the order the diminishing predictor climb was leaving on the table.

And this is where the twenty-step budget makes the corrector *specifically* the right lever rather than a
generic nicety, because the predictor's uncorrected error is exactly what blows up at large $h$. Track the
same order-2 predictor and corrector on the quadratic integrand as I grow the step: at $|h|=0.2$ the
predictor error is $5.7\times10^{-3}$, at $|h|=0.4$ it is $2.5\times10^{-2}$, at $|h|=0.8$ it is
$1.07\times10^{-1}$, at $|h|=1.6$ it is $3.7\times10^{-1}$ — growing like $h^3$, the predictor's leading
uncorrected term. The corrector, over the same range, stays at $\sim10^{-15}$ to $10^{-12}$ — flat, machine
zero, because it is exact on quadratics regardless of step size. So the gap the corrector closes *widens*
as the steps coarsen, and a twenty-step budget is exactly the coarse-$h$ regime (rung one already tabulated
$|\Delta\lambda|$ up to $\approx2.9$ on the SD grid). At those step sizes the predictor is leaving a large
error uncorrected and the corrector removes it for free — which is why "correct the step" beats "add one
more predictor order" precisely here, and would matter far less at a hundred fine steps where the predictor
error is already small.

One thing surfaced in that test that I want to record honestly, because it contradicts a tidy story I
nearly told. I expected the order-2 predictor (the $1\times1$ reduced solve) to come out to a clean
$\rho=0.5$, the trapezoid coefficient, and the code indeed hardcodes $0.5$ for these smallest cases. But
computing $\rho=\phi_2(hh)\cdot hh/(e^{hh}-1)$ directly, it is *not* $0.5$ for finite $hh$: it is $0.50083$
at $hh=-0.01$, $0.50833$ at $hh=-0.1$, and $0.54149$ at $hh=-0.5$, tending to $0.5$ only as $hh\to0$. So the
hardcoded $0.5$ is the leading-order ($hh\to0$) value, a deliberate small simplification at the cheap end
where the difference is $O(hh)$ and swamped by other error — not the exact solve. This shows up in the test:
because the order-2 predictor uses the hardcoded $0.5$ rather than the exact $1\times1$ solve, even on a
*linear* integrand it carries a small residual, $3.3\times10^{-3}$, where the full-solve corrector was
machine-zero. I keep the hardcoding to match the anchor implementation, but I now know it is an
approximation, not an identity, and I write it as the explicit special case rather than pretend the solve
returns it. The loop is therefore: at each step make the one model call, first run the corrector on the
*previous* step using that call, then run the predictor for the current step — one evaluation doing double
duty, with the order arithmetic checked rather than assumed.

Now make it concrete in this task's edit surface, because the substrate is the same `sample` body I have
been filling and I want UniPC expressed in its vocabulary, not a generic harness. The loop walks
`self.scheduler.timesteps` exactly as DDIM did (twenty steps, one `predict_noise` per step, so NFE stays
at twenty). At each timestep I form the guided prediction $\tilde\epsilon=\epsilon_{uc}+s(\epsilon_c-\epsilon_{uc})$
and the data prediction by Tweedie, $m_t=z_0=(z_t-\sqrt{1-\bar\alpha_t}\,\tilde\epsilon)/\sqrt{\bar\alpha_t}$
— that single evaluation is both the corrector input for the step I took last iteration and the predictor
base for this one. I keep $\bar\alpha$ via `self.alpha(t)` and compute
$\lambda=\tfrac12(\log\bar\alpha-\log(1-\bar\alpha))$ from it, so I never leave the substrate's
$(\bar\alpha,z_0)$ language for a separate $\sigma$ schedule. The $R\rho=b$ machinery, the reduced-vs-full
solves, the `expm1` for the $h\phi$ factors, and the order ramp all go inside the body. The order ramp is
forced by how much history exists: the very first step has only $m_0$, so it is the plain first-order base
step with no correction and no corrector (there is no previous step to correct); the second step has one
past point, so predictor and corrector reach order 2; from the third step on, two past points are available
and both reach the capped order 3. The corrector always lags the predictor by one step — it needs the
current step's evaluation to refine the *previous* step — so the sequencing inside the loop is: evaluate,
correct the last step, append to history, then predict the current step. The bookkeeping here has to be
exact and is easy to get wrong: the corrector refines the step that *produced* the point I just evaluated,
so it must be built from the *previous* step's start state and levels (`x_prev`, `at_prev_s0`), not the
current step's — it is retroactively fixing the jump that landed me here, using the fresh model output as
the endpoint evidence that jump was missing. On the final step there is no
future evaluation to correct with (and the trajectory is nearly straight at low noise anyway), so it drops
order and simply returns the clean data prediction $z_0$, exactly as 3M's $\sigma=0$ step does. Two things I am
deliberately *not* importing: there is no thresholding (this is latent space, no $[-1,1]$ bound), and there
is no SDE noise term — UniPC is a deterministic predictor-corrector, and the whole point is that
*correcting* each step's leading error is a more direct fix than 3M's stochastic compensation for
uncorrected error. I keep the CFG++ convention in that the data prediction uses the guided $\tilde\epsilon$
while the unconditional stream is available for the base; the renoising that 2S/DDIM did with $\epsilon_{uc}$
is subsumed here into the data-prediction step.

The bar this has to clear is 3M's measured numbers, and I will state the falsifiable expectation against
them rather than invent a result. The claim is that correcting each step's leading truncation error
within the loop — at the same twenty calls, with no extra evaluation — beats predicting-and-renoising. So
I expect UniPC to land *below* 27.98 on SD v1.5, below 23.45 on SD v2.0, and below 39.66 on SDXL, with the
largest absolute improvement again on SDXL, where uncorrected per-step error has been the dominant cost at
every rung and where the $16.2$-point gap over SD v2.0 leaves the most to take. The specific mechanism I am
betting on is that the corrector buys a genuine order over 3M's predictor at equal NFE — the machine-zero-
on-quadratics behavior I measured against 3M's $2.5\times10^{-2}$-predictor behavior — and that a
deterministic corrected step does not need the $\eta=1.2$ noise 3M spent to mask its drift. Line the two up
at equal cost: 3M spent its one call per step on a third-order *predictor* and then added Langevin noise to
paper over the leading error that predictor could not remove; UniPC spends its one call per step on a
third-order predictor *plus* a corrector that recovers one more realized order from the same evaluation,
and needs no noise because it removes that leading error directly instead of masking it. Same twenty calls,
same base DDIM step, but where 3M's interior steps were third-order-predictor accurate, UniPC's corrected
interior steps realize the order above — and I verified that "order above" is exact-one-degree-higher, not
a hope. If that mechanism is right, the deterministic corrected march should reach a lower FID than the
noisy uncorrected one at the same budget. The way this is
falsified is clean: if UniPC fails to beat 39.66 on SDXL, then the corrector's extra order is not worth
more than the stochastic correction it replaces at this budget — in which case the right reading is that at
twenty steps 3M's noise was doing real work that a deterministic corrector cannot replicate, and the next
move would be a *stochastic* predictor-corrector rather than abandoning the corrector. But the expectation
I commit to is that the unified predictor-corrector, correcting where every previous rung only predicted,
is the strongest sampler of the four, and the place it shows most is SDXL.
