The kernel-ridge run told me, in numbers, exactly what a model with no scaling-law prior buys, and the
answer is: nothing usable off the training hull. Every one of the three families came back with a
*negative* held-out $R^2$ — vocab at $-0.567$, dataconstrained at $-13.42$, and lrbsz at a
catastrophic $-413.7$ with an `NMAE` of $16.2$, meaning the predictions on the held-out lrbsz points
were sixteen times the target's own spread away from the truth. That is not a model that learned a little
and missed a little; it is a model whose predictions collapsed when the query left the region it was fit
on, which is precisely the failure I expected from a pure-locality kernel. The RBF similarity
$\exp(-\gamma\lVert x - x_t\rVert^2)$ decays to zero as a held-out configuration moves larger and denser
than anything in the fit, the kernel sum goes to zero, and the prediction has no power-law tail and no
irreducible floor to fall back on. The vocab family, the most saturated and the closest to a clean
additive power law, was the *only* one that even approached zero ($-0.567$) — confirming that where the
test region is nearly inside the training hull the black box is merely bad, and where it is far outside
(lrbsz, dataconstrained) it is hopeless. So the diagnosis is sharp and it is not a flexibility problem:
the missing ingredient is the *right asymptotic form*. I have to stop being model-free and impose the
power-law-plus-floor structure that the literature laws carry, because that structure is the only thing
that extrapolates by construction.

So the next rung is symbolic: a compact expression per family with the Chinchilla bones —
$E + A/N^\alpha + \dots$ — fit per group by nonlinear least squares. But I should not just transcribe
the textbook additive law, because the additive law has a specific blindness that the kernel-ridge
numbers already hint at. Look at where kernel ridge was *least* bad versus *most* bad. Vocab, where the
three axes ($N$, $V$, $D$) plausibly act almost independently, was the near-miss. Lrbsz, where loss is a
*basin* in $(l, b)$ — there is an interior optimum and you pay for being off it in either direction, and
the two axes interact — was the disaster. The additive Chinchilla form $E + A/N^\alpha + B/D^\beta + \dots$
is a sum of monotone decaying terms; it literally cannot represent a basin (a sum of monotone functions
is monotone in each axis) and it cannot represent a *cross-axis interaction* (its terms are decoupled by
construction). So a faithful transcription of the human additive law would inherit exactly the lrbsz
weakness. The move for this rung is to keep the additive Chinchilla backbone where it works but add the
*interaction* terms the additive law drops — to discover, per family, the multiplicative cross-axis
couplings and basins that capture the residual variance the pure-additive form misses.

Let me derive each family's form from what its axes actually do, in discovery order.

Start with vocab, because it is where I want to confirm the additive backbone is mostly right and then
ask what little it is missing. The established Tao-style form is additive: $E + A\,N^{-\alpha} +
B\,V^{-\beta} + C\,D^{-\gamma}$, a floor plus one power term per axis. Kernel ridge's near-miss on vocab
says the additive backbone already explains most of the structure, so I keep it — but I suspect $V$ and
$D$ are not independent: a larger vocabulary changes how much each training character teaches the model
(more tokens to allocate the unigram mass across), so the *value of data* should depend on the
*vocabulary*. That is a $V\times D$ coupling the additive form cannot express. So I write the vocab law as
a single multiplicative power term across all three axes, $A\,N^{-a_1}V^{-a_2}D^{-a_3}$, which captures
the joint decay, plus an explicit cross term $A_{vd}\,V^{-g_1}D^{-g_2}$ that links vocab and data
directly, plus the floor $E$. The multiplicative term is the geometric-mean reading of scaling — loss
falls as a product of per-axis powers — and the cross term picks up whatever residual the product misses
specifically on the $(V, D)$ pair. The floor $E$ stays *unconstrained* (not exponentiated) because the
unigram-normalised vocab target can be negative, so I refuse to force the additive constant positive; the
scale and exponent parameters are exponentiated so the fitter explores a well-conditioned, positive
region. And because the vocab target can be negative I fit the residuals in the *linear* domain, not the
log domain — a log residual would be undefined on a negative target.

Now lrbsz, the family that was the disaster, and the one where I most need to add structure. The physics
is a basin: hold $N$ and $D$ fixed and sweep the learning rate $l$, and loss falls then rises around an
optimum $l^\star$; sweep the batch size $b$ and the same; and $l^\star$ and $b^\star$ are *coupled* —
the best learning rate depends on the batch size. A sum of monotone power terms cannot bend down then up,
so the additive Chinchilla form is structurally wrong here, which is why kernel ridge (and any
additive law) lands at a hugely negative $R^2$. The natural representation of a basin is a *quadratic*,
and the natural coordinates are logarithmic because the optima scale multiplicatively — so I work in
$\Delta_x = \log l - \log l^\star$ and $\Delta_y = \log b - \log b^\star$, the log-distances from the
fitted optima. A correlated quadratic bowl is then $k\,(\Delta_x^2 + \Delta_y^2 + 2\rho\,\Delta_x\Delta_y)$:
the $\Delta_x^2$ and $\Delta_y^2$ terms make it cost to be off the optimum in either axis, and the
$\rho\,\Delta_x\Delta_y$ cross term tilts the bowl so the ridge of low loss runs diagonally in
$(\log l, \log b)$ — exactly the lr/bsz coupling. I keep $\rho$ inside $(-1, 1)$ via a $\tanh$
reparameterization so the quadratic stays a genuine bowl rather than a saddle. Onto this basin I add a
Chinchilla base $E + A\,N^{-\alpha} + B\,D^{-\beta}$ for the part of the loss that depends on scale, not
on the optimizer settings. So the full lrbsz law is the scale-driven base plus a correlated log-quadratic
penalty for being off the $(l, b)$ optimum. Because the lm_loss target here is strictly positive I can
fit this one in the *log* domain, which is the homoscedastic residual for a multiplicative quantity, and
I give it the most restarts of the three (the basin's center and curvature are the hardest parameters to
pin from data, so the multi-start matters most here).

Now dataconstrained, where the defining fact is that the total tokens $D$ can exceed the unique pool $U$
because data is repeated, and a re-read token is worth less than a fresh one. The additive Chinchilla
$B\,D^{-\beta}$ treats every token as fresh, so it predicts that the eleventh epoch helps as much as the
first — false, and exactly the kind of asymptotic error that wrecks extrapolation to the denser test
points. So I replace $D$ with an *effective* token count that discounts repetition. The cleanest
discovered-style form is a multiplicative repeat-efficiency factor: define the repetition ratio
$D/U$, and an efficiency multiplier $1/(1 + (D/U)/R)$ that is near 1 when data is barely repeated and
decays smoothly as repetition grows, with $R$ a learned repeat-budget constant; the effective tokens are
$D_{\text{eff}} = D\cdot\text{efficiency}$, and the law is $E + A\,N^{-\alpha} + B\,D_{\text{eff}}^{-\beta}$.
Read the asymptotics: when $D \approx U$ (single epoch) the multiplier is $\approx 1$ and
$D_{\text{eff}} \approx D$, recovering the fresh-data Chinchilla term; as $D/U$ grows the multiplier
shrinks, $D_{\text{eff}}$ saturates, and additional repeated tokens stop driving the loss down — the
diminishing-returns behavior the additive law cannot represent. This target is strictly positive so I fit
in the log domain. The multiplicative-efficiency factor here is the discovered-style counterpart to the
human effective-token saturation — same physics (repeats are worth less), a different functional shape
($1/(1 + \text{ratio}/R)$ rather than an exponential saturation), which is exactly the spirit of this
rung: take the literature backbone and add the interaction/saturation structure as a *discovered*
symbolic form rather than the canonical human one.

The fitting machinery is shared across all three and is the one piece the scaffold loop hands me. For
each `group` I fit the family's form by nonlinear least squares with multi-start initializations — the
provided `least_squares` with a soft-$\ell_1$ robust loss so a few large-residual runs do not dominate —
choosing the linear or log residual per family as above, and I keep the best restart by mean-squared (or
mean-log-squared) error. Positive quantities are carried as exponentials of free parameters so the
optimizer stays in a well-conditioned region; signed quantities ($E$ on vocab, $\rho$ on lrbsz) are left
free or squashed. After fitting every group I store a median-of-groups parameter vector as the fallback
for any group unseen at predict time. The whole thing fits coefficients per group while keeping one
shared expression per family — the contract the task asks for. The full scaffold module is in the answer.

So the delta from the kernel-ridge rung is concrete: where the black box collapsed off the hull because
it had no asymptotic form, I now impose, per family, a power-law-plus-floor backbone with the specific
interaction structure each family needs — a $V\times D$ cross term and a joint multiplicative power for
vocab, a correlated log-quadratic basin for lrbsz, and a repeat-efficiency effective-token term for
dataconstrained. Here is what I expect this to fix and where I am unsure, stated against the
kernel-ridge numbers. Vocab and dataconstrained should swing from negative to strongly positive $R^2$ —
both have a near-additive backbone that the symbolic form captures, so I expect vocab to leap from
$-0.567$ into the low-$0.9$s and dataconstrained from $-13.4$ to around $0.9$, because the effective-token
term finally lets the law extrapolate to the denser repeated-data test points instead of treating them as
fresh. Lrbsz is the open question, and I can already feel the risk in the construction: the basin's
center $(l^\star, b^\star)$ and curvature have to be fit from the runs, and if the held-out lrbsz points
sit at a scale where the *optimum itself drifts* (the best learning rate moving with $N$ and $D$), a
single fitted center cannot track it — the bowl will be centered in the wrong place out-of-sample and the
quadratic penalty will be large and wrong. So I expect lrbsz to improve enormously over the
$-413.7$ disaster (the basin form is at least the right shape, and the `MAE`/`RMSE` should drop by an
order of magnitude) but to *stay negative* in $R^2$, because a fixed-center basin still cannot follow a
scale-dependent optimum. If that is what I see — vocab and dataconstrained rescued, lrbsz still negative
in $R^2$ but far better in absolute error — then the diagnosis for the next rung is already written: the
symbolic form is competitive on the easy families but the lrbsz optimum drifts with scale, and either I
need a law whose basin center is an explicit function of $(N, D)$, or I need a fundamentally different
engine on the hard families that does not rely on getting one hand-shaped basin right.
