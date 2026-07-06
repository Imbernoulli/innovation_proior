The kernel-ridge run told me, in numbers, exactly what a model with no scaling-law prior buys, and the
answer is: nothing usable off the training hull. Every one of the three families came back with a
*negative* held-out $R^2$ — vocab at $-0.567$, dataconstrained at $-13.42$, and lrbsz at a catastrophic
$-413.7$ with an `NMAE` of $16.2$. That last pair — a small `MAE` of $0.62$ sitting next to an `NMAE` of
$16$ — is strange enough that I should not move on until I understand what the metric is actually doing,
because if I misread it I will chase the wrong quantity for three rungs. So let me reverse-engineer the
normalizer from the table itself. The `NMAE` and the `R^2` are computed on the same test targets, and the
relations $R^2 = 1 - \mathrm{RMSE}^2/\sigma^2$ and $\mathrm{NMAE} = \mathrm{MAE}/\sigma$ both reference the
test target's own spread $\sigma$. If that is the normalizer, I can recover $\sigma$ two independent ways
and check they agree. From vocab, $\sigma = \mathrm{RMSE}/\sqrt{1 - R^2} = 1.0586/\sqrt{1.5672} = 0.846$,
and then $\mathrm{MAE}/\sigma = 1.0383/0.846 = 1.228$ — which is the reported `NMAE` to the digit. So the
normalizer is confirmed: `NMAE` is `MAE` in units of the test standard deviation, and `R^2` is one minus
the squared `RMSE` in those same units. That single fact reorganizes everything I am about to do, because
it means each family's difficulty is set by its $\sigma$, and I can now read $\sigma$ off the table for all
three.

Doing that is the most useful thing I can extract from this rung. Vocab: $\sigma = 0.846$. Dataconstrained:
$\sigma = 2.1057/\sqrt{14.419} = 0.554$. Lrbsz: $\sigma = 0.7774/\sqrt{414.70} = 0.0382$. Stare at that
last number. The held-out lrbsz targets have a standard deviation of *four hundredths of a loss unit* —
the test configurations all sit near the bottom of an lm-loss basin, so their losses barely differ from
one another, and the whole family is a needle. That is why lrbsz punished kernel ridge so savagely: the
$R^2$ denominator is $\sigma^2 = 0.00146$, so an `RMSE` of $0.777$ becomes $(0.777/0.0382)^2 = 415$ and
$R^2 = -414$. The catastrophe was never a large absolute error — $\mathrm{RMSE}/\sigma$ is $20$ for lrbsz,
against only $1.25$ for vocab and $3.8$ for dataconstrained. Vocab's ratio of $1.25$ is the quantitative
version of "merely bad": kernel ridge's held-out error there was only a quarter larger than just guessing
a constant, consistent with predictions drifting toward zero while the vocab target is a near-zero-centered
signed quantity of comparable spread. Lrbsz's ratio of $20$ is the quantitative version of "hopeless." So
the ordering I predicted at the last rung — vocab least bad, lrbsz worst — is exactly what the $\sigma$
reconstruction shows, and it shows *why*: not that the black box tried harder on vocab, but that lrbsz's
needle-thin target makes any imperfect fit look like a disaster. This is the bar the next rung has to clear,
and it is a brutal one on lrbsz: to reach even $R^2 = 0$ there I need `RMSE` below $0.0382$, i.e. to
predict held-out lm-loss to within four hundredths.

With the metric understood, the diagnosis of the *mechanism* is unchanged and sharp: the RBF similarity
$\exp(-\gamma\lVert x - x_t\rVert^2)$ decays to zero as a held-out configuration moves larger and denser
than anything in the fit, the kernel sum goes to zero, and the prediction has no power-law tail and no
irreducible floor to fall back on. Vocab, the most saturated family whose test region nearly overlaps the
training hull, was the only near-miss; lrbsz and dataconstrained, whose test points sit further out, were
hopeless. So it is not a flexibility problem: the missing ingredient is the *right asymptotic form*. I have
to stop being model-free and impose the power-law-plus-floor structure that the literature laws carry,
because that structure is the only thing that extrapolates by construction.

So the next rung is symbolic: a compact expression per family with the Chinchilla bones —
$E + A/N^\alpha + \dots$ — fit per group by nonlinear least squares. But I should not just transcribe the
textbook additive law, because the additive law has a specific blindness that the kernel-ridge numbers
already hint at. Look at where kernel ridge was *least* bad versus *most* bad. Vocab, where the three axes
($N$, $V$, $D$) plausibly act almost independently, was the near-miss. Lrbsz, where loss is a *basin* in
$(l, b)$ — there is an interior optimum and you pay for being off it in either direction, and the two axes
interact — was the disaster. I can make the additive law's blindness exact rather than intuitive. Take the
additive Chinchilla form $L = E + A\,N^{-\alpha} + B\,D^{-\beta}$ and ask what it says about the learning
rate: $\partial L/\partial l = 0$ identically, because $l$ does not appear. The additive law is *flat* in
the entire $(l, b)$ plane — it cannot see the optimizer axes at all, let alone bend down and back up around
an optimum. A sum of monotone decaying terms is monotone in each axis it *does* contain, so it also cannot
represent a cross-axis interaction: its terms are decoupled by construction. So a faithful transcription of
the human additive law would leave lrbsz exactly as blind as kernel ridge left it — flat where the truth is
a bowl. The move for this rung is to keep the additive Chinchilla backbone where it works but add the
*interaction* terms the additive law drops — to discover, per family, the multiplicative cross-axis
couplings and basins that capture the residual variance the pure-additive form misses.

Before deriving the three forms I should rule out the two neighboring choices, because "additive backbone
plus interactions" is a middle path and I want to know the endpoints are worse. One endpoint is the pure
additive Chinchilla law transcribed verbatim — I have just shown it is flat in the optimizer axes, so it is
a non-starter on lrbsz; keeping it would concede the hard family before I begin. The other endpoint is a
single fully-multiplicative power, $L = A\,N^{-a_1}V^{-a_2}D^{-a_3}$ with no additive floor. That form has
one fatal property: as any scale grows it runs to zero, so it carries no irreducible floor, and worse, it
is strictly positive and cannot represent the vocab target, which is a unigram-normalised loss that goes
negative. So a bare product cannot even sit in the right sign regime for vocab. What both failures point to
is that I need the additive floor $E$ carried explicitly — an unconstrained constant the multiplicative and
cross terms decay *toward* — with the interaction structure layered on top of it rather than replacing it.
That is exactly the middle path, and the two rejected endpoints are why it is the right one: keep the
additive floor for sign and asymptote, keep the power terms for the decay, add the interactions the pure
additive form drops.

Let me derive each family's form from what its axes actually do, in discovery order.

Start with vocab, because it is where I want to confirm the additive backbone is mostly right and then ask
what little it is missing. The established Tao-style form is additive: $E + A\,N^{-\alpha} + B\,V^{-\beta} +
C\,D^{-\gamma}$, a floor plus one power term per axis. Kernel ridge's near-miss on vocab (the
$\mathrm{RMSE}/\sigma = 1.25$ I just reconstructed) says the additive backbone already explains most of the
structure, so I keep it — but I suspect $V$ and $D$ are not independent: a larger vocabulary changes how
much each training character teaches the model (more tokens to allocate the unigram mass across), so the
*value of data* should depend on the *vocabulary*. That is a $V\times D$ coupling the additive form cannot
express. So I write the vocab law as a single multiplicative power term across all three axes,
$A\,N^{-a_1}V^{-a_2}D^{-a_3}$, which captures the joint decay, plus an explicit cross term
$A_{vd}\,V^{-g_1}D^{-g_2}$ that links vocab and data directly, plus the floor $E$. The multiplicative term
is the geometric-mean reading of scaling — loss falls as a product of per-axis powers — and the cross term
picks up whatever residual the product misses specifically on the $(V, D)$ pair. I should be honest that
this cross term is a hunch, not a certainty: if $V$ and $D$ really do act independently on this data, the
fitter will drive $A_{vd}$ toward zero and I will have paid two extra parameters for nothing, landing right
back at the additive backbone's number. That is a fine outcome — it would tell me the interaction I
suspected is inert — and it is the reason I keep the backbone intact rather than betting the whole form on
the coupling. The floor $E$ stays *unconstrained* (not exponentiated) because the unigram-normalised vocab
target can be negative, so I refuse to force the additive constant positive; the scale and exponent
parameters are exponentiated so the fitter explores a well-conditioned, positive region. And because the
vocab target can be negative I fit the residuals in the *linear* domain, not the log domain — a log
residual would be undefined on a negative target.

Now lrbsz, the family that was the disaster, and the one where I most need to add structure. The physics is
a basin: hold $N$ and $D$ fixed and sweep the learning rate $l$, and loss falls then rises around an
optimum $l^\star$; sweep the batch size $b$ and the same; and $l^\star$ and $b^\star$ are *coupled* — the
best learning rate depends on the batch size. A sum of monotone power terms cannot bend down then up, which
is why kernel ridge (and any additive law) lands at a hugely negative $R^2$. The natural representation of a
basin is a *quadratic*, and the natural coordinates are logarithmic because the optima scale
multiplicatively — so I work in $\Delta_x = \log l - \log l^\star$ and $\Delta_y = \log b - \log b^\star$,
the log-distances from the fitted optima. A correlated quadratic bowl is then $k\,(\Delta_x^2 + \Delta_y^2 +
2\rho\,\Delta_x\Delta_y)$: the $\Delta_x^2$ and $\Delta_y^2$ terms make it cost to be off the optimum in
either axis, and the $\rho\,\Delta_x\Delta_y$ cross term tilts the bowl so the ridge of low loss runs
diagonally in $(\log l, \log b)$ — exactly the lr/bsz coupling. I have to make sure this stays a *bowl*
rather than a saddle, and that is a concrete check on $\rho$: the Hessian of the penalty is
$2k$ times the $2\times 2$ matrix with unit diagonal and off-diagonal $\rho$, with determinant
$4k^2(1-\rho^2)$ and trace
$4k$. For $k > 0$ this is positive definite exactly when $1 - \rho^2 > 0$, i.e. $|\rho| < 1$; at
$|\rho| = 1$ the determinant hits zero (a degenerate trough) and past it the form is indefinite (a saddle,
which would predict loss *decreasing* without bound along one diagonal — physically absurd for a penalty).
So I reparameterize $\rho = \tanh(\cdot)$, which lives strictly in $(-1, 1)$ and can never cross into the
saddle regime no matter what the optimizer does. Onto this basin I add a Chinchilla base $E + A\,N^{-\alpha}
+ B\,D^{-\beta}$ for the part of the loss that depends on scale, not on the optimizer settings, and I add it
rather than multiply for a reason I can check at the limit: at the optimizer optimum $\Delta_x = \Delta_y =
0$ the penalty vanishes and the law reduces to exactly the base $E + A\,N^{-\alpha} + B\,D^{-\beta}$. That
is the physically correct reading — the base *is* the loss a perfectly-tuned $(l, b)$ run would pay, and
being mistuned adds a cost on top; a multiplicative coupling would instead rescale the base by the penalty,
conflating "how big the loss is" with "how mistuned the run is." So the full lrbsz law is the scale-driven
base plus a correlated log-quadratic penalty for being off the $(l, b)$ optimum, reducing cleanly to the
base at the optimum. Because the lm_loss target here is strictly positive I can fit this one in the *log* domain, which
is the homoscedastic residual for a multiplicative quantity, and I give it the most restarts of the three
(the basin's center and curvature are the hardest parameters to pin from data, so the multi-start matters
most here).

Now dataconstrained, where the defining fact is that the total tokens $D$ can exceed the unique pool $U$
because data is repeated, and a re-read token is worth less than a fresh one. The additive Chinchilla
$B\,D^{-\beta}$ treats every token as fresh, so it predicts that the eleventh epoch helps as much as the
first — false, and exactly the kind of asymptotic error that wrecks extrapolation to the denser test
points. So I replace $D$ with an *effective* token count that discounts repetition. The cleanest
discovered-style form is a multiplicative repeat-efficiency factor: define the repetition ratio $D/U$, and
an efficiency multiplier $1/(1 + (D/U)/R)$ that is near 1 when data is barely repeated and decays smoothly
as repetition grows, with $R$ a learned repeat-budget constant; the effective tokens are $D_{\text{eff}} =
D\cdot\text{efficiency}$, and the law is $E + A\,N^{-\alpha} + B\,D_{\text{eff}}^{-\beta}$. Let me trace its
asymptotics to be sure they are the ones I want. When $D \approx U$ (single epoch) the multiplier is
$1/(1 + 1/R)$ — for a mid-range $R = 5$ that is $0.83$, so $D_{\text{eff}} \approx 0.83\,D$, essentially the
fresh-data term with a mild discount. As $D/U \to \infty$ the effective count does *not* run to infinity:
$D_{\text{eff}} = D/(1 + (D/U)/R) \to U R$, a finite ceiling, so $B\,D_{\text{eff}}^{-\beta} \to
B\,(UR)^{-\beta}$, a floor no amount of repetition can push below. That saturating ceiling is precisely the
diminishing-returns behavior the additive law cannot represent — and it is set by $UR$, the unique pool
times the learned repeat budget, which is a physically sensible knob. This target is strictly positive so I
fit in the log domain. The multiplicative-efficiency factor here is the discovered-style counterpart to the
human effective-token saturation — same physics (repeats are worth less), a different functional shape
($1/(1 + \text{ratio}/R)$ rather than an exponential saturation), which is exactly the spirit of this rung:
take the literature backbone and add the interaction/saturation structure as a *discovered* symbolic form
rather than the canonical human one.

The fitting machinery is shared across all three and is the one piece the scaffold loop hands me. For each
`group` I fit the family's form by nonlinear least squares with multi-start initializations — the provided
`least_squares` with a soft-$\ell_1$ robust loss so a few large-residual runs do not dominate — choosing
the linear or log residual per family as above, and I keep the best restart by mean-squared (or
mean-log-squared) error. The soft-$\ell_1$ choice matters because a single mis-recorded run in a group of a
few hundred would, under plain squared loss, drag the whole coefficient vector toward it; soft-$\ell_1$
grows linearly in the tail so the fit is anchored by the bulk. Positive quantities are carried as
exponentials of free parameters so the optimizer stays in a well-conditioned region (an exponent that must
stay positive never has to cross zero, where the power term is non-differentiable); signed quantities ($E$
on vocab, $\rho$ on lrbsz) are left free or squashed. The restart budget is not uniform, and the reason is
identifiability: counting free parameters, vocab carries eight ($E, A, a_1, a_2, a_3, A_{vd}, g_1, g_2$),
dataconstrained six ($E, A, \alpha, B, \beta, R$), and lrbsz nine ($E, A, \alpha, B, \beta, k, \log
l^\star, \log b^\star, \rho$). With a few hundred runs per group all three are nominally over-determined,
but the lrbsz nine are the *least* identifiable — the two basin centers, the curvature $k$, and the
correlation $\rho$ trade off against one another, and a shift of the center can be partly absorbed by a
change in curvature, so the objective has long shallow valleys that a single start can slide into and stop.
That is exactly why lrbsz gets the most restarts: I need several independent descents to have a chance of
finding the true bowl rather than a degenerate near-flat one. After fitting every group I store a
median-of-groups parameter vector as the fallback for any group unseen at predict time — a median rather
than a mean so one badly-fit group cannot skew the fallback.

There is a subtlety in the linear-versus-log residual choice worth being honest about, because the fit
domain and the scoring domain are not the same. The metric $R^2$ is computed on *linear* residuals in the
raw target domain, but for lrbsz and dataconstrained I fit by minimizing *log* residuals. Is that a
mismatch that will cost me at scoring time? For a strictly-positive target clustered near a value $y_0$, a
log residual is $\log(\hat y/y) \approx (\hat y - y)/y_0$ to first order — the linear residual rescaled by
$1/y_0$ — so minimizing the log residual minimizes a scale-normalized version of the very thing the metric
measures, and the two agree up to a constant when the target's spread is small relative to its level. That
is exactly the lrbsz and dataconstrained regime (tight positive clusters), so the log fit is a
well-conditioned proxy for the linear objective, with the bonus that it puts groups at different loss
levels on a common footing. For vocab the log domain is simply unavailable — the target crosses zero — so
I fit linear residuals there, which happens to be the exact domain the metric scores in. So the residual
choice is not arbitrary: it is log where log is defined and a faithful proxy, linear where log is undefined
and linear is what the score wants anyway. The whole thing fits coefficients per group while keeping one
shared expression per family — the contract the task asks for. The full scaffold module is in the answer.

So the delta from the kernel-ridge rung is concrete: where the black box collapsed off the hull because it
had no asymptotic form, I now impose, per family, a power-law-plus-floor backbone with the specific
interaction structure each family needs — a $V\times D$ cross term and a joint multiplicative power for
vocab, a correlated log-quadratic basin for lrbsz, and a repeat-efficiency effective-token term for
dataconstrained. Now I can turn the reconstructed $\sigma$ values into falsifiable predictions instead of
hand-waving. On vocab, the additive backbone should shrink `RMSE` from kernel ridge's $1.06$ down to the
scale of the in-family noise — call it $\sim 0.23$ — and with $\sigma = 0.846$ that gives
$R^2 = 1 - (0.23/0.846)^2 = 0.93$, so I expect vocab in the low-$0.9$s. On dataconstrained the effective-token
term should let the law extrapolate to the denser repeated-data test points instead of treating them as
fresh; if `RMSE` falls to $\sim 0.15$, then with $\sigma = 0.554$ that is $R^2 = 1 - (0.15/0.554)^2 = 0.93$,
again low-$0.9$s. Lrbsz is the open question, and the $\sigma$ arithmetic makes the risk quantitative. The
basin's center $(l^\star, b^\star)$ and curvature have to be fit from the runs, and if the held-out lrbsz
points sit at a scale where the *optimum itself drifts* (the best learning rate moving with $N$ and $D$), a
single fitted center cannot track it — the bowl will be centered in the wrong place out-of-sample and the
quadratic penalty will be large and wrong. The right basin shape should still cut `MAE` by an order of
magnitude, from $0.62$ toward $\sim 0.06$, because at least it is bending the surface the right way; but
watch what that does to $R^2$. An `MAE` of $0.06$ implies an `RMSE` around $0.077$, and against lrbsz's
needle-thin $\sigma = 0.0382$ that is $R^2 = 1 - (0.077/0.0382)^2 = 1 - 4.1 = -3.1$. So the very same fit
that is an order of magnitude better in absolute terms is *still* worse than the mean in $R^2$, purely
because $\sigma$ is so small that the bar for a positive $R^2$ ($\mathrm{RMSE} < 0.0382$) is nearly
unreachable for a fixed-center basin. If that is what I see — vocab and dataconstrained rescued into the
low-$0.9$s, lrbsz still negative near $-3$ in $R^2$ but an order of magnitude better in absolute error —
then the diagnosis for the next rung is already written: the symbolic form is competitive on the easy
families, but the lrbsz optimum drifts with scale, and either I need a law whose basin center is an explicit
function of $(N, D)$, or I need a fundamentally different engine on the hard family that does not rely on
getting one hand-shaped basin right.
