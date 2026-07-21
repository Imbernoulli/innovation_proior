The kernel-ridge run told me, in numbers, exactly what a model with no scaling-law prior buys off the
training hull: nothing. Every family came back with a negative held-out $R^2$ — vocab at $-0.567$,
dataconstrained at $-13.42$, and lrbsz at a catastrophic $-413.7$ with an `NMAE` of $16.2$. That last pair
— a small `MAE` of $0.62$ next to an `NMAE` of $16$ — is strange enough that I should not move on until I
understand what the metric is doing, or I will chase the wrong quantity for three rungs. So reverse-engineer
the normalizer from the table. `NMAE` and `R^2` are computed on the same test targets, and
$R^2 = 1 - \mathrm{RMSE}^2/\sigma^2$ and $\mathrm{NMAE} = \mathrm{MAE}/\sigma$ both reference the test
target's own spread $\sigma$. If so, I can recover $\sigma$ two ways and check they agree: from vocab,
$\sigma = \mathrm{RMSE}/\sqrt{1 - R^2} = 1.0586/\sqrt{1.5672} = 0.846$, and $\mathrm{MAE}/\sigma =
1.0383/0.846 = 1.228$, the reported `NMAE` to the digit. So `NMAE` is `MAE` in units of the test standard
deviation and `R^2` is one minus the squared `RMSE` in those same units — which means each family's
difficulty is set by its $\sigma$, and I can read $\sigma$ off the table for all three.

Vocab: $\sigma = 0.846$. Dataconstrained: $\sigma = 2.1057/\sqrt{14.419} = 0.554$. Lrbsz:
$\sigma = 0.7774/\sqrt{414.70} = 0.0382$. That last number is the whole story. The held-out lrbsz targets
have a standard deviation of *four hundredths of a loss unit* — the test configurations all sit near the
bottom of an lm-loss basin, so their losses barely differ, and the family is a needle. That is why lrbsz
punished kernel ridge so savagely: with $\sigma^2 = 0.00146$ an `RMSE` of $0.777$ becomes
$(0.777/0.0382)^2 = 415$. The catastrophe was never a large absolute error — $\mathrm{RMSE}/\sigma$ is $20$
for lrbsz against $1.25$ for vocab and $3.8$ for dataconstrained. Vocab's $1.25$ is "merely bad": the
held-out error was only a quarter larger than guessing a constant, consistent with predictions drifting
toward zero while the vocab target is a near-zero-centered signed quantity of comparable spread. Lrbsz's
$20$ is "hopeless." So the ordering I predicted last rung — vocab least bad, lrbsz worst — is what the
$\sigma$ reconstruction shows, and it shows *why*: not that the black box tried harder on vocab, but that
lrbsz's needle-thin target makes any imperfect fit look like a disaster. The bar the next rung has to clear
is brutal on lrbsz: to reach even $R^2 = 0$ I need `RMSE` below $0.0382$ — held-out lm-loss predicted to
within four hundredths.

The mechanism is the one I predicted: the RBF sum decays to zero off the hull with no power-law tail and no
floor, so the missing ingredient is not flexibility but the *right asymptotic form*. I have to impose the
power-law-plus-floor structure the literature laws carry, because that is the only thing that extrapolates
by construction.

So the next rung is symbolic: a compact expression per family with the Chinchilla bones —
$E + A/N^\alpha + \dots$ — fit per group by nonlinear least squares. But I should not just transcribe the
textbook additive law, because it has a specific blindness the kernel-ridge numbers already hint at. Vocab,
where the three axes ($N$, $V$, $D$) plausibly act almost independently, was the near-miss; lrbsz, where
loss is a *basin* in $(l, b)$ with an interior optimum and coupled axes, was the disaster. Make the
blindness exact: take $L = E + A\,N^{-\alpha} + B\,D^{-\beta}$ and ask about the learning rate —
$\partial L/\partial l = 0$ identically, because $l$ does not appear. The additive law is *flat* in the
entire $(l, b)$ plane, and a sum of monotone decaying terms is monotone in each axis it contains, so it
cannot represent a cross-axis interaction either. A faithful transcription would leave lrbsz as blind as
kernel ridge left it. The move is to keep the additive backbone where it works but add the *interaction*
terms the additive law drops — per family, the multiplicative cross-axis couplings and basins that capture
the residual variance.

The two endpoints fix the shape. A bare fully-multiplicative power $A\,N^{-a_1}V^{-a_2}D^{-a_3}$ runs to
zero as any scale grows (no floor) and is strictly positive, so it cannot even sit in the sign regime the
vocab target needs. Both the flat-additive and the floorless-multiplicative failures say to carry the
additive floor $E$ *explicitly* — an unconstrained constant the decaying terms fall toward — with the
interaction structure layered on top: floor for sign and asymptote, power terms for decay, interactions for
what the additive form drops. Let me derive each family's form from what its axes do.

Vocab first, to confirm the additive backbone is mostly right and ask what little it misses. The Tao-style
form is additive, $E + A\,N^{-\alpha} + B\,V^{-\beta} + C\,D^{-\gamma}$, and kernel ridge's near-miss says
that backbone already explains most of the structure. But I suspect $V$ and $D$ are not independent: a
larger vocabulary changes how much each training character teaches (more tokens to allocate the unigram
mass across), so the value of data should depend on the vocabulary — a $V\times D$ coupling the additive
form cannot express. So I write the vocab law as a single multiplicative power $A\,N^{-a_1}V^{-a_2}D^{-a_3}$
across all three axes, plus an explicit cross term $A_{vd}\,V^{-g_1}D^{-g_2}$ linking vocab and data, plus
the floor $E$. This cross term is a hunch, not a certainty: if $V$ and $D$ act independently the fitter will
drive $A_{vd}$ toward zero and I will have paid two parameters for nothing, landing back at the additive
backbone's number — a fine outcome, since it would tell me the interaction is inert, and the reason I keep
the backbone intact rather than betting the whole form on the coupling. The floor $E$ stays *unconstrained*
(not exponentiated) because the unigram-normalised target can be negative; the scale and exponent parameters
are exponentiated so the fitter explores a well-conditioned positive region. And because the target can be
negative I fit the residuals in the *linear* domain — a log residual is undefined on a negative target.

Now lrbsz. The physics is a basin: hold $N, D$ fixed and sweep $l$, and loss falls then rises around an
optimum $l^\star$; sweep $b$ and the same; and $l^\star, b^\star$ are coupled. The natural representation is
a quadratic, and the natural coordinates are logarithmic because the optima scale multiplicatively — so I
work in $\Delta_x = \log l - \log l^\star$ and $\Delta_y = \log b - \log b^\star$. A correlated quadratic
bowl is $k\,(\Delta_x^2 + \Delta_y^2 + 2\rho\,\Delta_x\Delta_y)$: the squared terms cost being off the
optimum in either axis, and the $\rho$ cross term tilts the bowl so the low-loss ridge runs diagonally in
$(\log l, \log b)$ — the lr/bsz coupling. It has to stay a bowl rather than a saddle, which is a constraint
on $\rho$: the penalty's Hessian is $2k$ times the matrix with unit diagonal and off-diagonal $\rho$, with
determinant $4k^2(1-\rho^2)$, so for $k > 0$ it is positive definite exactly when $|\rho| < 1$; at
$|\rho| = 1$ it degenerates and past it the form is a saddle predicting loss decreasing without bound along
one diagonal. So I reparameterize $\rho = \tanh(\cdot) \in (-1, 1)$, which can never cross into the saddle
regime. Onto the basin I *add* a Chinchilla base $E + A\,N^{-\alpha} + B\,D^{-\beta}$ for the scale-driven
part of the loss, and I add rather than multiply because at the optimum $\Delta_x = \Delta_y = 0$ the
penalty vanishes and the law reduces to exactly the base — which is the physically correct reading: the base
*is* the loss a perfectly-tuned $(l, b)$ run would pay, and being mistuned adds a cost on top, whereas a
multiplicative coupling would rescale the base by the penalty, conflating how big the loss is with how
mistuned the run is. The lm_loss target is strictly positive so I fit this one in the log domain, and I give
it the most restarts because the basin's center and curvature are the hardest parameters to pin.

Now dataconstrained, where the total tokens $D$ can exceed the unique pool $U$ because data is repeated, and
a re-read token is worth less than a fresh one. The additive $B\,D^{-\beta}$ treats every token as fresh, so
it predicts the eleventh epoch helps as much as the first — false, and exactly the asymptotic error that
wrecks extrapolation to denser test points. So I replace $D$ with an *effective* count that discounts
repetition: a multiplicative efficiency multiplier $1/(1 + (D/U)/R)$ near 1 when data is barely repeated and
decaying as repetition grows, with $R$ a learned repeat-budget constant, giving $D_{\text{eff}} =
D\cdot\text{efficiency}$ in $E + A\,N^{-\alpha} + B\,D_{\text{eff}}^{-\beta}$. Its asymptotics are the ones I
want: at $D \approx U$ (single epoch) the multiplier is $1/(1 + 1/R)$ — a mild discount — and as
$D/U \to \infty$ the effective count does *not* diverge but tends to $UR$, a finite ceiling, so
$B\,D_{\text{eff}}^{-\beta} \to B\,(UR)^{-\beta}$, a floor no repetition can push below. That saturating
ceiling, set by the unique pool times the learned repeat budget, is precisely the diminishing-returns
behavior the additive law cannot represent. This target is strictly positive so I fit in the log domain.
The multiplicative-efficiency factor is the discovered-style counterpart to the human effective-token
saturation — same physics, a different shape ($1/(1 + \text{ratio}/R)$ rather than an exponential) — which
is the spirit of this rung: literature backbone plus discovered interaction structure.

The fitting machinery is shared. For each `group` I fit the form by nonlinear least squares with multi-start
initializations (the provided `least_squares` with a soft-$\ell_1$ robust loss so a few large-residual runs
do not dominate — soft-$\ell_1$ grows linearly in the tail so the fit is anchored by the bulk), choosing
the linear or log residual per family, and keep the best restart by mean-(log-)squared error. Positive
quantities are carried as exponentials of free parameters so the optimizer never has to cross zero where a
power term is non-differentiable; signed quantities ($E$ on vocab, $\rho$ on lrbsz) are left free or
squashed. The restart budget is not uniform: lrbsz's nine parameters
($E, A, \alpha, B, \beta, k, \log l^\star, \log b^\star, \rho$) are the least identifiable — the two basin
centers, the curvature, and $\rho$ trade against one another and the objective has long shallow valleys —
so lrbsz gets the most restarts, against vocab's eight and dataconstrained's six. After fitting every group
I store a median-of-groups parameter vector as the fallback for any unseen group.

The linear-vs-log residual choice is not arbitrary even though the metric scores linear residuals. For a
strictly-positive target clustered near $y_0$, a log residual $\log(\hat y/y) \approx (\hat y - y)/y_0$ is
the linear residual rescaled by $1/y_0$, so minimizing it minimizes a scale-normalized version of what the
metric measures — well-conditioned in exactly the tight-positive-cluster regime lrbsz and dataconstrained
live in, with the bonus of putting different loss levels on common footing. For vocab the log domain is
unavailable (the target crosses zero), so I fit linear residuals, the exact domain the metric scores.

Against kernel ridge, then: where the black box collapsed off the hull for want of an asymptotic form, I now
impose per family a power-law-plus-floor backbone with the interaction each needs — a $V\times D$ cross term
and a joint multiplicative power for vocab, a correlated log-quadratic basin for lrbsz, a repeat-efficiency
effective-token term for dataconstrained. Read family by family through the locked $\sigma$ values: vocab
and dataconstrained should flip cleanly from negative to solidly positive $R^2$, because the additive
backbone and the saturating effective-token term extrapolate to the larger/denser test points by
construction where kernel ridge only collapsed.

Lrbsz is the open question, and the $\sigma$ arithmetic sets the risk. The basin's center $(l^\star,
b^\star)$ is fit from the runs, and if the held-out points sit at a scale where the *optimum itself drifts*
— the best learning rate moving with $N$ and $D$ — a single fitted center cannot track it, so out-of-sample
the bowl is centered wrong and its penalty is large and wrong. Bending the surface the right way should
still cut the absolute error by an order of magnitude from kernel ridge's `MAE` $0.62$, because it at least
has the shape. But the needle-thin $\sigma = 0.0382$ means a positive $R^2$ needs `RMSE` below $0.0382$, and
a fixed-center basin has little chance of clearing it. So I expect the same fit an order of magnitude better
in absolute terms and *still* negative in $R^2$. If that is the pattern — vocab and dataconstrained rescued
into strong positives, lrbsz far better in absolute error but still worse than the mean — then the next rung
is already framed: the lrbsz optimum drifts with scale, and I need either a law whose basin center is an
explicit function of $(N, D)$ or a different engine on the hard family that does not rely on one hand-shaped
basin.
