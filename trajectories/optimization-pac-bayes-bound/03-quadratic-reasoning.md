McAllester did exactly what I bet it would, and the KL numbers are the proof. Removing the `λ` knob
collapsed and stabilized the KL: on MNIST-FCN the mean fell from Catoni's 539 to 81, and — the part I
care about most — it is now *tight across seeds*, 71.90 / 80.28 / 91.83, no bimodal blowup. The
seed-to-seed high/low ratio fell from Catoni's 5.7× to 91.83/71.90 = 1.28×, right in the "under 1.5×"
band I was hoping for, so the two-attractor system really did collapse to one basin the moment the knob
was gone. On MNIST-CNN the KL fell from 192 to 11 and is rock-stable (10.0 / 12.1 / 12.1, ratio 1.21×);
FashionMNIST-CNN from 286 to 32 (38.32 / 30.87 / 27.29). The certificate followed the KL down almost
one-for-one, exactly as the separable-inversion argument predicted: MNIST-FCN mean 0.0558 → 0.0373,
MNIST-CNN 0.0250 → 0.0164, FashionMNIST-CNN 0.1215 → 0.1002. So the parameter-free additive bound is the
right *stability* fix — the diagnosis that the runaway KL was a `λ`-mechanism artifact, not a
weak-penalty problem, was correct, and the fact that the weak sublinear KL gradient was *enough* once
nothing pushed KL up is now confirmed rather than conjectured.

But now look at where the additive bound is *stuck*, because that is the next wall, and I want to locate
it precisely rather than wave at it. On MNIST-CNN the empirical 0-1 risk is 0.0119 and the KL is only 11
nats over `n ≈ 30000` — the posterior is essentially perfect and barely moved from the prior — yet the
certificate sits at 0.0164. The gap between the certificate and the empirical risk is ~0.0045, and it is
*not* coming from a large KL anymore. Decompose it: at emp 0.012 and KL 11 the inversion budget is
`c = (11+9.54)/30000 = 6.85·10⁻⁴`, and `inv_kl(0.012, 6.85·10⁻⁴) ≈ 0.0165`; drop the KL all the way to
*zero* and the floor from the confidence term alone is `inv_kl(0.012, 9.54/30000) ≈ 0.0150`. So of the
0.0045 daylight, killing the entire residual KL would recover at most 0.0015 — two-thirds of the gap is
the irreducible `Λ/n` confidence budget and the shape of the inversion, not complexity I can still
squeeze. The `ce_bound` column says the same thing in the training metric: McAllester's MNIST-CNN
ce_bound is 0.0805 at emp-NLL that is tiny and KL 11, which means the additive `√((KL+Λ)/(2n))` term is
contributing essentially all of it — `√((11+9.54)/60000) = √(3.42·10⁻⁴) = 0.0185` is the KL part, and the
rest is the unrescaled NLL. The point stands across settings: on MNIST-FCN emp 0.0235, KL 81 gives
certificate 0.0373, and even at KL 0 the floor is `inv_kl(0.0235, 9.54/30000) ≈ 0.0298`, so again most of
the ~0.014 gap above emp is confidence-budget-plus-shape, not squeezable KL. With KL this small,
`kl_term = (KL + Λ)/(2n)` is around `1e-3`, and the additive bound adds `√(1e-3) ≈ 0.0316` on top of the
empirical risk — except the empirical risk is already near zero, so on the *training* bound the
certificate-shaped quantity is essentially `√(kl_term)` itself. The square root is the floor. McAllester
fixed the KL; the residual looseness is now the `√` shape of the relaxation, not the complexity it is
penalizing.

So the target is precise: I have a small, stable KL, and I am being penalized by the square-root shape of
the relaxation rather than by the complexity itself. I need a bound whose complexity contribution is
*linear* in `kl_term` at small empirical risk, not the square root — without reintroducing a free `λ`,
because the Catoni run showed me what a free knob costs. Let me lay out the honest options before I derive
anything, because more than one thing could plausibly attack this wall. The first is to train the network
*directly* on the tight object I already report at certificate time — the numerical `inv_kl` of the parent
— instead of on any closed relaxation at all. That is appealing because it is by construction the tightest
per the parent, but I have to check differentiability and cost: `inv_kl` here is a bisection loop over the
binary-KL root, and gradients would have to flow either through 200 unrolled bisection steps or via an
implicit-function-theorem correction at the root; the first is a heavy, memory-hungry graph per minibatch,
the second is fiddly to get right and easy to mis-sign. The whole reason the parameter-free additive bound
was well-behaved is that it is a *closed, convex, cheaply differentiable* objective; trading that for a
non-smooth bisection in the training loop reintroduces exactly the kind of optimization fragility I just
spent a rung removing. So I set that aside: I want a closed-form training bound, and I will keep `inv_kl`
where it belongs, at certificate time only. The second option is to bring back Catoni but *pinned* to its
analytic optimum — but I settled that last rung on arithmetic: the optimal `λ*→1` sits at the point of
*minimum* KL penalty (prefactor `2/n ≈ 6.7·10⁻⁵`, weaker than the additive gradient), and pinning
re-couples `λ` to the posterior. It does not fix the shape and it risks the corner. The third option is
the one that actually targets the shape: go back to the parent PAC-Bayes-kl and ask why Pinsker gave me a
square root, and whether a *different* parameter-free lower bound on the binary KL is tight in my regime.

The parent is `kl(E_Q[r]‖E_Q[R]) ≤ (KL + log(2√n/δ))/n`. Pinsker lower-bounds the binary KL by
`2(p−q)²`, a *parabola* symmetric in `p` around `q`. Inverting that parabola gives the additive `√`. But
the true binary KL is *asymmetric*: for `q < p` (empirical risk below true risk, my case), as `p` shrinks
toward `q` the true `kl` grows faster than the symmetric parabola allows. There is a sharper, asymmetric
lower bound for exactly that side — the refined Pinsker inequality `kl(q‖p) ≥ (p−q)²/(2p)`, valid for
`q < p`. Compare the prefactors: refined has `1/(2p)`, standard has `2`; refined is the tighter lower
bound on `kl` precisely when `1/(2p) > 2`, i.e. `p < 1/4`. A true risk below 25% is exactly where a
network that works lives — and I can check that this is my regime from the McAllester numbers rather than
assuming it: the certified true-risk upper bounds are all well under 0.13 (MNIST-CNN 0.0164, MNIST-FCN
0.0373, FashionMNIST-CNN 0.1002), so `p ≤ 0.10 < 0.25` with a huge margin, and at `p = 0.10` the refined
prefactor `1/(2p) = 5` is 2.5× the standard `2`. So I have been relaxing the parent with the wrong
inequality for my regime; the refined one is sharp here, and by a factor that grows as the risk shrinks —
on MNIST-CNN at `p ≈ 0.016`, `1/(2p) ≈ 31`, more than fifteen times Pinsker's `2`.

Substitute the refined bound into the parent. Let `C = (KL + log(2√n/δ))/n` be the bare parent budget.
Then `(E_Q[R] − E_Q[r])²/(2 E_Q[R]) ≤ C`, i.e. `(E_Q[R] − E_Q[r])² ≤ 2 E_Q[R] C`. The certificate case
is `E_Q[R] ≥ E_Q[r]`, so taking the root, `E_Q[R] − E_Q[r] ≤ √(2 E_Q[R] C)`, which gives
`E_Q[R] ≤ E_Q[r] + √(2 E_Q[R] C)`. And there is the catch that always shows up next: `E_Q[R]`, the thing
I am bounding, is on *both* sides, under the root on the right. This is not an explicit upper bound, so I
cannot just minimize the right side. But it is a quadratic — not in `E_Q[R]`, but in `√(E_Q[R])`. Set
`x = √(E_Q[R])`, so `E_Q[R] = x²`. Then `x² ≤ E_Q[r] + √(2C)·x`, i.e. `x² − √(2C)·x − E_Q[r] ≤ 0`. A
quadratic in `x` with positive leading coefficient is non-positive between its roots, so `x` is at most
the larger root, `x ≤ (√(2C) + √(2C + 4 E_Q[r]))/2`. Square back: pulling the `1/2` inside the roots,
`√(2C)/2 = √(C/2)` and `√(2C + 4E_Q[r])/2 = √(C/2 + E_Q[r])`, so
```
E_Q[R] ≤ ( √(E_Q[r] + C/2) + √(C/2) )².
```
with `C/2 = (KL + log(2√n/δ))/(2n)` — the *same* `kl_term` as the additive bound, the factor `2n`
re-explained (the `n` from the parent, the `2` from the refined `(p−q)²/(2p)`). Crucially, `E_Q[R]` is
gone from the right side. The quadratic-in-`√(E_Q[R])` trick simultaneously used the inequality that is
tight in my low-risk regime *and* eliminated the implicit risk, and it did so with no free parameter.
This is the fquad bound.

Before I trust it I want to run the algebra backwards on a concrete number, because the "square back"
step is exactly where a factor of two likes to hide. Take `C/2 = kl_term = 1·10⁻³` and `E_Q[r] = 0.012`
(the MNIST-CNN operating point). The larger root in `x` is `(√(2·2·10⁻³) + √(2·2·10⁻³ + 4·0.012))/2`.
Now `2C = 4·kl_term = 4·10⁻³`, so `√(2C) = 0.06325`; and `2C + 4E_Q[r] = 4·10⁻³ + 0.048 = 0.052`, so
`√(0.052) = 0.22804`. The root is `(0.06325 + 0.22804)/2 = 0.14564`, and squaring gives `E_Q[R] ≤ 0.02121`.
Cross-check against the closed form directly: `(√(0.012 + 0.001) + √(0.001))² = (√0.013 + 0.031623)² =
(0.114018 + 0.031623)² = 0.145641² = 0.021211`. The two routes agree to five digits, so the `√(C/2)`
repackaging is correct and I have not dropped or doubled the `2`. Two more sanity checks on the shape. As
`kl_term → 0` the bound is `(√(E_Q[r]) + 0)² = E_Q[r]`, so it collapses to the empirical risk with zero
complexity — correct, a zero-KL posterior should certify at its empirical risk plus only the confidence
budget. And expanding the square in general, `(√(emp+k) + √k)² = emp + 2k + 2√(k(emp+k)) ≥ emp`, so the
bound never dips below the empirical risk — it is a genuine upper relaxation, monotone increasing in both
`emp` and `k`, exactly the two-lever monotonicity I rely on at inversion time.

Now let me verify it actually attacks the wall — the square-root floor — by comparing head to head at
zero empirical risk, which is McAllester's stuck regime. At `E_Q[r] = 0`: the additive bound gives
`√(kl_term)`; fquad gives `(√(kl_term) + √(kl_term))² = (2√(kl_term))² = 4·kl_term`. So at zero empirical
risk the comparison is `4·kl_term` versus `√(kl_term)`. With `kl_term ≈ 1e-3` (the MNIST-CNN operating
point I just measured), that is `4·10⁻³` versus `3.16·10⁻²` — fquad is roughly eight times tighter. And
the advantage is not a fixed factor; it *widens* as `kl_term` shrinks, because `4k / √k = 4√k → 0`. At the
FCN's larger `kl_term` the two are closer, at the CNN's smaller `kl_term` fquad pulls further ahead —
which is the right qualitative direction, since the CNN is where the KL is smallest and the `√` floor
bites hardest. I can also see the effect at nonzero risk, on the actual bounded CE bound: at emp 0.012,
kl_term 1e-3, McAllester gives `0.012 + 0.0316 = 0.0436` while fquad gives 0.0212 — a 2× tighter training
bound at the real operating point, and the gap is entirely the shape. The structural reason is exactly the
wall diagnosis: fquad's complexity contribution is `O(kl_term)`, *linear* in the complexity, while
McAllester's is `O(√(kl_term))`. When `kl_term` is small the linear term wins decisively — this is the
fast-rate, realizable-case `1/n` behavior instead of `1/√n`, and it is the direct payoff of the refined
Pinsker being sharp below true risk 1/4. So I am not removing the stability I won; I am keeping the
parameter-free property *and* fixing the square-root floor.

I should be careful not to overclaim, though: fquad is not *universally* tighter than McAllester, only in
this regime, and I want the crossover in hand so I know I am safely on the right side of it. Setting the
two equal at `E_Q[r] = 0` gives `4·kl_term = √(kl_term)`, i.e. `kl_term = 1/16 = 0.0625`; below that fquad
wins, above it the additive bound is actually tighter. My measured `kl_term` values are `~10⁻³` (MNIST-CNN)
up to `~7·10⁻⁴` (FashionMNIST-CNN) — all roughly two orders of magnitude below the 0.0625 crossover, so I
am deep in fquad's favorable region with enormous margin, and it would take a KL of order `0.0625·2n ≈ 3750`
nats to push `kl_term` up to the crossover, a KL I am nowhere near and, given the stability fix, will not
approach. The general crossover at nonzero `emp` shifts a little, but the qualitative picture is the same:
fquad dominates whenever the complexity term is small relative to the empirical risk, which is precisely
the realizable regime the whole trajectory has been driving toward. This is reassuring rather than merely
convenient — it means the shape improvement is not a fragile knife-edge; I would have to inflate KL by more
than three orders of magnitude to lose it.

I should stress-test the regime claim on the setting where it is weakest rather than only where it is
strongest, because a bound that only helps on the easy settings is not the win I am claiming. The hardest
of the three is FashionMNIST-CNN: McAllester certifies it at 0.1002 with empirical 0-1 risk ~0.085 and KL
32 — the highest true-risk of the three, and therefore the setting closest to the `p = 1/4` crossover
above which refined Pinsker stops being the sharper inequality. So this is where fquad's advantage is
thinnest, and I want to see the margin explicitly. At `p ≈ 0.10` the refined prefactor is `1/(2p) = 5`
versus standard Pinsker's `2`, a factor of 2.5 — still a genuine improvement, but nothing like the 15×+
margin the CNN's `p ≈ 0.016` enjoyed. And FashionMNIST's `kl_term` is larger too: at KL 32,
`kl_term = (32+9.54)/60000 = 6.9·10⁻⁴`, so the head-to-head at emp ~0.085 is McAllester's
`0.085 + √(6.9·10⁻⁴) = 0.085 + 0.0263 = 0.111` against fquad's `(√(0.085+6.9·10⁻⁴) + √(6.9·10⁻⁴))² =
(0.29271 + 0.02627)² = 0.3190² = 0.1018` on the bounded CE bound — fquad still tighter, but by ~0.009
rather than the CNN's 2× factor. This is the honest shape of the win: fquad helps *most* exactly where
McAllester was most stuck (tiny risk, tiny KL, the `√` floor dominating) and helps *least* where the
empirical risk is already an appreciable fraction of the certificate. That is the right behavior — it
concentrates its improvement on the fast-rate regime — and it means my headline expectation should be a
large relative improvement on the MNIST settings and a smaller absolute one on FashionMNIST, not a uniform
shift. If I saw the reverse (big FashionMNIST gain, flat MNIST), the mechanism story would be wrong.

There is one implementation detail this rung introduces that the previous two deliberately omitted, and it
is load-bearing for fquad specifically. The earlier rungs fed the NLL surrogate *unrescaled*. fquad will
not tolerate that, and I can see why from the formula rather than by trial. The bound
`(√(E_Q[r] + C/2) + √(C/2))²` assumes `E_Q[r] ∈ [0,1]`; the clamped NLL, however, ranges up to
`log(1/pmin) = ln(10⁵) = 11.51`, more than an order of magnitude above 1. Feed that raw into the formula
and the empirical term inside the root is mis-scaled relative to the `C/2 ≈ 10⁻³` complexity term by a
factor of thousands, and the outer `(·)²` *amplifies* the miscalibration — the training objective and the
fquad formula stop upper-bounding anything the parent guarantees, and in practice the posterior, chasing a
grossly inflated empirical term, drifts far from the prior and reinflates the very KL I worked to
suppress. In the additive bound an over-1 NLL was only an additive offset that shifted the loss uniformly;
inside fquad's square it becomes a multiplicative distortion of the fit-versus-complexity balance. So for
fquad I add the bounded-loss rescaling: clamp `log_softmax` at `log(pmin)` (caps NLL at `log(1/pmin)`),
then multiply by `_loss_scale = 1/log(1/pmin) = 1/11.51 = 0.0869`, landing the surrogate in `[0,1]`. That
is not cosmetic — it is the calibration that lets the linear-in-KL fquad form actually be tighter in
practice rather than only on paper, and it is why this rung must change the surrogate even though the last
rung deliberately did not. I also clamp both arguments under the square roots at zero, defensively, so a
floating-point negative from a tiny `kl_term` does not crash `sqrt`. `compute_bound` becomes
`(√(clamp(emp + kl_term)) + √(clamp(kl_term)))²` with `kl_term = (kl + log(2√n/δ))/(2n)`, and `train_step`
feeds the rescaled bounded NLL through it.

The certificate stays separate, and for the same reason as before it stays the tighter object. I train
against fquad — convex, parameter-free, tight at low risk — but report the tightest valid bound on the
learned posterior, the PAC-Bayes-kl inversion. `compute_risk_certificate` MC-samples the empirical 0-1
risk via `compute_01_risk`, reads the KL, forms the *bare* parent budget `c = (KL + log(2√n/δ))/n` — note
`/n`, not `/(2n)`: the `2` lived only in the fquad relaxation, never in the inversion — and returns
`inv_kl(emp_risk_01, c)`. It is worth being clear that the fquad shape improvement acts on the certificate
*indirectly*: the reported number is still the parent inversion, identical machinery to the two prior
rungs, so fquad cannot change it except by changing what the training run *lands at* — a smaller KL and a
low empirical 0-1 risk. The tighter training objective is a better search direction for the posterior, not
a different readout. As with the previous rungs I keep a single, uncorrected inversion (no inner
Monte-Carlo correction for posterior-sampling error), matching the scaffold style; the empirical NLL fed
into `ce_bound` uses the same `_loss_scale` rescaling as training so the two are consistent — and this is
also why I expect the `ce_bound` column, which *does* wear the fquad shape directly, to move far more
dramatically than the `risk_certificate` column, which only inherits the shape through the learned KL.

Here is my falsifiable expectation against McAllester's numbers. The KL should stay small and stable —
fquad is parameter-free, so I keep the stability win, and the bounded-loss rescaling should if anything
let KL drop *further* than McAllester's because the objective is no longer over-weighting an unrescaled
empirical term against the complexity. So I predict the KL collapses by roughly another order of
magnitude — from McAllester's 81 on MNIST-FCN down into the single digits, and from its already-small 11
and 32 on the two CNNs down to low single digits. The certificate should improve modestly but unmistakably,
because at these small KLs the *shape* change from `√(kl_term)` to `4·kl_term` is exactly what reduces the
floor: each of McAllester's certificates (0.0373, 0.0164, 0.1002) should tick down, most visibly on the
MNIST settings where the `√` floor was proportionally largest and least on FashionMNIST where the empirical
risk already dominates. The improvement is bounded from above by the floor arithmetic I did earlier — recall
that even at KL 0 the MNIST-CNN certificate floor is ~0.0150, so I cannot expect to beat that; getting close
to it means crushing the residual KL down against the irreducible confidence budget, which is precisely what
a smaller, rescaled-objective KL should do. The empirical 0-1 risk may tick up slightly (fquad with rescaled loss
fits a touch less hard than the unrescaled additive run), but the linear-in-KL shape more than
compensates. The signature that confirms the mechanism: `kl_divergence` should drop dramatically *and*
`ce_bound` should fall by a large factor — McAllester's MNIST-CNN ce_bound 0.0805 → fquad well under 0.01,
close to an order of magnitude, because the CE bound wears the shape change directly whereas the reported
certificate only inherits it through the KL. What would falsify me: if fquad's certificate matches or
exceeds McAllester's, then either the rescaling destabilized the posterior — I would see it as a KL that
fails to drop or even grows — or the low-risk regime is not actually below the 1/4 crossover where refined
Pinsker helps. But the measured true risks are all well under 0.13, so I am confident this is the regime
fquad is built for. This is the strongest rung I can build on this edit surface: the parent PAC-Bayes-kl
is the tightest standard bound, fquad is its sharpest parameter-free convex relaxation at low risk, and the
certificate is already the kl-inversion of the parent itself. The full scaffold module — the rescaled fquad
`compute_bound`, `train_step`, and the single-`inv_kl` `compute_risk_certificate` — is in the answer.
