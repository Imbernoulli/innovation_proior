McAllester did exactly what I bet it would, and the KL numbers are the proof. Removing the `λ` knob
collapsed and stabilized the KL: on MNIST-FCN the mean fell from Catoni's 539 to 81, and — the part I
care about most — it is now *tight across seeds*, 71.90 / 80.28 / 91.83, no bimodal blowup. On MNIST-CNN
the KL fell from 192 to 11 and is rock-stable (10.0 / 12.1 / 12.1); FashionMNIST-CNN from 286 to 32. The
certificate followed: MNIST-FCN mean 0.0558 → 0.0373, MNIST-CNN 0.0250 → 0.0164, FashionMNIST-CNN
0.1215 → 0.1002. So the parameter-free additive bound is the right *stability* fix — the diagnosis that
the runaway KL was a `λ`-mechanism artifact, not a weak-penalty problem, was correct. But now look at
where the additive bound is *stuck*, because that is the next wall. On MNIST-CNN the empirical 0-1 risk
is 0.0119 and the KL is only 11 nats over `n ≈ 30000` — the posterior is essentially perfect and barely
moved from the prior — yet the certificate sits at 0.0164. The gap between the certificate and the
empirical risk is ~0.005, and it is *not* coming from a large KL anymore. It is coming from the shape of
the bound. With KL this small, `kl_term = (11 + log(2√n/δ))/(2n)` is around `1e-3`, and the additive
bound adds `√(1e-3) ≈ 0.032` on top of the empirical risk — except the empirical risk is already near
zero, so the certificate is essentially `√(kl_term)` itself. The square root is the floor. McAllester
fixed the KL; the residual looseness is now entirely the `√`.

So the target is precise: I have a small, stable KL, and I am being penalized by the square-root shape of
the relaxation rather than by the complexity itself. I need a bound whose complexity contribution is
*linear* in `kl_term` at small empirical risk, not the square root — without reintroducing a free `λ`,
because the Catoni run showed me what a free knob costs. Let me go back to the parent PAC-Bayes-kl and
ask why Pinsker gave me a square root in the first place, and whether there is a relaxation that keeps
the parameter-free property but is tight at low risk.

The parent is `kl(E_Q[r]‖E_Q[R]) ≤ (KL + log(2√n/δ))/n`. Pinsker lower-bounds the binary KL by
`2(p−q)²`, a *parabola* symmetric in `p` around `q`. Inverting that parabola gives the additive `√`. But
the true binary KL is *asymmetric*: for `q < p` (empirical risk below true risk, my case), as `p` shrinks
toward `q` the true `kl` grows faster than the symmetric parabola allows. There is a sharper, asymmetric
lower bound for exactly that side — the refined Pinsker inequality `kl(q‖p) ≥ (p−q)²/(2p)`, valid for
`q < p`. Compare the prefactors: refined has `1/(2p)`, standard has `2`; refined is the tighter lower
bound on `kl` precisely when `1/(2p) > 2`, i.e. `p < 1/4`. A true risk below 25% is exactly where a
network that works lives — McAllester's certificates are all well under 0.13. So I have been relaxing the
parent with the wrong inequality for my regime; the refined one is sharp here.

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

Now let me verify it actually attacks the wall — the square-root floor — by comparing head to head at
zero empirical risk, which is McAllester's stuck regime. At `E_Q[r] = 0`: the additive bound gives
`√(kl_term)`; fquad gives `(√(kl_term) + √(kl_term))² = (2√(kl_term))² = 4·kl_term`. So at zero empirical
risk the comparison is `4·kl_term` versus `√(kl_term)`. With `kl_term ≈ 1e-3` (the MNIST-CNN operating
point I just measured), that is `4e-3` versus `3.2e-2` — fquad is roughly eight times tighter. The
structural reason is exactly the wall diagnosis: fquad's complexity contribution is `O(kl_term)`, *linear*
in the complexity, while McAllester's is `O(√(kl_term))`. When `kl_term` is small the linear term wins
decisively — this is the fast-rate, realizable-case `1/n` behavior instead of `1/√n`, and it is the direct
payoff of the refined Pinsker being sharp below true risk 1/4. So I am not removing the stability I won;
I am keeping the parameter-free property *and* fixing the square-root floor.

There is one implementation detail this rung introduces that the previous two deliberately omitted, and it
is load-bearing for fquad specifically. The earlier rungs fed the NLL surrogate *unrescaled*. fquad will
not tolerate that. The bound `(√(E_Q[r] + C/2) + √(C/2))²` assumes `E_Q[r] ∈ [0,1]`; if I feed it an
unbounded NLL that exceeds 1, the empirical term inside the root is mis-scaled relative to the `C/2`
complexity term, and the `(·)²` outer square *amplifies* the miscalibration — the training objective and
the fquad formula stop upper-bounding anything, and in practice the posterior drifts far from the prior,
inflating the very KL I worked to suppress. So for fquad I add the bounded-loss rescaling: clamp
`log_softmax` at `log(pmin)` (caps NLL at `log(1/pmin)`), then multiply by `_loss_scale = 1/log(1/pmin)`,
landing the surrogate in `[0,1]`. This is not cosmetic — it is the calibration that lets the linear-in-KL
fquad form actually be tighter in practice rather than just on paper. I also clamp both arguments under
the square roots at zero, defensively, so a floating-point negative does not crash `sqrt`. `compute_bound`
becomes `(√(clamp(emp + kl_term)) + √(clamp(kl_term)))²` with `kl_term = (kl + log(2√n/δ))/(2n)`, and
`train_step` feeds the rescaled bounded NLL through it.

The certificate stays separate. I train against fquad — convex, parameter-free, tight at low risk — but
report the tightest valid bound on the learned posterior, the PAC-Bayes-kl inversion. `compute_risk_
certificate` MC-samples the empirical 0-1 risk via `compute_01_risk`, reads the KL, forms the *bare*
parent budget `c = (KL + log(2√n/δ))/n` — note `/n`, not `/(2n)`: the `2` lived only in the fquad
relaxation, never in the inversion — and returns `inv_kl(emp_risk_01, c)`. As with the previous rungs I
keep a single, uncorrected inversion (no inner Monte-Carlo correction), matching the scaffold style; the
empirical NLL fed into `ce_bound` uses the same `_loss_scale` rescaling as training so the two are
consistent.

Here is my falsifiable expectation against McAllester's numbers. The KL should stay small and stable —
fquad is parameter-free, so I keep the stability win, and the bounded-loss rescaling should if anything
let KL drop *further* than McAllester's because the objective is no longer over-weighting an unrescaled
empirical term. So I predict the KL collapses by another order of magnitude: on MNIST-FCN from
McAllester's 81 toward single digits (~4), on MNIST-CNN from 11 toward ~2, on FashionMNIST-CNN from 32
toward ~3. The certificate should improve modestly but unmistakably, because at these small KLs the
*shape* change from `√(kl_term)` to `4·kl_term` is exactly what reduces the floor: MNIST-FCN
0.0373 → low 0.03s (~0.031), MNIST-CNN 0.0164 → mid 0.015s, FashionMNIST-CNN 0.1002 → ~0.097. The
empirical 0-1 risk may tick up slightly (fquad with rescaled loss fits a touch less hard than the
unrescaled additive run), but the linear-in-KL shape more than compensates. The signature that confirms
the mechanism: `kl_divergence` should drop dramatically *and* `ce_bound` should fall by a large factor
(McAllester's MNIST-CNN ce_bound 0.081 → fquad ~0.008, an order of magnitude, because the CE bound is
where the shape change is most visible). What would falsify me: if fquad's certificate matches or exceeds
McAllester's, then either the rescaling destabilized the posterior or the low-risk regime is not actually
below the 1/4 crossover where refined Pinsker helps — but the measured true risks are all well under 0.13,
so I am confident this is the regime fquad is built for. This is the strongest rung I can build on this
edit surface: the parent PAC-Bayes-kl is the tightest standard bound, fquad is its sharpest
parameter-free convex relaxation at low risk, and the certificate is already the kl-inversion of the
parent itself. The full scaffold module — the rescaled fquad `compute_bound`, `train_step`, and the
single-`inv_kl` `compute_risk_certificate` — is in the answer.
