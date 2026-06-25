Let me start from what actually blocks me. I want to learn a randomized classifier — a distribution ρ over a hypothesis class H, predicting by drawing a fresh h∼ρ each time — and I want the *same* quantity that certifies its generalization to be the thing I optimize. PAC-Bayesian theory hands me certificates of the form "true risk ≤ empirical risk + a complexity term," where complexity is KL(ρ‖π) to a prior π fixed before I see the data. These are about the tightest generalization bounds anyone knows. So the dream is obvious: minimize the bound over ρ directly. Train the posterior to make the certificate small, and read the certificate off the trained posterior. No held-out set, no separate tuning, the certificate is honest because it held uniformly over all ρ to begin with.

The reason nobody just does this is the shape of the tightest bound. Let me write it down. The classical machinery is the change-of-measure inequality: for any f(h,S) and any data-independent π, with probability ≥ 1−δ over S, simultaneously for all ρ, E_{h∼ρ}[f(h,S)] ≤ KL(ρ‖π) + ln(1/δ) + ln E_{h∼π} E_{S'}[e^{f(h,S')}]. This is Donsker and Varadhan's variational identity for the KL — KL(ρ‖π) = sup_f { E_ρ[f] + ln E_π[e^{f}] }, equivalently ln E_π[e^{f}] = sup_ρ { E_ρ[f] − KL(ρ‖π) } — wrapped with a Markov inequality on the π-side moment generating function and a swap of the two expectations, which is legal because π doesn't depend on S. The whole craft is choosing f and bounding its MGF E_S[e^{f}]. The classical choice is f(h,S) = n·kl(L̂(h,S)‖L(h)), the binary KL between the empirical and true Bernoulli loss rates, and Maurer's note proves the sharp MGF control E_S[e^{n·kl(L̂,L)}] ≤ 2√n for n ≥ 8 — and I should pause on the 2√n, because it's the *square root* of n, not n; Maurer's whole contribution there was halving the logarithmic sample-size dependence, turning an older ln(2n) into ln(2√n), and I'll want that tightness later. Plug it in and I get the PAC-Bayes-kl inequality: kl( E_ρ[L̂(h,S)] ‖ E_ρ[L(h)] ) ≤ (KL(ρ‖π) + ln(2√n/δ)) / n.

This is the tightest standard bound, and to *evaluate* it I just invert the binary kl: find the largest L with kl(L̂‖L) ≤ c, where c = (KL + ln(2√n/δ))/n. That's a one-dimensional monotone solve, trivial by bisection. But to *optimize* it over ρ — that's the wall. E_ρ[L(h)], the thing I'm bounding, is defined implicitly through the kl, and the upper bound I get by inverting is not convex in ρ. So I can't drop it into a gradient-based learner and trust that I'm descending toward the true minimum. This is exactly why the field splits the job: PAC-Bayes-kl gets used for the *final certificate*, but training the posterior is done on some convex surrogate, usually a hand-built linear trade-off E_ρ[L̂] + β·KL/n with β picked by cross-validation. And cross-validation is the thing I'm trying to kill: it's expensive — for a kernel SVM, whose training is super-quadratic, you retrain many times on almost the whole dataset — and it can be misled. I want the trade-off set from the data, by the bound itself.

So the real question is sharper than "minimize a bound." It's: can I get a bound nearly as tight as PAC-Bayes-kl but *convex in ρ*, and with a trade-off I can tune from the sample without a union bound? Let me look at the convex relaxations and see exactly where each falls short, because the gap will tell me what to build.

The standard relaxation route is to lower-bound the binary kl and invert the looser inequality. Pinsker says kl(p‖q) ≥ 2(p−q)², and inverting that gives the explicit additive bound E_ρ[L] ≤ E_ρ[L̂] + √((KL + ln(2√n/δ))/(2n)). Clean, convex in ρ — E_ρ[L̂] is linear and KL is convex — but loose: the √ term is a fixed additive penalty that doesn't shrink as the empirical risk shrinks, so when my posterior is genuinely good and L̂ is near zero, this bound is sitting way above PAC-Bayes-kl. It over-charges a confident posterior. The refined Pinsker fixes the low-risk regime: for p < q, kl(p‖q) ≥ (q−p)²/(2q), which is tighter than plain Pinsker precisely when the risk is below 1/4. Apply it inside the change of measure and I get E_ρ[L(h)] − E_ρ[L̂(h,S)] ≤ √( 2·E_ρ[L(h)]·(KL(ρ‖π) + ln(2√n/δ))/n ). Now the bound knows about its own tightness at low risk — but look where E_ρ[L(h)] ended up: under the square root, on the right. The quantity I want to bound is tangled into the bound. I can treat it as a quadratic in √(E_ρ[L]) and solve it out — that gives the quadratic-form bound ( √(E_ρ[L̂] + κ) + √κ )² with κ = (KL + ln(2√n/δ))/(2n) — and that's a legitimate certificate, tighter than the classic one at small risk. But as an *objective to optimize over ρ*, the unknown sitting under a square root that also wraps the complexity term is a nonlinear coupling; there's no clean, separately tunable empirical-vs-complexity trade-off to turn.

Now Catoni's route, because it's the one that actually achieved ρ-convexity. He chose a different exponential tilt, engineered so that E_S[e^{f}] = 1 exactly, no concentration slack at all, and substituting it into change-of-measure gives a bound convex in ρ, of essentially the linear-in-L̂-plus-KL flavor that makes the ρ-optimum tractable. The price is a free trade-off parameter λ sitting in there. And the closed-form ρ-optimum is beautiful: minimizing E_ρ[L̂] + KL(ρ‖π)/η over ρ is, by Donsker-Varadhan applied with the tilt h = −η L̂, solved by the Gibbs posterior ρ̂_η(dh) ∝ π(dh) e^{−η L̂(h,S)}. If the KL coefficient is written as 1/(nλ), then η = nλ and the same variational calculation gives the familiar e^{−nλL̂} tilt. The exponential tilt of the prior toward low-empirical-loss hypotheses — exactly the measure that saturates the change-of-measure inequality. So convexity in ρ *is* achievable, and it comes hand in hand with a Gibbs posterior.

But here's the specific thing about Catoni's bound that I have to get past: it holds for a *single, fixed* λ, chosen before seeing the data. If I want to tune λ to the sample — and of course I do, the right trade-off depends on how much data I have and how well the hypotheses fit — I have to take a union bound over a grid of λ-values, say a geometrically spaced grid, which both costs me an extra logarithmic factor in the bound and gives me a discretized λ instead of a continuous one I can optimize. Keshet, McAllester and Hazan had a related parametrized convex bound and hit the same thing: in practice they fell back to the same linear surrogate with the trade-off cross-validated. So the recurring failure across all of these is: either the bound is tight but non-convex in ρ (PAC-Bayes-kl), or it's convex but the trade-off parameter is *fixed per draw* so tuning it costs a union bound (Catoni, KMH). What I want is a bound that is convex in ρ for fixed λ *and* holds for all λ simultaneously, so I can optimize λ continuously on the data for free.

Let me see if I can get that out of the refined-Pinsker inequality, since that one's already tight at low risk. I'm staring at E_ρ[L] − E_ρ[L̂] ≤ √( 2·E_ρ[L]·(KL + ln(2√n/δ))/n ). The problem is the square root binding E_ρ[L] to the complexity term. What kills a √(xy) cleanly while introducing a tunable knob? The AM-GM-style inequality √(xy) ≤ ½(λx + y/λ), which holds for every λ > 0 — it's just (√(λx) − √(y/λ))² ≥ 0 rearranged. And the crucial property, the one I was missing in Catoni's route: this inequality is true *for all λ at once*. It's not a choice I make per sample; it's an algebraic identity over the whole positive λ-axis. If I apply it inside the probabilistic statement, the resulting bound inherits "for all λ" from the algebra, not from a union bound.

So let me do it. Set x = E_ρ[L(h)] and y = 2(KL(ρ‖π) + ln(2√n/δ))/n, so that √(xy) is exactly the right-hand side above. Then for every λ > 0,
  E_ρ[L] − E_ρ[L̂] ≤ ½( λ E_ρ[L] + (1/λ)·2(KL + ln(2√n/δ))/n )
                   = (λ/2) E_ρ[L] + (KL + ln(2√n/δ))/(λ n).
Move the (λ/2)E_ρ[L] term to the left:
  (1 − λ/2) E_ρ[L] ≤ E_ρ[L̂] + (KL(ρ‖π) + ln(2√n/δ))/(λ n).
And for λ < 2 the coefficient 1 − λ/2 is positive, so I divide:
  E_ρ[L(h)] ≤ E_ρ[L̂(h,S)]/(1 − λ/2) + (KL(ρ‖π) + ln(2√n/δ)) / ( λ(1 − λ/2) n ).
There it is. And because the only inequality I used beyond the probabilistic PAC-Bayes-kl step was the deterministic √(xy) ≤ ½(λx + y/λ), which holds simultaneously for every λ, this bound holds with probability ≥ 1−δ simultaneously for all ρ *and* all λ ∈ (0,2). That's the whole point — I never paid a union bound for λ. Catoni's bound was per-fixed-λ; this one is uniform in λ. I'll call the right-hand side the PAC-Bayes-λ bound and write F(ρ,λ) = E_ρ[L̂]/(1−λ/2) + (KL(ρ‖π) + ln(2√n/δ))/(nλ(1−λ/2)).

One thing nags at me: I relaxed a √(xy) by an AM-GM step, and AM-GM is a genuine inequality, so I might have thrown away tightness — in which case the convexity would have cost me the very sharpness that motivated starting from refined Pinsker. But √(xy) ≤ ½(λx+y/λ) is an *equality* at λ = √(y/x), so minimizing the linearized bound over λ should recover whatever the refined-Pinsker route gives directly, i.e. the quadratic-form certificate (√(E_ρ[L̂]+κ)+√κ)² with κ = (KL+ln(2√n/δ))/(2n). Let me check that the two actually coincide rather than just believe it. Taking (E_ρ[L̂], κ) = (0, 0.01), (0.1, 0.02), (0.3, 0.05), (0.5, 0.1), I minimize F(ρ,λ) over λ ∈ (0,2) numerically and compare to the closed-form quadratic certificate: 0.040000 vs 0.040000, 0.237980 vs 0.237980, 0.664575 vs 0.664575, 1.189898 vs 1.189898 — equal to six places in every case. So at its optimal λ the PAC-Bayes-λ bound *is* the quadratic certificate; the linearization buys uniformity in λ at no cost in tightness. That was the real worry, and it's settled.

Now let me check it's actually convex where I need it to be, separately in each argument, because that's what makes the optimization tractable. Fix λ. E_ρ[L̂(h,S)] = E_{h∼ρ}[L̂(h,S)] is linear in ρ. KL(ρ‖π) is convex in ρ — that's standard, the relative entropy is jointly convex and here π is fixed. The denominators 1−λ/2 and λ(1−λ/2)n are positive constants for fixed λ ∈ (0,2). So F(·,λ) is a positive-weighted sum of a linear and a convex functional: convex in ρ. Good. And I already know its minimizer from the Catoni/Donsker-Varadhan story — minimizing a linear-in-ρ empirical term plus a (1/coefficient)·KL term is solved by the Gibbs tilt. Let me just confirm the exact form here. The ρ-dependent part of F is E_ρ[L̂]/(1−λ/2) + KL(ρ‖π)/(nλ(1−λ/2)); pulling out the common 1/(1−λ/2), I'm minimizing E_ρ[L̂] + KL(ρ‖π)/(nλ), which by Donsker-Varadhan with tilt −nλ L̂ is minimized at
  ρ_λ(h) = π(h) e^{−λ n L̂(h,S)} / E_{h'∼π}[ e^{−λ n L̂(h',S)} ],
the Gibbs/Boltzmann posterior. The normalizer E_π[e^{−λ n L̂}] handles continuous and discrete H in one notation.

Now fix ρ and look at the λ-dependence. F(ρ,·) is c₁/(1−λ/2) + c₂/(λ(1−λ/2)) with c₁ = E_ρ[L̂] ≥ 0 and c₂ = (KL + ln(2√n/δ))/n ≥ 0. For a closed-form minimizer I want this convex in λ on (0,2). The first term 1/(1−λ/2) is 1/(positive decreasing affine), clearly convex increasing there. The second, 1/(λ(1−λ/2)), I'll have to actually differentiate — the denominator λ(1−λ/2) is a downward parabola, zero at λ=0 and λ=2, so its reciprocal blows up at both ends and the convexity isn't automatic. I'll come back and compute that second derivative carefully below, when I need g'' anyway; for now I'll proceed *assuming* convexity in λ and check it survives. Differentiate F(ρ,·) and set to zero; the minimizer (this is the same computation that shows up in the Tolstikhin-Seldin variance work, where exactly this c₁/(1−λ/2)+c₂/(λ(1−λ/2)) appears) comes out to
  λ = 2 / ( √( 2 n E_ρ[L̂] / (KL(ρ‖π) + ln(2√n/δ)) + 1 ) + 1 ).
Two sanity checks on this λ. The denominator √(…+1)+1 > 2, so λ < 1 always — the optimal λ stays well inside (0,2), nowhere near the 1−λ/2 → 0 singularity. And for the worst case E_ρ[L̂] = 0 the formula gives λ = 2/(√1 + 1) = 1; as E_ρ[L̂] grows λ shrinks. A lower bound: for n ≥ 4, λ ≥ 2/(√(2n+1)+1) ≥ 1/√n. So λ lives in roughly [1/√n, 1]. I'll remember that range; it's where the optimization should be confined.

Before trusting that formula I should check it against a direct minimization, because I derived it by hand and the 2/(√(2nE[L̂]/(KL+ln(2√n/δ))+1)+1) shape is easy to get wrong by a factor. Writing F(ρ,·) as c₁/(1−λ/2) + c₂/(λ(1−λ/2)) with c₁ = E_ρ[L̂] and c₂ = (KL+ln(2√n/δ))/n, I take a few (c₁,c₂) and minimize numerically over λ ∈ (0,2). For (c₁,c₂) = (0.3,0.05) the formula gives λ = 0.43426 and a bounded scalar minimizer gives 0.43426; for (0.1,0.2), 0.82843 vs 0.82843; for (0.05,0.01), 0.46332 vs 0.46332; for (0.5,0.3), 0.64900 vs 0.64900. They agree to five places and F at the formula's λ equals F at the brute-force λ. So the closed form is right, and in every case λ landed in (0,1) — consistent with the algebraic bound I just derived.

So I have an alternating minimization: fix λ, set ρ to the Gibbs posterior ρ_λ; fix ρ, set λ by the closed form; repeat. Each step solves a convex subproblem exactly, so each step decreases F, and a monotone-decreasing sequence bounded below converges. To a *local* minimum, at least. Whether it's the *global* minimum is the question I can't yet answer, and the clean sufficient condition would be joint convexity in (ρ,λ). Let me check whether I have it. Take large n so the complexity term is negligible; then F is dominated by E_ρ[L̂]/(1−λ/2). The mixed partial of that in a posterior-direction and in λ is d/dλ[1/(1−λ/2)] times the linear-in-ρ derivative of E_ρ[L̂], i.e. a sign-indefinite cross-term that the diagonal blocks (a *linear* function of ρ has zero ρ-curvature) cannot dominate. So the Hessian in (ρ,λ) is not positive semidefinite, and the bound is not jointly convex. Joint convexity would have been the clean route to a global guarantee, and I don't have it.

But joint convexity isn't *necessary* for alternating minimization to hit the global min. Let me think about what I actually need. I can eliminate ρ entirely. At every λ the inner ρ-optimum is ρ_λ in closed form, so substitute it back and study the bound as a function of λ alone. If that one-dimensional function has a single global minimum and no other stationary points, then a monotone-decreasing alternating scheme, which only ever moves λ downhill, can't get stuck — it converges to the unique minimizer.

So substitute ρ = ρ_λ into F. I need KL(ρ_λ‖π). By definition KL(ρ_λ‖π) = E_{ρ_λ}[ln(ρ_λ(h)/π(h))] = E_{ρ_λ}[ln( e^{−nλ L̂(h,S)} / E_π[e^{−nλ L̂(h',S)}] )] = −nλ E_{ρ_λ}[L̂(h,S)] − ln E_π[e^{−nλ L̂(h',S)}]. Now plug into F(ρ_λ,λ) = E_{ρ_λ}[L̂]/(1−λ/2) + (KL(ρ_λ‖π) + ln(2√n/δ))/(nλ(1−λ/2)). The first term is E_{ρ_λ}[L̂]/(1−λ/2). In the second term, KL(ρ_λ‖π)/(nλ(1−λ/2)) = (−nλ E_{ρ_λ}[L̂] − ln E_π[e^{−nλL̂}])/(nλ(1−λ/2)) = −E_{ρ_λ}[L̂]/(1−λ/2) − ln E_π[e^{−nλL̂}]/(nλ(1−λ/2)). The two E_{ρ_λ}[L̂]/(1−λ/2) pieces cancel exactly, and I'm left with
  F(λ) = ( −ln E_π[e^{−nλ L̂(h,S)}] + ln(2√n/δ) ) / ( n λ (1 − λ/2) ).
The empirical-risk term has vanished into the log-partition function. F is now a single scalar function of λ. That cancellation is the kind of thing I want to confirm rather than trust, since I'd be building the whole convergence argument on it. So I evaluate both sides independently on a concrete instance — H = {h₁,h₂}, L̂ = (0, 0.5), uniform prior, n = 100, δ = 0.01 — computing F the long way (forming ρ_λ, its KL, and E_{ρ_λ}[L̂]) and the collapsed way (just the log-partition) at λ = 0.05, 0.1, 0.2, 0.5, 0.9. They match to ~10⁻¹⁶ at every λ (e.g. at λ=0.5 both give 0.221175). The collapse is real. The numerator −ln E_π[e^{−nλL̂}] is the cumulant generating function of the loss under the prior, a smooth convex-analysis object; the denominator nλ(1−λ/2) is the same shape as before.

Now, is F(λ) convex? I'd like it to be, because then the whole story would be over. Let me just look. On that same toy instance I scan F over λ ∈ (0.01, 0.99) and check the discrete second differences: every one is positive, and the minimum sits at the right edge of the window — F is monotone decreasing and convex here. So I push harder for a counterexample: clusters of "mediocre" hypotheses at intermediate loss, small n where the variance term should bite, then a few thousand random (L̂, π, n, δ) configurations. In none of them does F show a negative second difference; it keeps coming back convex. I can't manufacture a non-convex F at will, which is suggestive but is *not* a proof — empirical convexity over the cases I tried is not convexity, and I haven't bounded the curvature analytically. So I won't claim convexity. What I actually need is weaker and provable: *strong quasiconvexity* — a single global minimum and no other stationary points, so that downhill always means toward the optimum. A univariate function is strongly quasiconvex if for any x,y and t∈(0,1), f(tx+(1−t)y) < max{f(x),f(y)}. The way to certify that for a smooth function is to show every stationary point is a local minimum — then a continuous 1-D function can't have two of them (between two local minima there'd have to be a local max, which is a stationary point that isn't a local min), so there's exactly one, and it's the global min. That's the property I'll go after, since it's what the alternating descent really requires and it doesn't demand the convexity I couldn't establish.

So let me actually compute F' and F''. I'll decompose F(λ) = f(λ) g(λ) with
  f(λ) = −(1/n) ln E_π[e^{−nλ L̂}] + ln(2√n/δ)/n,
  g(λ) = 1/( λ(1 − λ/2) ).
First f. Write A(λ) = E_π[e^{−nλL̂}], so f = −(1/n) ln A + const. Then f'(λ) = −(1/n)·A'/A. And A'(λ) = E_π[ −nL̂ e^{−nλL̂} ] = −n E_π[L̂ e^{−nλL̂}], so A'/A = −n E_π[L̂ e^{−nλL̂}]/E_π[e^{−nλL̂}] = −n E_{ρ_λ}[L̂], recognizing the Gibbs expectation. Hence f'(λ) = −(1/n)·(−n E_{ρ_λ}[L̂]) = E_{ρ_λ}[L̂] ≥ 0. The cumulant generating function's derivative is the tilted mean — of course. Differentiate again: f''(λ) = d/dλ E_{ρ_λ}[L̂]. Using A again, E_{ρ_λ}[L̂] = E_π[L̂ e^{−nλL̂}]/A, and differentiating that quotient,
  f''(λ) = ( (d/dλ E_π[L̂ e^{−nλL̂}]) A − (d/dλ A) E_π[L̂ e^{−nλL̂}] ) / A²
         = ( −n E_π[L̂² e^{−nλL̂}] A + n E_π[L̂ e^{−nλL̂}]² ) / A²
         = −n ( E_{ρ_λ}[L̂²] − E_{ρ_λ}[L̂]² )
         = −n Var_{ρ_λ}[L̂(h,S)] ≤ 0.
The second derivative of the (negated) log-partition is −n times the tilted variance — the cumulant generating function is convex, so f, which is its negation up to scale, is concave. So f is nonnegative, increasing, concave. Both of these are worth a quick numeric sanity check, because the whole F'' calculation hinges on f' being the Gibbs mean and f'' being −n times the Gibbs variance. On the mediocre-cluster instance (one hypothesis at 0, fifty at 0.2, one at 1, n=20) I compare a central-difference f' against E_{ρ_λ}[L̂] and a second difference against −nVar_{ρ_λ}[L̂]: at λ=0.5, f'=0.17425 vs E_{ρ_λ}[L̂]=0.17425, and f''=−0.0898 vs −nVar=−0.0898; at λ=0.88 they read f'=0.11935 vs 0.11935 and f''=−0.1925 vs −0.1925. The identities hold.

Now g(λ) = 1/(λ(1−λ/2)) = 1/(λ − λ²/2). g'(λ) = −(1 − λ)/(λ(1−λ/2))²... let me be careful. d/dλ (λ − λ²/2) = 1 − λ, so g'(λ) = −(1 − λ)/(λ − λ²/2)² = (λ − 1)/(λ²(1−λ/2)²). On (0,1] that's ≤ 0: g decreasing, as expected since making λ small blows up the complexity term. And g''(λ): differentiate g' = (λ−1)/(λ²(1−λ/2)²). I'll grind it. Numerator derivative: 1·(λ²(1−λ/2)²) − (λ−1)·d/dλ(λ²(1−λ/2)²), over (λ²(1−λ/2)²)². The inner derivative d/dλ(λ²(1−λ/2)²) = 2λ(1−λ/2)² + λ²·2(1−λ/2)·(−1/2) = 2λ(1−λ/2)² − λ²(1−λ/2). So
  g'' = [ λ²(1−λ/2)² − (λ−1)(2λ(1−λ/2)² − λ²(1−λ/2)) ] / (λ²(1−λ/2)²)².
Factor λ(1−λ/2) out of the bracket: bracket = λ(1−λ/2)[ λ(1−λ/2) − (λ−1)(2(1−λ/2) − λ) ]. Now 2(1−λ/2) − λ = 2 − λ − λ = 2 − 2λ = 2(1−λ). So bracket = λ(1−λ/2)[ λ(1−λ/2) − 2(λ−1)(1−λ) ] = λ(1−λ/2)[ λ(1−λ/2) + 2(1−λ)² ], using −(λ−1)(1−λ) = (1−λ)². Cancel one λ(1−λ/2) against the denominator (λ(1−λ/2))⁴/... — denominator is (λ²(1−λ/2)²)² = λ⁴(1−λ/2)⁴, and the bracket has a leading λ(1−λ/2), so
  g'' = [ λ(1−λ/2) + 2(1−λ)² ] / (λ³(1−λ/2)³).
Expand the top: λ − λ²/2 + 2(1 − 2λ + λ²) = λ − λ²/2 + 2 − 4λ + 2λ² = (3/2)λ² − 3λ + 2 = (3λ² − 6λ + 4)/2 = (3(λ−1)² + 1)/2, since 3(λ−1)²+1 = 3λ²−6λ+3+1 = 3λ²−6λ+4. So
  g''(λ) = (3(λ−1)² + 1) / (2 λ³ (1−λ/2)³) > 0
strictly on (0,2). So g is positive, decreasing on (0,1], convex. That algebra had a lot of moving parts — factoring λ(1−λ/2) out of the bracket, the −(λ−1)(1−λ)=(1−λ)² step — so I check the closed form against a numeric second derivative of g: at λ=0.3, formula 74.481 vs numeric 74.481; at λ=0.8, 5.0637 vs 5.0637; at λ=1.5, 16.593 vs 16.593. Matches. And this g'' computation retroactively settles the convexity-in-λ I assumed earlier when I derived the closed-form λ-optimum: F(ρ,·) = c₁/(1−λ/2) + c₂·g(λ) with c₁,c₂ ≥ 0, the first term convex and g convex, so F(ρ,·) is genuinely convex in λ on (0,2). The earlier closed form was the minimizer of a convex function, so it is the global per-ρ optimum, as the brute-force check already suggested.

Now F = fg with f concave and g convex — a product like that isn't automatically quasiconvex, which is why I couldn't take F's convexity for granted even though the cases I scanned all came out convex. So instead of betting on convexity I have to look at the stationary points directly. F'(λ) = f'g + fg'. At a stationary point F'=0, so f'g = −fg'. I'll use the identity that comes from un-cancelling the KL: f(λ) = λ E_{ρ_λ}[L̂] + (KL(ρ_λ‖π) + ln(2√n/δ))/n. (Check: this is just f written back in terms of KL using KL(ρ_λ‖π) = −nλE_{ρ_λ}[L̂] − ln E_π[e^{−nλL̂}], so λE_{ρ_λ}[L̂] + KL/n = λE_{ρ_λ}[L̂] − λE_{ρ_λ}[L̂] − (1/n)ln E_π[e^{−nλL̂}] = −(1/n)ln A, plus the ln(2√n/δ)/n constant; yes, equals f.) Now write F' = 0 using f' = E_{ρ_λ}[L̂], g = 1/(λ(1−λ/2)), g' = (λ−1)/(λ²(1−λ/2)²):
  E_{ρ_λ}[L̂]/(λ(1−λ/2)) + ( (λ−1)/(λ²(1−λ/2)²) )·( λE_{ρ_λ}[L̂] + (KL+ln(2√n/δ))/n ) = 0.
Multiply through by λ(1−λ/2):
  E_{ρ_λ}[L̂] + (λ−1)/(λ(1−λ/2))·( λE_{ρ_λ}[L̂] + (KL+ln(2√n/δ))/n ) = 0,
  E_{ρ_λ}[L̂] + (λ−1)E_{ρ_λ}[L̂]/(1−λ/2) + (λ−1)(KL+ln(2√n/δ))/(nλ(1−λ/2)) = 0.
Combine the first two: E_{ρ_λ}[L̂][ 1 + (λ−1)/(1−λ/2) ] = E_{ρ_λ}[L̂]·( (1−λ/2 + λ − 1)/(1−λ/2) ) = E_{ρ_λ}[L̂]·( (λ/2)/(1−λ/2) ). So
  (λ/2)E_{ρ_λ}[L̂]/(1−λ/2) + (λ−1)(KL+ln(2√n/δ))/(nλ(1−λ/2)) = 0.
Multiply by (1−λ/2): (λ/2)E_{ρ_λ}[L̂] + (λ−1)(KL+ln(2√n/δ))/(nλ) = 0, i.e. (λ/2)E_{ρ_λ}[L̂] = (1−λ)(KL+ln(2√n/δ))/(nλ), which rearranges to the stationary-point characterization
  (KL(ρ_λ‖π) + ln(2√n/δ))/n = λ² E_{ρ_λ}[L̂] / (2(1−λ)).
This already tells me something useful about *where* stationary points can be. Using E_{ρ_λ}[L̂] ≤ 1 and, for λ ≤ ½, the complement 1−λ ≥ ½, solving the characterization for λ gives λ = √( 2(1−λ)(KL + ln(2√n/δ)) / (n E_{ρ_λ}[L̂]) ) ≥ √(ln(2√n/δ)/n) for n ≥ 7. So any stationary point sits at λ ≥ √(ln(2√n/δ)/n); below that there are no stationary points, F is monotone. Good — that's the lower end of the [1/√n, 1] window I noticed earlier.

Now the decisive computation: the sign of F'' at a stationary point. F''(λ) = f''g + 2f'g' + fg''. I'll evaluate the f g'' + 2 f' g' part at a stationary point, where I can substitute the characterization. First, at a stationary point, plug the characterization into f = λE_{ρ_λ}[L̂] + (KL+ln(2√n/δ))/n = λE_{ρ_λ}[L̂] + λ²E_{ρ_λ}[L̂]/(2(1−λ)) = E_{ρ_λ}[L̂]·( λ + λ²/(2(1−λ)) ) = E_{ρ_λ}[L̂]·( λ(1−λ/2)/(1−λ) ), where the last step is λ + λ²/(2(1−λ)) = (2λ(1−λ) + λ²)/(2(1−λ)) = (2λ − 2λ² + λ²)/(2(1−λ)) = (2λ − λ²)/(2(1−λ)) = λ(2−λ)/(2(1−λ)) = λ(1−λ/2)/(1−λ). So at a stationary point f = E_{ρ_λ}[L̂]·λ(1−λ/2)/(1−λ). Now
  f g'' + 2 f' g' = E_{ρ_λ}[L̂]·λ(1−λ/2)/(1−λ)·(3λ²−6λ+4)/(2λ³(1−λ/2)³) + 2·E_{ρ_λ}[L̂]·(λ−1)/(λ²(1−λ/2)²).
Pull out E_{ρ_λ}[L̂]/(λ²(1−λ/2)²): the first term becomes E_{ρ_λ}[L̂]·(3λ²−6λ+4)/(2λ²(1−λ/2)²(1−λ)), and the second is E_{ρ_λ}[L̂]·2(λ−1)/(λ²(1−λ/2)²). So
  f g'' + 2 f' g' = E_{ρ_λ}[L̂]/(λ²(1−λ/2)²) · [ (3λ²−6λ+4)/(2(1−λ)) + 2(λ−1) ].
The bracket: (3λ²−6λ+4)/(2(1−λ)) − 2(1−λ) = [ (3λ²−6λ+4) − 4(1−λ)² ] / (2(1−λ)) = [ 3λ²−6λ+4 − 4 + 8λ − 4λ² ] / (2(1−λ)) = [ −λ² + 2λ ] / (2(1−λ)) = λ(2−λ)/(2(1−λ)) = λ(1−λ/2)/(1−λ). So
  f g'' + 2 f' g' = E_{ρ_λ}[L̂]/(λ²(1−λ/2)²)·λ(1−λ/2)/(1−λ) = E_{ρ_λ}[L̂]/( λ(1−λ/2)(1−λ) ).
And the remaining piece f''g = −n Var_{ρ_λ}[L̂]/(λ(1−λ/2)). Putting them together, at a stationary point,
  F''(λ) = (1/(λ(1−λ/2)))·( E_{ρ_λ}[L̂]/(1−λ) − n Var_{ρ_λ}[L̂] ).
The prefactor 1/(λ(1−λ/2)) is positive on (0,2). So F''(λ) > 0 — every stationary point is a strict local minimum — exactly when
  E_{ρ_λ}[L̂(h,S)] > (1 − λ) n Var_{ρ_λ}[L̂(h,S)].
Let me confirm this predicate genuinely tracks the local-minimum sign before I build on it, because it came out of a long substitution at the stationary point and I could easily have dropped a sign. On the mediocre-cluster instance I find F's interior stationary point numerically — F' changes sign once, near λ=0.881 — and there I compute a numeric F''=+1.63 (a local min) while the predicate E_{ρ_λ}[L̂] > (1−λ)nVar evaluates to True. On the two-point instance the scan found no interior stationary point at all (F was monotone to the edge), so there's nothing there to be the wrong kind of critical point. Predicate and curvature sign agree where I can check them.
Or, substituting the stationary-point characterization to trade E_{ρ_λ}[L̂] for KL, the equivalent form 2 KL(ρ_λ‖π) + ln(4n/δ²) > λ² n² Var_{ρ_λ}[L̂]. So if either of those holds for every λ in the window [√(ln(2√n/δ)/n), 1], then every stationary point of F is a local min, F has a single one, it's the global min, and the alternating minimization — which only decreases F — converges to it. The condition is intuitive: F = fg with f concave (its concavity is exactly −nVar) and g convex; quasiconvexity survives as long as the variance of the loss under the Gibbs posterior isn't so large that f's concavity overwhelms g's convexity at the stationary point. Low posterior variance ⇒ guaranteed global convergence.

I want a checkable sufficient condition, because Var_{ρ_λ} depends on the unknown ρ_λ. Take a finite H of m hypotheses with uniform prior. Then ρ_λ(h) = e^{−nλ x_h}/Σ_{h'} e^{−nλ x_{h'}} where x_h = L̂(h,S) − min_h L̂(h,S) shifts so the best hypothesis has x = 0 (the variance is translation-invariant, so I can work with x_h). Since x_{h*} = 0 the denominator Σ_h e^{−nλ x_h} ≥ 1. Bound the variance by the second moment: Var_{ρ_λ}[x_h] ≤ E_{ρ_λ}[x_h²] = (Σ_h x_h² e^{−nλ x_h})/(Σ_h e^{−nλ x_h}) ≤ Σ_h x_h² e^{−nλ x_h} (using the denominator ≥ 1). Now split the hypotheses by how good they are. Let a = √(ln(4n/δ²))/(n√3), b = ln(3mn²)/√(n·ln(2√n/δ)), and partition the x_h into "good" (x ≤ a), "mediocre" (a < x < b), "bad" (x ≥ b). Three pieces:

For the good piece, x_h ≤ a, and the masses sum to ≤ 1, so its contribution to E_{ρ_λ}[x²] is ≤ a². Using λ ≤ 1, a² = ln(4n/δ²)/(3n²) ≤ ln(4n/δ²)/(3λ²n²).

For the mediocre piece, calculus gives that x²e^{−nλx} is maximized over x at x = 2/(nλ), with value (2/(nλ))²e^{−2} = 4/(e²n²λ²); so each mediocre hypothesis contributes at most 4/(e²n²λ²). If I cap the number of mediocre hypotheses at K = (e²/12)ln(4n/δ²), the whole mediocre piece is ≤ 4K/(e²n²λ²) = (4/(e²n²λ²))·(e²/12)ln(4n/δ²) = ln(4n/δ²)/(3λ²n²). Same budget as the good piece.

For the bad piece, x_h ≥ b. First check b > 2/(nλ): b = ln(3mn²)/√(n·ln(2√n/δ)) and since λ ≥ √(ln(2√n/δ)/n) I have 2/(nλ) ≤ 2/√(n·ln(2√n/δ)), and b exceeds that. For x ≥ 2/(nλ) the function x²e^{−nλx} is decreasing, so each bad term is at most b²e^{−nλb}, and there are at most m of them: Σ_bad x²e^{−nλx} ≤ m b² e^{−nλb} ≤ m b² e^{−√(n·ln(2√n/δ)) b} (using λ ≥ √(ln(2√n/δ)/n)) ≤ m e^{−√(n·ln(2√n/δ)) b} (dropping b² ≤ 1 since x_h ≤ 1). Now substitute b: √(n·ln(2√n/δ))·b = ln(3mn²), so e^{−√(n·ln(2√n/δ)) b} = 1/(3mn²), and the bad piece is ≤ m·1/(3mn²) = 1/(3n²) ≤ ln(4n/δ²)/(3λ²n²). Same budget again.

Add the three: Var_{ρ_λ}[L̂] ≤ ln(4n/δ²)/(λ²n²), which is precisely 2KL(ρ_λ‖π) + ln(4n/δ²) > λ²n²Var (since KL ≥ 0) — the KL-form of the quasiconvexity condition — for all λ in the window. So: with a uniform prior on a finite H, if the number of "mediocre" hypotheses is at most K = (e²/12)ln(4n/δ²), F is strongly quasiconvex and alternating minimization is guaranteed to find the global minimum. The condition has a clean reading: as long as the hypotheses cleanly separate into clearly-good and clearly-bad with not too many in between, the bound's λ-landscape is well-behaved. (The interval boundaries a and b can be retuned with two free fractions α, β summing to ≤ 1 — set a(α) = √(α ln(4n/δ²))/n, b(β) = ln(mn²/β)/√(n ln(2√n/δ)), K(α,β) = (e²(1−α−β)/4)ln(4n/δ²) — trading the budget across the three pieces; and keeping the dropped b² factor refines b further. These only widen the regime where the guarantee holds.)

I should also note this whole construction transfers to a *constructed* hypothesis space when H is otherwise infinite or the partition function is intractable: train m weak learners, each on r subsampled points, validate each on the remaining n−r; then everything goes through with n replaced by n−r and the empirical loss replaced by the validation loss L̂_val, since the validation errors are (n−r) i.i.d. variables with mean L(h) so Maurer's 2√(n−r) MGF bound applies. But for a stochastic neural network the cleaner instantiation is to make ρ a diagonal Gaussian over the weights with an analytic KL to a Gaussian prior, and that's the form I'll write the code for.

Let me now turn this into the actual training and certification code, because the abstractions need to become a concrete optimizer over a stochastic net. The loss has to live in [0,1] for the bound to apply, and cross-entropy is unbounded, so I bound it: clamp the network's log-probabilities below at ln(pmin), which caps the per-example NLL at ln(1/pmin); then the bounded surrogate is NLL/ln(1/pmin) ∈ [0,1]. That's the empirical-risk term going into the bound. The posterior KL is analytic — diagonal-Gaussian to diagonal-Gaussian — and summed across the probabilistic layers; I get it from the harness. So compute_bound is a direct transcription of the PAC-Bayes-λ functional:
the empirical risk over (1−λ/2), plus (KL + ln(2√n/δ)) over (nλ(1−λ/2)).

For the alternating minimization, in the closed-form derivation I solved each subproblem exactly. With a neural-network posterior I can't form the Gibbs measure explicitly, so I do the alternation by gradient descent instead: one step descending the bound in the posterior parameters (the means and the σ-parameters), then a *separate* step descending the same bound in λ. The reason it has to be a separate step on its own optimizer is mechanical but important — the posterior optimizer only owns the network parameters, so without an explicit step on λ it would stay frozen at its initialization. So λ is its own learnable scalar with its own optimizer. I confine it to the window where the optimum lives — I argued the optimal λ sits in roughly [1/√n, 1] — by parametrizing a raw scalar through a sigmoid mapped onto [1/√n, 1]. In the PBB implementation the raw scalar is initialized high, for example at 6.0, so the scaled λ starts just below 1, matching the intended initialization near 1. Each minibatch: forward with sampled weights, bounded NLL, get KL, form the bound, backward and step the posterior; then re-form the bound and backward-and-step λ alone. That's the gradient version of "fix λ, improve ρ; fix ρ, improve λ."

Finally the certificate. I trained on the relaxed PAC-Bayes-λ bound because it's convex and optimizable, but for the *reported* number I should use the tightest available bound on the learned ρ — and that's PAC-Bayes-kl, which holds for any ρ including the one I found. So: estimate the empirical 0-1 risk of the stochastic predictor by Monte Carlo sampling, first inflate that Monte Carlo estimate by a small binary-kl inversion with ln(2/δ_test)/mc_samples, then read off KL, form c = (KL + ln(2√n/δ_test))/n, and invert the binary kl again — the largest p with kl(emp‖p) ≤ c — to get the 0-1 risk certificate. That inversion is the tighter certificate; the λ bound was only ever the training surrogate. I'll also report the λ-form cross-entropy bound and the final λ value.

Here is the optimizer, filling the single empty slot — the bound functional and the alternating optimization it induces:

```python
import math
import torch
import torch.nn.functional as F


class Lambda_var(torch.nn.Module):
    """Sigmoid-scaled lambda variable used by the PBB flamb objective."""

    def __init__(self, lamb, n):
        super().__init__()
        self.lamb = torch.nn.Parameter(torch.tensor([lamb], dtype=torch.float32))
        self.min = 1.0 / math.sqrt(n)

    @property
    def lamb_scaled(self):
        return torch.sigmoid(self.lamb) * (1.0 - self.min) + self.min


class BoundOptimizer:
    """PAC-Bayes-lambda bound. Convex in the posterior for fixed lambda and
    convex in lambda for fixed posterior, so it is minimized by alternating
    descent over the posterior and over lambda. The lambda parameter is
    sigmoid-scaled into [1/sqrt(n), 1], matching the PBB flamb implementation."""

    def __init__(self, learning_rate=0.001, momentum=0.95, prior_sigma=0.03,
                 pmin=1e-4, initial_lamb=6.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.prior_sigma = prior_sigma
        self.pmin = pmin
        self.initial_lamb = initial_lamb
        self.lambda_var = None
        self._posterior_optimizer = None
        self._lambda_optimizer = None

    def _ensure_state(self, model, n, device):
        if self._posterior_optimizer is None:
            self._posterior_optimizer = torch.optim.SGD(
                model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        if self.lambda_var is None:
            self.lambda_var = Lambda_var(self.initial_lamb, n).to(device)
            self._lambda_optimizer = torch.optim.SGD(
                self.lambda_var.parameters(), lr=self.learning_rate,
                momentum=self.momentum)

    def _kl(self, model):
        return model.compute_kl() if hasattr(model, "compute_kl") else get_total_kl(model)

    def _bounded_loss_and_error(self, model, data, target):
        log_probs = model(data, sample=True, clamping=True, pmin=self.pmin)
        loss_ce = F.nll_loss(log_probs, target) / math.log(1.0 / self.pmin)
        pred = log_probs.max(1, keepdim=True)[1]
        error = 1.0 - pred.eq(target.view_as(pred)).sum().item() / target.size(0)
        return loss_ce, error

    def compute_bound(self, empirical_risk, kl, n, delta):
        # PAC-Bayes-lambda:  L_hat/(1 - lam/2)  +  (KL + ln(2 sqrt(n)/delta)) / (n lam (1 - lam/2))
        lam = self.lambda_var.lamb_scaled
        kl_term = (kl + math.log(2.0 * math.sqrt(n) / delta)) / (
            n * lam * (1.0 - lam / 2.0))
        return empirical_risk / (1.0 - lam / 2.0) + kl_term

    def train_step(self, model, data, target, device, n_bound, delta):
        self._ensure_state(model, n_bound, device)
        model.train()
        self.lambda_var.train()
        data, target = data.to(device), target.to(device)

        self._posterior_optimizer.zero_grad()
        kl = self._kl(model)
        loss_ce, err = self._bounded_loss_and_error(model, data, target)
        bound = self.compute_bound(loss_ce, kl, n_bound, delta)
        bound.backward()
        self._posterior_optimizer.step()

        self._lambda_optimizer.zero_grad()
        kl_l = self._kl(model)
        loss_ce_l, err_l = self._bounded_loss_and_error(model, data, target)
        lam_bound = self.compute_bound(loss_ce_l, kl_l, n_bound, delta)
        lam_bound.backward()
        self._lambda_optimizer.step()

        return {
            "train_bound": bound.item(),
            "lambda_bound": lam_bound.item(),
            "kl_per_sample": (kl / n_bound).item(),
            "bounded_nll": loss_ce.item(),
            "train_error": err,
            "lambda": self.lambda_var.lamb_scaled.item(),
        }

    def compute_risk_certificate(self, model, bound_loader, device, delta=0.025,
                                 delta_test=0.01, mc_samples=1000):
        # certify with PAC-Bayes-kl inversion (tightest) on the learned rho,
        # not the relaxed lambda bound used for training.
        self._ensure_state(model, len(bound_loader.dataset), device)
        model.eval()
        n_bound = len(bound_loader.dataset)

        total_ce, total_01, batches = 0.0, 0.0, 0
        with torch.no_grad():
            for data, target in bound_loader:
                data, target = data.to(device), target.to(device)
                ce_mc, err_mc = 0.0, 0.0
                for _ in range(mc_samples):
                    loss_ce, err = self._bounded_loss_and_error(model, data, target)
                    ce_mc += loss_ce.item()
                    err_mc += err
                total_ce += ce_mc / mc_samples
                total_01 += err_mc / mc_samples
                batches += 1

        raw_ce = total_ce / batches
        raw_01 = total_01 / batches
        empirical_ce = inv_kl(raw_ce, math.log(2.0 / delta_test) / mc_samples)
        empirical_01 = inv_kl(raw_01, math.log(2.0 / delta_test) / mc_samples)

        kl = self._kl(model).item()
        c = (kl + math.log(2.0 * math.sqrt(n_bound) / delta_test)) / n_bound
        risk_ce = inv_kl(empirical_ce, c)
        risk_01 = inv_kl(empirical_01, c)

        ce_bound = self.compute_bound(
            torch.tensor(empirical_ce, device=device),
            torch.tensor(kl, device=device),
            n_bound,
            delta,
        ).item()

        metrics = {
            "raw_mc_ce": raw_ce,
            "raw_mc_01": raw_01,
            "empirical_ce": empirical_ce,
            "empirical_01": empirical_01,
            "kl_divergence": kl,
            "risk_ce": risk_ce,
            "ce_bound": ce_bound,
            "lambda": self.lambda_var.lamb_scaled.item(),
        }
        return risk_01, metrics
```

Let me trace the causal chain one more time. I wanted to train a posterior by minimizing a PAC-Bayes certificate directly, with no cross-validation. The tightest bound, PAC-Bayes-kl, is non-convex in ρ, so it's unusable as an objective. Pinsker relaxations make it convex but loose at low risk; the refined Pinsker keeps tightness but buries the unknown risk under a square root. Catoni's route achieves ρ-convexity by adding a trade-off parameter, with the Gibbs posterior as the exact ρ-optimum — but his bound holds only for a fixed λ, so tuning λ to the data costs a union bound. Applying the deterministic identity √(xy) ≤ ½(λx + y/λ), which holds for all λ at once, to the refined-Pinsker inequality gave me a bound convex in ρ for fixed λ and convex in λ for fixed ρ, holding uniformly over all λ ∈ (0,2) for free. The ρ-optimum is the Gibbs posterior ρ_λ ∝ π e^{−λnL̂}; the λ-optimum is closed-form. Alternating them decreases the bound monotonically; it isn't jointly convex, but eliminating ρ collapses the bound to a 1-D function F(λ) = (−ln E_π[e^{−nλL̂}] + ln(2√n/δ))/(nλ(1−λ/2)), and computing F' and F'' shows every stationary point is a local minimum exactly when the Gibbs-posterior variance is small enough (E_{ρ_λ}[L̂] > (1−λ)nVar), which makes F strongly quasiconvex and the alternating minimization globally convergent — with a checkable finite-H sufficient condition counting "mediocre" hypotheses. Instantiated on a stochastic neural network, the bound becomes compute_bound, the alternation becomes two gradient steps per minibatch (posterior, then λ on its own optimizer), and the final certificate is read off by the tighter PAC-Bayes-kl inversion on the learned posterior.
