The SEAG numbers are the best so far and they point at the exact ceiling I now have to break. The mean
`final_gradient_norm` dropped from SEG's `0.182141` to `0.135449`, and both halves improved as I
predicted: `bilinear_fgn` `0.173788→0.160590` and `delta_nu_fgn` `0.190493→0.110307`. The decaying
anchor added the early contraction SEG lacked without R-SEG's permanent bias — the bilinear bias
floor stayed gone *and* the merely-monotone `(δ,ν)` field finally got contracted. The signature I told
myself to watch confirms the mechanism cleanly: `auc_log_iteration_log_grad` plunged from SEG's
`−0.346938` to `−1.107236`, a far steeper log-log descent, while the *final* norm improved only
moderately. That gap — strongly negative AUC, not-dramatically-smaller final value — is exactly the
fingerprint of "fast `1/k²` transient, then a noise-set floor."

But I have to read the noise columns before I trust that diagnosis, because last time I predicted
that a transient-only method would keep a wide `delta_nu` noise span, and this is where that prediction
gets tested. So let me pull the three regimes apart on paper. `bilinear_fgn` runs `0.158595 / 0.160590
/ 0.178678` across low/default/high — a span of `0.178678/0.158595 = 1.13×`, essentially flat, the
near-noiseless rotation (`σ=0.001`) barely feeling the regime change, exactly as it should. But
`delta_nu_fgn` runs `0.006672 / 0.110307 / 0.582557` — a span of `0.582557/0.006672 = 87×`. That is not
a mild widening, it is an *enormous* fan-out: `0.110307/0.006672 = 16.5×` from low to default and
`0.582557/0.110307 = 5.3×` from default to high. And the mean is the flat average of the two halves in
every regime — `(0.160590+0.110307)/2 = 0.135449`, `(0.158595+0.006672)/2 = 0.082633`,
`(0.178678+0.582557)/2 = 0.380617`, all matching the reported means to the digit — so the entire noise
sensitivity of SEAG lives in its `delta_nu` half, and that half swings by nearly two orders of
magnitude. The low-noise `delta_nu` of `0.006672` is the tell: when I turn the noise down the `1/k²`
transient is allowed to run almost to completion and reaches a value SEG never approached, while at
high noise the very same schedule stalls at `0.582557`. Same code, same decay, floor entirely set by
`σ`. That is the cleanest possible confirmation of "fast transient, `σ`-limited floor," and it is
worse than SEG's `delta_nu` span (`0.072 / 0.190 / 0.937`, only `13×`) — the acceleration bought a
spectacular low-noise number at the cost of an even wider variance fan. The AUC tells the same story
from the other side: it runs `−1.547 / −1.107 / −0.638` across low/default/high, steepest where the
noise is smallest, because when the floor is low the slope-`−2` plunge has the most room to run before
it flattens. So the binding defect is now sharp and it is *statistical*, not optimization.

Let me make the coupling precise, because the fix has to attack it directly rather than around it. An
anchor at strength `λ` toward a point `a` turns `F` into `G(z)=F(z)+λ(z−a)`, which is `λ`-strongly
monotone with a noise floor `~ησ²/λ` — so a *large* `λ` would crush the noise. But a large fixed `λ`
toward `z0` reintroduces R-SEG's bias `λ‖z0−z*‖`, and on bilinear `‖z0−z*‖=‖[10,10]‖=√200=14.14`, so
the bias is `14.14λ` — the disease I escaped earlier. The strength `λ` that fights noise and the
bias that a fixed far anchor incurs are coupled through the anchor-to-`z*` distance, and that is the
trap SEAG sits in: it can only make `λ` small — in fact it makes it *decay*, `c_k=1/(k+3)`, so by the
end `c_{6000}=1/6003=1.7·10^{-4}`, effectively zero — so it can never crush the noise, and the tiny
late strength is exactly why the high-noise floor sits at `0.58`.

Before the recursive fix, the cheaper moves that could plausibly kill that floor. The first is to stop
SEAG's anchor decaying all the way to zero: floor it at a positive constant, `c_k = 1/(k+3) + c_∞`, so there is permanent
strength `c_∞` to hold down the variance. But `c_∞` is a *constant* pull toward the fixed far `z0`, so
it is exactly R-SEG with the transient bolted on: it carries the permanent bias `c_∞·14.14` on
bilinear, and to keep that below the `0.16` I just won I would need `c_∞ < 0.011`, at which strength the
`delta_nu` noise floor `~σ²/c_∞` is barely touched. Same stick, same two ends — reject. The second is
Robbins–Monro: shrink the *step* rather than add strength, `τ_k=τ0/√k`, so the injected-noise
variance per step `~τ_k σ²` decays and the floor is annealed away. But a decaying step also throttles
the contraction I need on the merely-monotone `delta_nu`, and I can size the damage: the cumulative
progress over the run scales with `Σ_k τ_k ~ τ0·2√n`, which over the 6000 `delta_nu` steps is `~155τ0`
against the `6000τ0` a constant step delivers — a `~39×` collapse in total drive. The bilinear rotation
is worse off still, because it needs a step bounded away from zero to keep turning; a vanishing `τ`
freezes the angular motion before the radius has decayed. So Robbins–Monro trades the floor for a
crippled rate, and it still anchors nowhere near `z*`. Reject. The third is to average the noise out of
the oracle — draw the gradient several times and mean it — but the interface forbids this: `oracle.noise()`
gives one fresh draw per call, `sfo_calls` must equal the number of `oracle.grad` calls consumed, and
the iteration counts are fixed, so extra evaluations to shrink `σ²/m` would blow the budget the metric
is scored against. Reject on the interface. Every cheap move either re-pays the bias, throttles the
rate, or breaks the contract.

So I go back to the one fact that has been nagging me since the R-SEG step, because it is the lever the
cheap moves all miss. The anchored solution `w*` — the zero of `G(z)=F(z)+λ(z−z0)` — is *closer* to
`z*` than `z0` was: `‖w*−z*‖ ≤ ‖z*−z0‖` and `‖w*−z0‖ ≤ ‖z*−z0‖`, both from `λ‖w*−z*‖² ≤
λ(z*−z0)ᵀ(z*−w*)` (monotonicity of `F` at `w*` and `z*`) and polarization. The bias and the strength
are only coupled through *distance to the anchor*, and this inequality says solving the anchored
problem hands me a point at a *smaller* distance. So what if I do not anchor once at the far `z0`? Solve
the anchored problem approximately, land at a point closer to `z*`, *re-anchor there*, and crank `λ`
up — because the new anchor is closer, the bias `λ·(distance)` tolerates a bigger `λ`, and a bigger
`λ` makes the next subproblem more strongly monotone, better conditioned, cheaper to solve under noise.
A chain of warm restarts, each anchor closer and each penalty larger, breaks the coupling: the bias
stays bounded because every anchor is close, while `λ` climbs geometrically and the noise floor
`~σ²/λ` shrinks geometrically. This is the RAIN idea, and I have to build it at the operator level.

I redo the recursion on `F`, never touching function values, because the single-player convex version
of this trick rests on `min f ≤ f`, an inequality with no saddle-point analogue — neither
`min-max f ≤ f` nor `≥ f` holds, so I cannot descend a merit function; I can only drive the residual
`‖F‖`. Define a recursively regularized sequence
`f^(s+1) = f^(s) + (λ_{s+1}/2)‖x−x_{s+1}‖² − (λ_{s+1}/2)‖y−y_{s+1}‖²`, where `(x_{s+1},y_{s+1})` is the
approximate solution returned by round `s` (the sign split — `+` on `x`, `−` on `y` — is what keeps it a
saddle regularizer rather than a joint convexification), and the strengths grow geometrically,
`λ_{s+1}=(1+γ)λ_s`. In operator form, after `s` anchors,
`F^(s)(z)=F(z)+Σ_{i=1}^s λ_i(z−z_i)` — the penalties accumulate, each a fresh anchor `z_i` with its own
strength. Run `S=⌊log(L/λ)⌋` rounds so the accumulated strength climbs from `λ` up to about `L`: then
`F^(s)` is at least `~λ(1+γ)^s`-strongly monotone while staying `≤2L`-Lipschitz (it is `F`, `L`-Lipschitz,
plus a sum of penalties whose total coefficient I cap at `~L`), so its condition number
`2L/(λ(1+γ)^s)` *shrinks* toward `O(1)`. The later subproblems are well-conditioned, so even under
noise they are cheap to solve accurately — that is the whole payoff.

The central question is whether driving each subproblem's residual small actually makes the final
`‖F(z_S)‖` small, or whether the accumulating penalty terms leave a residue I never remove. Peeling off
the penalties, using strong monotonicity of `F^(S-1)` to convert distances back into residuals, and
chaining the exact solutions through `‖z*_j−z*_{j-1}‖ ≤ ‖z*_{j-1}−z_j‖` (the same non-expansiveness,
applied round to round), the messy double sum collapses to a clean recursive anchoring lemma:

  `‖F(z_S)‖ ≤ 16λ Σ_{s=1}^S (1+γ)^{s-1} ‖z*_{s-1}−z_s‖`,

the final gradient norm is a *geometrically-weighted sum of per-round subroutine errors*. This is the
inequality the whole method turns on, so let me read it for the balance it enforces. The weight
`(1+γ)^{s-1}` grows, which looks alarming — later errors count more. But the strong monotonicity of
subproblem `s` is `λ(1+γ)^s`, growing at the *same* geometric rate, so I can drive the per-round error
`‖z*_{s-1}−z_s‖` down at that rate and the product `(1+γ)^{s-1}·(error)` stays flat, not exploding — a
fair fight, growth against growth. Solving each subproblem with a two-phase epoch-SEG (a fixed step to
kill optimization error, then a shrinking step to eat the statistical floor of that well-conditioned
subproblem) lands the total oracle cost at the additive statistical floor `Õ(σ²ε^{-2}+κ)`, with no
spurious `L²` and no `ε^{-4}` — the recursion buys back the two factors of `ε` that a single far anchor
cannot, precisely because each subproblem it solves is `O(1)`-conditioned instead of `L/λ`-conditioned.
The merely-monotone case reduces to this via one cold anchor at `z0` with `λ=min(ε/D,L)`. This is the
method I want; the theory is the point of it, but the algorithm as stated is a
triple-nested loop and the harness runs exactly one loop, so I have to collapse it.

Collapse it. Set the inner counts to one — `N_s=1`, `K_s=0`, a single SEG iteration per round — so
"round `s`" and "SEG step `t`" become the same index, and the accumulated penalty `Σ_i λ_i(z−z_i)`
becomes, at iteration `t`, an anchor pulling toward the stored past iterates with geometric weights.
One extragradient step with that running anchor:

  predictor: `w = z − τF(z) + τλ·Σ_j w_j(z_j − z) + noise`,
  corrector: `z_next = z − τF(w) + τλ·Σ_j w_j(z_j − w) + noise`,

the regularizer being a geometrically-weighted running average of the trajectory already written into
state. Two implementation facts make this runnable in `O(d)` rather than `O(td)`. First, I never need
the iterates individually: `λΣ_j w_j(z_current − z_j) = λ[(Σ_j w_j)z_current − Σ_j w_j z_j]`, so I keep
two running buffers — a scalar `weight_sum = Σ_j w_j` and a vector `weighted_flow_sum = Σ_j w_j z_j` —
and the anchor contribution is `τλ(weighted_flow_sum − weight_sum·z_current)`, a single scalar-times-vector
and a subtraction, `O(d)`. Second, the sign: that expression is a pull from the current point *toward*
the weighted average of past points, which is the moving anchor — re-anchoring toward where the
trajectory has been, weighted to favor the recent past, with `γ` setting how fast the weighting grows.
Each new iterate `z_next` is inserted with weight `current_weight = γ(1+γ)^{step_index+1}` (the
stored-iterate convention is one-based even though the displayed recurrence is zero-based, because
`z_next` is written *after* the step is taken), updating both buffers by adding `current_weight` and
`current_weight·z_next`.

A growing geometric weight is exactly the kind of thing that silently overflows, so the magnitudes
matter. For bilinear, `γ=0.001` over 900 steps: `(1+γ)^{900}=1.001^{900}=
e^{900·0.0009995}=e^{0.900}=2.459`, so nothing overflows, and the running `weight_sum` telescopes to
`Σ_{j=1}^{900} γ(1+γ)^j = (1+γ)((1+γ)^{900}−1) = 1.001·1.459 = 1.46`. For `delta_nu`, `γ=0.0001` over
6000 steps: `(1.0001)^{6000}=e^{6000·0.00009999}=e^{0.600}=1.822`, and `weight_sum → (1.0001)(0.822)=
0.822`. Both `O(1)` — the tiny `γ` is chosen precisely so `(1+γ)^t` stays bounded over the fixed
horizons while the *accumulation* still climbs. The effective anchor strength — the coefficient
multiplying `z` in the pull — is `τλ·weight_sum`, which grows from `0` at the start to `0.1·0.1·1.46 =
0.0146` on bilinear and `1.0·0.01·0.822 = 0.0082` on `delta_nu` by the end. Set that against SEAG's
`delta_nu` end strength `c_{6000}=1/6003=1.67·10^{-4}`: RAIN finishes with an effective strength
`0.0082/1.67·10^{-4} ≈ 49×` larger, and *growing* where SEAG's was *shrinking*. That single ratio is
the whole mechanism — the noise floor `~σ²/strength` that stranded SEAG at `0.58` gets divided by a
factor that keeps climbing.

The moving anchor lags the current iterate, and the switch-on timing matters. The first step has empty
buffers (plain extragradient, the same `14.142→14.072` opening as before). At step 2 the only stored
iterate is `z_1` and the current point *is* `z_1`, so `anchor = τλ(w_1 z_1 − w_1 z_1) = 0` — still no
pull. Only at step 3, with `z_2` current, does `anchor = τλ w_1(z_1 − z_2)` become nonzero. So the
moving anchor is genuinely a *memory* term: it pulls toward the older stored points and switches on as
the trajectory advances past them, never dragging the current point toward a stale copy of itself.

Now why this should finally cross below SEAG's floor, and where its own limit sits. The structural
difference is the *direction* of the strength. SEAG's regularization decayed toward zero, so its
strong-monotonicity vanished and its noise floor was fixed by `σ`; RAIN's effective regularization
toward the recent trajectory *grows* — the geometric weights `(1+γ)^j` accumulate into `weight_sum`, so
late iterations are increasingly strongly monotone around where the trajectory has settled, which is
increasingly close to `z*`. The contraction keeps tightening instead of stalling, so the noise floor
keeps shrinking, while the bias stays bounded because the anchor tracks the moving trajectory, not the
far fixed `z0` — the non-expansiveness guarantees each stored point is no farther from `z*` than `z0`
was, and the weighting favors the recent (closer) ones. The state carries `z`, `step_index`, and the
two buffers; two operator evaluations, two noise draws, `O(d)` buffer updates. The harness fixes
`τ=0.1, λ=0.1, γ=0.001` on bilinear and `τ=1.0, λ=0.01, γ=0.0001` on `(δ,ν)`; the full module is in the
answer.

The two instances get different constants, and I want to see that the split is forced by the problem
geometry rather than tuned by hand. Two knobs matter: the per-step anchor gain `τλ` — how hard a single
step pulls toward the running average — and the growth rate `γ` — how fast the weighting, and hence the
accumulated strength, climbs. Take `γ` first, because it is the one that can break the run outright.
`weight_sum` is bounded iff `(1+γ)^n` is, and the exponent is `γ·n` (since `(1+γ)^n = e^{n·ln(1+γ)} ≈
e^{γn}` for tiny `γ`), so the binding constraint is `γn = O(1)`: I need the total geometric climb over
the *whole horizon* to be order one, not per step. Bilinear runs `n=900`, so `γ=0.001` gives `γn=0.9`
and `(1+γ)^n=e^{0.9}=2.46`; `delta_nu` runs `n=6000`, so `γ=0.0001` gives `γn=0.6` and `e^{0.6}=1.82`.
Both land the exponent near one. Had I reused bilinear's `γ=0.001` on the 6000-step `delta_nu`, the
exponent would be `γn=6` and `(1+γ)^n=e^6≈403`, so `weight_sum≈400` and the effective strength
`τλ·weight_sum≈1·0.01·400=4` — vastly over-`L` (the field is `~O(1)`-Lipschitz), an anchor so strong it
would drag the iterate to the trajectory average and swamp the operator entirely. So `γ` must scale as
`~1/n`, and the `10×` gap between the two `γ` values is exactly the `~6.7×` gap in horizons, rounded.
Now `τλ`: on bilinear `τ=0.1, λ=0.1` gives `τλ=0.01`, and on `delta_nu` `τ=1.0, λ=0.01` also gives
`τλ=0.01` — the per-step anchor gain is held *identical* across the two instances, and only the
gradient step `τ` and the raw strength `λ` are swapped inversely to keep their product fixed. That is
the right invariant: `τ` is dictated by the field (the pure rotation tolerates only `τ=0.1` before the
`−τ²I` curvature stops being a contraction, while the clipped `delta_nu` field takes the full `τ=1`),
so once `τ` is pinned by stability, `λ` is chosen to hold the anchor gain `τλ` at the common `0.01` that
makes the accumulated end-strength `τλ·weight_sum` land at `~0.008–0.015` — comparable on both problems,
large enough to bite the noise floor, small enough not to bias. The constants are not free; they are two
degrees of freedom (`τλ` and `γn`) both pinned to `O(10^{-2})` and `O(1)` by stability and by the fixed
horizons.

So the falsifiable bar this method must clear, against the SEAG numbers. The growing moving anchor
attacks the binding constraint — the `σ`-set floor — rather than the transient, so I expect a large,
not marginal, drop in the mean `final_gradient_norm` from SEAG's `0.135449`. Both halves should fall
substantially: `bilinear_fgn` from `0.160590`, since the near-noiseless rotation lets the growing
contraction run almost unimpeded, and `delta_nu_fgn` from `0.110307`, since the growing strength
around the settled trajectory directly crushes the noise that stranded SEAG. So the mean should land
roughly an order of magnitude below SEAG. The AUC I will not call
precisely, and I want to be clear why: SEAG's `−1.107` came from a sharp early plunge into a low floor,
whereas RAIN trades that sharp early plunge for a *sustained* descent — the anchor is weak early
(`weight_sum` near zero) and strong late — and those two shapes push the integrated log-log area in
opposite directions, so I cannot honestly predict whether RAIN's AUC lands more or less negative than
SEAG's without running it. The single cleanest test is instead the high-noise `delta_nu`: SEAG blew up
to `0.582557` there because its floor scaled with `σ` and its late strength was `~10^{-4}`; if the
growing anchor genuinely controls variance, this method's high-noise `delta_nu` should be *far* smaller —
the `49×`-larger, still-growing end strength should divide that floor down by a comparable factor. If
it instead still blows up to several tenths, then the moving anchor is not actually crushing the noise
floor and the recursion's promise has not materialized in the collapsed single-loop form. That
high-noise `delta_nu` number, against SEAG's `0.582557`, is the decisive falsifiable comparison.
