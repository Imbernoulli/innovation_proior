Uniform did its job, and the numbers say where it stopped. Constant spacing in `t` pulled the two
workloads loglinear had wrecked back into reasonable territory — ImageNet inpainting fell from 13.748 to
6.070, and DIODE from 17.048 to 15.002 — and edges2handbags improved modestly from 5.634 to 5.180,
geometric-mean 7.78 against loglinear's 10.97. So the minimax lesson held: evenness removed the
oversized-step vulnerability, and the catastrophic workloads recovered most, exactly as the `(Δt)^2`
argument predicted. But uniform is, by its own admission, the "no bet" floor, and the residual tells me
the floor is not the ceiling. DIODE is still stuck at 15.0 — far worse than the other two — and it is the
NFE = 3 workload, where there are only `n = 2` steps and the uniform grid is `{1.0, 0.50005, 1e-4}`. With
the data endpoint at `t_min = 1e-4`, that means the *entire* near-data region — everything below `t ≈
0.5` where the denoiser commits to actual image content — is resolved by a single coarse step from 0.5
straight down to 1e-4. Uniform spent half its budget on the high-`t` half (`1.0 → 0.5`), where on a VP
bridge the trajectory is nearly straight and the conditioning is mostly already determined, and only half
on the entire low-`t` decade-and-a-half where the fine structure actually forms. That is the symptom of a
schedule that, having refused to bet on curvature at all, is now under-resolving the one region where
curvature genuinely lives.

So the minimax stance has earned its keep and now needs to be relaxed, with discipline. The lesson from
the two failures so far is sharp: loglinear bet *everything* on low-`t` and left a ruinous high-`t` step;
uniform bet *nothing* and left the near-data region coarse. The right answer is between them — a placement
that shifts resolution toward low `t`, where the bridge curves and image content is decided, *without*
recreating the single oversized high-`t` step that destroyed loglinear. I want a smooth, one-parameter
family that interpolates between uniform and the low-`t`-heavy extreme, so I can dial the amount of
low-`t` concentration rather than swing all the way to a constant ratio. The power-law warp is exactly
that family.

Let me derive it from the integration-error picture I already trust. The deterministic DBIM step is
Euler-style integration of the bridge trajectory, with per-step local error `≈ (Δt_i)^2·C_i` and total
`≈ Σ_i (Δt_i)^2·C_i`. The single-step truncation error of this kind of sampler is, empirically, large at
*low* noise and small at high noise — the trajectory is straight where the state is mostly noise/source
and bends where it resolves into data. So the curvature `C_i` is concentrated toward `t_min`, and the
Lagrange-optimal response `Δt_i ∝ 1/C_i` says: shrink the steps as `t` decreases. The question is the
*shape* of that shrinkage, and I want a controllable knob, not a fixed curve.

Build the family by warping a uniform grid through a monotone unbounded map. Place
`t_{i} = w(A·i + B)` with `w` a monotone warp and the endpoints fixing `A, B`. The power law
`w(z) = z^ρ` gives a clean closed form: interpolate *linearly in the warped coordinate* `t^{1/ρ}` and
undo the warp. With a ramp `u_i ∈ [0, 1]`,
`t_i = (t_max^{1/ρ} + u_i·(t_min^{1/ρ} − t_max^{1/ρ}))^ρ`. Check the endpoints: `u = 0` gives
`(t_max^{1/ρ})^ρ = t_max`, `u = 1` gives `(t_min^{1/ρ})^ρ = t_min`. The knob `ρ` does exactly the
interpolation I wanted. At `ρ = 1` this is plain linear interpolation in `t` — the uniform grid I just
ran, recovered as a special case (reassuring: the right generalization contains the thing it improves on).
As `ρ → ∞`, `t^{1/ρ} ≈ 1 + (1/ρ)log t`, and the warp degenerates to uniform spacing in `log t` — the
geometric ladder that collapsed. So the power-law family is a continuous *bridge* between the two extremes
I have measured: `ρ = 1` is uniform (gmean 7.78), `ρ → ∞` is loglinear (gmean 10.97), and somewhere in
between is the right amount of low-`t` concentration. That framing is the whole reason to try it — I am
not placing a fresh blind bet, I am moving along the one axis that connects my two data points, toward the
low-`t` end but stopping well short of the extreme that already failed.

What `ρ`? The cleanest objective would equalize the truncation error across `t`, and on this kind of
problem that numerical-balance point sits at a modest `ρ` (around 2–3 with a second-order corrector). But
I am not judged by raw trajectory-tracking accuracy; I am judged by FID, and accuracy is not perceptually
equal across `t`. Error at high `t` is cheap — near `t_max` the state is close to the source/noise
endpoint, the exact placement there barely changes the final image distribution. Error at low `t` is
decisive — that is where the denoiser commits to fine structure, and a misplacement shows up as a visible
artifact. So the FID-optimal `ρ` deliberately *over-allocates* to low `t`, pushing past the
numerical-balance point: trade tolerable high-`t` error for reduced low-`t` error. Across diffusion
models and step counts, `ρ = 7` sits in the broad, robust sweet spot of that trade — past the balance
point, but not so far that the high-`t` steps get coarse enough to recreate loglinear's failure. It is a
single fixed constant, the same for all three workloads, with no per-dataset tuning — which keeps me
inside the task's no-dataset-constants rule. So I fix `ρ = 7` and stop treating it as a knob.

Now I have to be careful, because the power-law schedule I know from variance-exploding diffusion
sampling is *not quite* the schedule this bridge sampler wants, and the harness makes the difference
concrete. In the EDM diffusion setting the schedule produces `N` nonzero noise levels and then *appends a
terminal zero* — the clean-data endpoint is `σ = 0`, reached by a final Euler hop, so the construction is
`ramp = linspace(0, 1, N)` over the nonzero levels plus an appended `0`. That is wrong here in two ways.
First, this is a *bridge*: the data endpoint is `t_min = 1e-4`, not zero. The DBIM update marches to
`t_min` and reads the bridge's `get_abc` coefficients there; appending a zero would push a denoiser query
to `t = 0`, off the schedule the sampler expects. Second, the contract is fixed at `n + 1` nodes with the
*terminal element equal to `t_min` exactly*. So the bridge power-law fill is: ramp across the *full*
`n + 1` nodes (`linspace(0, 1, n + 1)`, not `linspace(0, 1, n)`), apply the warp from `t_max` down to
`t_min`, and then pin the last element to `t_min` to undo the floating-point round-trip in the
`(·)^{1/ρ}` / `(·)^ρ` pair. No appended zero, no separate clean-data node — the warp itself runs all the
way to the bridge's data endpoint. This is the one place an off-by-one would silently change the method:
using the EDM `linspace(0, 1, n)`-plus-`append_zero` form would put a query at `t = 0` and leave only `n`
warped levels; the bridge form puts all `n + 1` nodes on the warp and terminates at `t_min`. I want the
bridge form, and I want the terminal pin for exactness, just as I needed it for the geometric fill (and
unlike uniform, which hit `t_min` for free).

Trace the construction once for DIODE (NFE = 3, `n = 2`) to see what changes versus uniform. The ramp is
`{0, 0.5, 1}`; with `ρ = 7`, `t_max^{1/7} = 1`, `t_min^{1/7} = (1e-4)^{1/7} ≈ 0.264`, so the warped
coordinate is `{1, 0.632, 0.264}` and the schedule is `{1.0, 0.632^7, 1e-4} ≈ {1.0, 0.0405, 1e-4}`.
Compare uniform's `{1.0, 0.50005, 1e-4}`: the middle node moved from 0.50 down to 0.04, dragging the
interior stop deep into the near-data region. Now the high-`t` half is covered by one step `1.0 → 0.04`
(a big step, but on the nearly-straight high-`t` VP trajectory where error is cheap), and the near-data
decade gets the second step `0.04 → 1e-4` — far finer resolution exactly where uniform was coarse. This
is the relaxation I wanted: more low-`t` resolution than uniform, but the high-`t` step lands at 0.04, not
loglinear's 0.01 with γ = 100, so it does not reproduce the catastrophic 99%-in-one-step placement. The
full scaffold module is in the answer.

So the delta from the uniform rung is a deliberate, bounded bet on low-`t` curvature: where uniform set a
constant `Δt`, the power-law warp shrinks `Δt` toward `t_min` with a single fixed exponent `ρ = 7`,
recovering uniform at `ρ = 1` and approaching loglinear only as `ρ → ∞` — so it lives strictly between my
two measured points. Here is what I expect against uniform's numbers. DIODE, where uniform left the entire
near-data region under one coarse step (still 15.0), is the workload with the most to gain — concentrating
the scarce NFE = 3 budget toward `t_min` should resolve the fine structure uniform missed, so I expect the
largest improvement there. edges2handbags and ImageNet have NFE = 5 (`n = 4`), so uniform already gave the
low-`t` region two or three steps; the power-law warp should still help by tightening the lowest steps,
but the gain should be smaller and is not guaranteed — if those bridges curve less than the EDM trajectory
`ρ = 7` was tuned for, the heavy low-`t` concentration could even cost a little at high `t` where the
inpainting mask (ImageNet) carries structure. The falsifiable claim is therefore: karras should edge below
uniform on the geometric mean, driven mainly by DIODE, while staying close on the two NFE = 5 workloads;
and it must *not* slide back toward loglinear's failure, which would show up as a regression on ImageNet
if `ρ = 7` over-bets low `t` on a bridge whose high-`t` end needs the resolution. If karras clears uniform
only narrowly and leaves one workload where the low-`t` bet costs more than it pays, that is the signal
that a fixed power-law exponent is still the wrong *shape* — too monotone, with no resolution returned to
the bending mid-range — which is the question the next rung will take up.
