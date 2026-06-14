I have a frozen bridge denoiser and a frozen DBIM update rule, and the one thing left to me is the list
of times the sampler stops at: `t_max = t_0 > t_1 > ... > t_n = t_min`, with `n = 4` here. Everything
about the model and the per-step coefficients is fixed; I am choosing node placement on a fixed
trajectory under a tiny budget. So before I reach for any curvature-aware warp, I want to start from the
schedule that comes with an actual *argument* attached, rather than from raw even spacing in `t` (that
is the scaffold default, and it is a "no bet" placement — it assumes the trajectory is equally hard
everywhere, which I have no reason to believe). The schedule with a real derivation behind it in this
family is the geometric one: equal spacing in `log t`, constant ratio `t_{i-1}/t_i`. Let me reconstruct
why that spacing is principled, decide whether the argument actually transfers to a frozen few-step
bridge sampler, and land it as the literal fill of `get_sigmas_uniform`.

The geometric ladder earns its keep in *annealed* sampling, so let me state that mechanism precisely,
because whether it applies here is the whole question. In annealed score-based sampling you do not run
one sampler; you run a descending ladder of noise levels and **warm-start each level from the output of
the level above**. The score at level `i` is trustworthy where the level-`i` perturbed distribution has
mass; the incoming samples are typical of level `i-1`. For the handoff to be worth anything, the
high-density region of level `i-1` must be covered by level `i`. If consecutive levels are too far
apart, level `i` has little mass where the level-`(i-1)` samples land, the score there is queried in a
weakly supported region, and the warm start fails. If they are too close, every rung barely changes
anything and a whole sampler run is wasted. So the spacing controls exactly one quantity — the
**overlap** between adjacent levels — and because a chain is only as strong as its weakest handoff, I
want that overlap *equal* at every rung.

Now compute the overlap, which means I need the shape of each perturbed distribution. Strip the data to
one point at the origin, so level `i` is `N(0, σ_i^2 I)` in `R^D`. The low-dimensional intuition says
two same-center Gaussians "overlap a lot near zero," and that intuition is a trap in high dimension.
For `x ~ N(0, σ^2 I)`, the squared norm `||x||^2/σ^2` is `χ^2_D`, and by the law of large numbers plus
the delta method the radius concentrates: `r = ||x|| ≈ N(√D·σ, σ^2/2)`. The mass of a high-dimensional
Gaussian is not near the center — it sits on a thin shell of radius `√D·σ` and width `≈ σ/√2`, the
Gaussian soap bubble. Two same-center Gaussians of different widths therefore live on two *different
shells*, radius `√D·σ_{i-1}` and `√D·σ_i`, and overlap is the one-dimensional question of whether those
shells are close enough on the radial line to intersect.

Model each level's radial mass as `N(m_i, s_i^2)` with `m_i = √D·σ_i`, `s_i^2 = σ_i^2/2`. The incoming
samples for level `i` populate level `i-1`'s high-density band `I_{i-1} = [m_{i-1} − 3s_{i-1},
m_{i-1} + 3s_{i-1}]`. The quantity I want constant is `C_i = P_{r ~ N(m_i, s_i^2)}(r ∈ I_{i-1})`.
Standardize `r` under level `i` and write `γ_i := σ_{i-1}/σ_i`. Then `m_{i-1} = γ_i m_i`,
`s_{i-1} = γ_i s_i`, and `m_i/s_i = √D·σ_i/(σ_i/√2) = √(2D)` — the `σ_i` cancels completely. The upper
standardized endpoint is `√(2D)(γ_i − 1) + 3γ_i`, the lower is `√(2D)(γ_i − 1) − 3γ_i`, so with `Φ` the
standard normal CDF, `C_i = Φ(√(2D)(γ_i − 1) + 3γ_i) − Φ(√(2D)(γ_i − 1) − 3γ_i)`. The overlap depends
on `i` through **one** thing only: the ratio `γ_i = σ_{i-1}/σ_i`. Not the absolute scale, not `σ_i`
itself. Demanding a common `C` across rungs therefore *forces* a common `γ`: a geometric progression
`σ_i = σ_1·γ^{-(i-1)}`. Take logs — `log σ_i = log σ_1 − (i−1)log γ`, linear in `i`. Constant
adjacent overlap **is** equal spacing in `log σ`. The geometric ladder is not an arbitrary smooth warp;
it is the unique spacing that equalizes the one quantity the annealing handoff depends on. That is a
genuinely strong argument, and it is why I want to try this schedule first rather than even spacing.

Equal spacing in `log t` also exposes its own concentration pattern, which I should be honest about,
because it is what I will be watching. Geometric spacing puts the nodes far apart at the high-`t` end
(one ratio-`γ` jump that, in absolute `t`, is enormous near `t_max`) and packs them tightly at the
low-`t` end near `t_min`. With `t_max = 1.0`, `t_min = 1e-4`, and `n = 4`, the constant ratio is
`γ = (t_max/t_min)^{1/n} = 10^{4/4} = 10`, so the schedule is `{1.0, 0.1, 0.01, 0.001, 0.0001}`. The
very first step covers `1.0 → 0.1` — nine-tenths of the entire interval in `t` — in a single denoiser
call, and the remaining three steps split the last 0.1. So this is an extreme low-`t` concentration: it
bets almost the whole budget on resolving the near-data region and trusts a single coarse jump to carry
the trajectory across the entire high-`t` half.

Here is where I have to check the transfer, because the bridge sampler I am filling is **not** an
annealed sampler. The geometric spacing was derived from one mechanism: each level *warm-starts the
next and the per-level sampler re-equilibrates*, so the only thing that matters is adjacent-level
overlap and a coarse early step is fine because the next level's own dynamics absorb it. The frozen
DBIM update does nothing of the kind. It is a deterministic (η = 0) marcher: at each scheduled `(s, t)`
pair it computes fixed coefficients from the bridge's own `get_abc`/`get_alpha_rho` at exactly those
two times and takes one step — there is no per-level Langevin run to re-equilibrate, no warm-start
overlap to preserve. The single denoiser call at the top of the schedule has to carry the state across
the whole `1.0 → 0.1` jump in one deterministic step, and the integration error of that step is set by
how far apart the two times are and how the trajectory curves between them, not by any overlap. So the
property that made geometric spacing optimal — equalizing handoff overlap — is *not the property this
sampler rewards*. What this sampler punishes is a single oversized step across a region where the
trajectory bends; and a γ = 10 jump from `t_max` is precisely one oversized step.

That gives me a clear, falsifiable read on what I expect. The low-`t` packing should help in the sense
that the near-data region — where the denoiser commits to actual image content and where error becomes
a visible artifact — gets three of the five stops, so fine structure should be resolved. But the lone
coarse `1.0 → 0.1` opening step is a genuine liability on a curving bridge trajectory: whatever error it
injects at the top is carried all the way down, and the tightly packed low-`t` steps cannot undo a
trajectory that has already drifted off the manifold up high. I therefore expect this schedule to be
*worse*, not better, than even the uniform default on workloads where the bridge curves appreciably in
the high-`t` half — and the bridge's source endpoint here carries real conditioning structure (edges,
the inpainting mask, the depth map), so the high-`t` region is not the "almost arbitrary pure noise"
that EDM gets to treat as cheap. That is the gap I am starting the ladder with on purpose: the most
*principled-looking* schedule in the family, derived from a mechanism the frozen sampler does not have,
so that the failure it produces tells me what the sampler actually wants.

Two implementation guards I owe the fill, both grounded in the contract. First, `log(0)` is `−∞`, and
the data endpoint `t_min` is tiny (1e-4) and could in principle be set to zero, so I floor it before
taking the log: `log(max(t_min, 1e-10))`. Second, the round trip `exp(linspace(...))` can miss `t_min`
by a few ULPs, and the contract demands the terminal node equal `t_min` *exactly*, so I pin the last
entry to `t_min` after exponentiating. With those two guards the construction is: log the endpoints,
space evenly across `n + 1` nodes in the log domain, exponentiate back, pin the terminal. That is
exactly the geometric branch of the standard noise-schedule routine, written in the sampler's `t`
domain — and it is the literal body that replaces `get_sigmas_uniform`. Monotonicity is automatic
(`linspace` in log is strictly decreasing, `exp` is increasing), the length is `n + 1` by construction,
the first node is `t_max` and the last is pinned to `t_min`, and the tensor goes to `device`. The full
scaffold module is in the answer.

So the delta from the scaffold default is concrete: where the default fills `get_sigmas_uniform` with
`linspace(t_max, t_min, n+1)` — constant spacing in `t` — I fill it with equal spacing in `log t`,
moving from a placement that assumes uniform difficulty to one that bets the whole budget on the
near-data end. The expectation I will hold this against is sharp. If geometric spacing's near-data
packing is what matters, FID should drop versus uniform across the three workloads. If, instead, the
single coarse high-`t` opening step is the dominant error source on a curving conditioned bridge — which
is what the absence of any re-equilibration mechanism leads me to predict — then this principled-looking
ladder should land *behind* the uniform default, especially on the workloads whose bridges carry the
most high-`t` structure. Either way the number tells me whether the few-step bridge sampler cares about
overlap (it shouldn't) or about not leaving any one interval oversized (it should), and that reading is
what the next rung will be built on.
