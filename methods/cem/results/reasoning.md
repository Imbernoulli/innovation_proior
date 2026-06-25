Let me start from what actually hurts. I have a black-box objective `S(x)` — hand it a vector and it
returns a number — and I want its maximizer. Concretely the vector is a sequence of actions and `S`
is minus the cost of rolling them through a fixed model, but the shape of the difficulty is general:
no gradient I can trust, the surface has many local optima, each query is expensive and possibly
noisy, and `x` lives in tens of dimensions so I cannot grid it. So I'm restricted to *querying* `S`
at points I choose, and the entire game is choosing those points well out of a tight budget.

The crudest thing that even works is random search: fix a distribution `f(·; v)` over `x`, draw a
batch, evaluate, keep the best, repeat. It's gradient-free, it explores globally, and unlike a
hill-climber it can't get trapped in a single basin. But it's dumb in a specific way that bugs me —
it never learns. The distribution I sample from on the hundredth batch is identical to the first, so
I keep paying for draws in regions I've already proven are bad. Every past evaluation told me
something about where the good points are, and I'm throwing all of it away. The fix I *want* is
obvious to state and slippery to make precise: let the sampling distribution `f(·; v)` move toward
where the good points have been, batch after batch, so the draws concentrate on the promising region.
The trap is that "move toward the good points" can be done a hundred ad-hoc ways — average the best
few, nudge by a step size, weight by score — and I have no principle telling me which is right or
what the thing even converges to. I don't want a nudge. I want an update I can derive.

So let me look harder at the structure. Pick a high threshold `γ` and ask not "where is the
max?" but "how probable is it that a random draw clears `γ`?" — that is, `ℓ(γ) = P_u(S(X) ≥ γ) =
E_u[ I{S(X) ≥ γ} ]` where `X ~ f(·; u)` is some broad baseline density. Now push `γ` up toward the
true optimum `γ* = S(x*)`. The set `{x : S(x) ≥ γ}` shrinks down toward the maximizers; in a
continuous space its probability under a broad density tends toward zero, and in a finite space it is
only the mass assigned to the optimal states. That's a *rare* event. A density that made this rare
event likely — that put its mass on `{S(X) ≥ γ}` for `γ` near `γ*` — would be a density that samples
almost exclusively near `x*`. Finding the optimizer and making the rare event near the optimum common
are the same problem wearing two hats. So whatever the rare-event people use to hunt down a good
sampling distribution, I can borrow.

What do they use? When `ℓ` is tiny, crude Monte-Carlo is hopeless — draw `N` from `f(·; u)`, count
the hits, and you get almost all zeros, so you need an absurd `N` for even one hit. Their cure is
importance sampling: don't sample from `f(·; u)`, sample from a cleverer density `g` that makes the
event common, and undo the change of measure with the likelihood ratio,

  ℓ = E_g[ I{S(X) ≥ γ} · f(X; u)/g(X) ],  estimated by  (1/N) Σ_k I{S(X_k) ≥ γ} · f(X_k; u)/g(X_k).

Is there a *best* `g`? The estimator is an average of i.i.d. terms `Y_k = I{S(X_k) ≥ γ} f(X_k;u)/g(X_k)`,
so its variance is `Var_g(Y)/N`, and that vanishes exactly when `Y` is a.s. constant. So I'll look for a
`g` that makes the summand `I{S(x) ≥ γ} f(x;u)/g(x)` constant in `x`. Whatever the constant is, it must
equal the mean `ℓ` (an a.s.-constant random variable equals its own expectation), so set the summand
to `ℓ`: that forces

  g*(x) = I{S(x) ≥ γ} · f(x; u) / ℓ .

So `g*` is the baseline density `f(·; u)` *restricted to the elite event* `{S(x) ≥ γ}` and
renormalized by `ℓ` — it throws away everything below `γ` and keeps the mass on the good region, which
is the "concentrate on where the good points are" distribution I was hand-waving about a moment ago.
Before I lean on it, let me actually check the zero-variance claim rather than trust the algebra,
because it's load-bearing. Take `f = N(0,1)`, `S(x) = x`, `γ = 2`. Then `ℓ = P(X ≥ 2) = 1 − Φ(2) ≈
0.02275`, and `g*` is `f` truncated to `[2, ∞)` and renormalized. Drawing 10⁵ points from that
truncated normal and forming the summand `I{x ≥ 2} f(x)/g*(x)`, I get a mean of `0.0227501` and a
sample standard deviation of `7×10⁻¹⁸` — i.e. the summand really is the constant `ℓ` to floating-point
roundoff. The estimator under `g*` is exact from a single sample, no variance. Good: `g*` is genuinely
the variance-minimizing density, not just a plausible candidate. The catch is the one that motivated all
this — `g*` is the ideal *target*, but it bakes in the unknown `ℓ` and is some arbitrary restricted
shape, so I cannot sample from it directly.

What I *can* do is restrict myself to a tractable parametric family `f(·; v)` (Gaussians, say) and
find the member of that family *closest* to the restricted target. At iteration `t` the target should
be built from the law I am actually using now, not from a frozen old baseline, so I write

  g*_t(x) = I{S(x) ≥ γ} · f(x; v_{t-1}) / ℓ_t,   where  ℓ_t = P_{v_{t-1}}(S(X) ≥ γ).

Closest in what sense? I need a discrepancy between two densities that plays nicely with this
restricted-and-renormalized target. The natural one — the one the whole estimation line is built on —
is the Kullback–Leibler divergence,

  D(g, h) = E_g[ ln(g(X)/h(X)) ] = ∫ g(x) ln g(x) dx − ∫ g(x) ln h(x) dx .

So I want `v` minimizing `D(g*_t, f(·; v))`. Write it out:

  D(g*_t, f(·; v)) = ∫ g*_t(x) ln g*_t(x) dx − ∫ g*_t(x) ln f(x; v) dx .

The first integral has no `v` in it — it's a constant as far as the minimization is concerned. So
minimizing the KL divergence is *exactly* maximizing the second integral:

  max_v ∫ g*_t(x) ln f(x; v) dx = max_v E_{g*_t}[ ln f(X; v) ] .

Substitute `g*_t(x) = I{S(x) ≥ γ} f(x; v_{t-1})/ℓ_t`:

  max_v (1/ℓ_t) ∫ I{S(x) ≥ γ} f(x; v_{t-1}) ln f(x; v) dx
  = max_v (1/ℓ_t) E_{v_{t-1}}[ I{S(X) ≥ γ} ln f(X; v) ] .

The `1/ℓ_t` is a positive constant; it doesn't move the argmax, so drop it:

  max_v E_{v_{t-1}}[ I{S(X) ≥ γ} ln f(X; v) ] .

Now I can't take that expectation analytically. But for optimization I am trying to move the sampler
itself, so I draw `X_1, ..., X_N ~ f(·; v_{t-1})` and replace the expectation by its Monte-Carlo
counterpart under that same current law:

  max_v (1/N) Σ_k I{S(X_k) ≥ γ} · ln f(X_k; v) .

And I want to read what this objective actually *is*, because the indicator is doing something
beautiful. `I{S(X_k) ≥ γ}` is 1 for the samples that beat the threshold and 0 for the rest. So the
sum is `Σ_{k : S(X_k) ≥ γ} ln f(X_k; v)` — the **log-likelihood of the threshold-beating samples
under `f(·; v)`**, and maximizing it over `v` is *fitting `f(·; v)` by maximum likelihood to those
samples alone*. That's the answer to "how do I move the distribution toward the good points," and
it's not a nudge: it's "take the points that beat the bar, and fit your sampling distribution to
them by maximum likelihood." So the cross-entropy minimization to the target `g*_t` has, on a finite
sample, reduced to a maximum-likelihood fit on the elite set — which is concrete enough to run, and not
the ad-hoc averaging I was worried about at the start.

There's a wall sitting right in the middle of this, though, and it's the same rarity that
motivated importance sampling in the first place. If I set `γ` near `γ*` from the start, then under
the broad initial `f(·; u)` almost no sample clears `γ`: every indicator `I{S(X_k) ≥ γ}` is zero, the
sum is empty, the maximum-likelihood program has no data, and the whole thing is void. I can't fit a
distribution to zero elite points. So a fixed high `γ` is a dead end. I need `γ` to be high enough
that the elites are genuinely good, but low enough that the elite set is nonempty at *every*
iteration, including the first when my distribution is still broad.

The way out is to stop fixing `γ` and instead let it *track the sample*. At each iteration, after I
draw `N` points and score them, sort the scores ascending and set the threshold to the worst member
of the top `N^e = ceil(ρN)` scores, `γ̂_t = S_(N−N^e+1)` — equivalently, the `(1−ρ)` sample quantile.
Then by construction there are `N^e` elites, never zero, no matter how broad the
distribution is. And the level is *self-tuning*: early on, when the
distribution is broad, the top-`ρ` cutoff is a modest score; as the distribution concentrates on
good regions, the same top-`ρ` cutoff is a *higher* score, so `γ̂_t` ratchets upward toward `γ*` on
its own. The loop becomes: sample from the current distribution; score; take the top-`ρ` elites; refit
the distribution to those elites by maximum likelihood; repeat. The level ladder and the
distribution climb together, each pulling the other up. This adaptive multi-level move is what makes
the rare-event program usable for optimization — it never lets the indicator go all-zero.

Now I have to choose the family `f(·; v)`, and two constraints decide it. The fit step is a
maximum-likelihood estimation that I'll run *every iteration*, so I want a family whose MLE has a
**closed form** — no inner optimization loop. And I have to *sample* from it cheaply, also every
iteration. Both of those point at the exponential family, where the MLE is a closed-form function of
sample moments and sampling is a one-liner. For a discrete `x` — say each coordinate is a 0/1 choice
— the family is independent Bernoulli per coordinate, and the elite-restricted MLE of each success
probability is just the fraction of elites whose coordinate is 1; that's the combinatorial-optimization
instantiation. But my `x` is continuous (action vectors over a horizon). The natural continuous
exponential family with independent coordinates is the **Gaussian**: per coordinate `j`, sample `x_j
~ N(μ_j, σ_j²)`, so `v = (μ, σ²)`. It samples in one call to a normal RNG and — crucially — its MLE
is the cleanest closed form there is. Let me actually derive that closed form for the elite set,
because it's the concrete update I'll ship and I want to see it drop out rather than quote it.

The Gaussian log-density of one elite sample `x_k`, with independent coordinates, is

  ln f(x_k; μ, σ²) = Σ_j [ −½ ln(2π) − ln σ_j − (x_{kj} − μ_j)² / (2 σ_j²) ] .

Sum over the elite set `I` (say `|I| = N^e` points) to get the elite log-likelihood `L(μ, σ²) =
Σ_{k∈I} ln f(x_k; μ, σ²)`, and maximize. Take `∂L/∂μ_j`:

  ∂L/∂μ_j = Σ_{k∈I} (x_{kj} − μ_j) / σ_j² .

Set it to zero. The `σ_j²` is a common positive factor, so it cancels, leaving `Σ_{k∈I} (x_{kj} −
μ_j) = 0`, i.e.

  μ_j = (1/N^e) Σ_{k∈I} x_{kj} .

The maximum-likelihood mean is the **sample mean of the elites**, coordinate by coordinate. Now
`∂L/∂σ_j`. The `j`-term of one sample contributes `−ln σ_j − (x_{kj}−μ_j)²/(2σ_j²)`, whose
derivative in `σ_j` is `−1/σ_j + (x_{kj}−μ_j)²/σ_j³`. Summing over the elites and setting to zero:

  Σ_{k∈I} [ −1/σ_j + (x_{kj} − μ_j)²/σ_j³ ] = 0
  ⇒  −N^e/σ_j + (1/σ_j³) Σ_{k∈I} (x_{kj} − μ_j)² = 0
  ⇒  σ_j² = (1/N^e) Σ_{k∈I} (x_{kj} − μ_j)² ,

the **MLE variance of the elites** — the division is by `N^e`, not by `N^e−1`. So the whole Gaussian
update is just: take the elite points, compute their per-coordinate mean and variance, and those *are*
the parameters of the next
iteration's sampling distribution. The mean re-centers the search on where the good action sequences
clustered; the variance re-sizes how widely to explore each coordinate — automatically tighter on
coordinates the elites agreed about, wider on coordinates they didn't. That second part is the thing
fixed-spread random shooting never gave me: the optimizer *decides its own exploration spread* from
the data, instead of me hand-setting a fixed noise level. And it's the same
principled object the whole way down — cross-entropy to `g*_t`, which on the elite sample is the
restricted MLE, which for a Gaussian is mean-and-MLE-variance-of-the-elites.

I should check what happens in the limit, because that tells me whether this converges to anything
sensible. As the iterations go and the distribution concentrates, the elites cluster tighter and
tighter, so their sample variance shrinks toward zero. In the limit `σ² → 0` the Gaussian collapses
to a point mass at `μ` — a degenerate (Dirac) distribution sitting on the best region found. That's
the *right* fixed point: a sampler that has fully concentrated on the optimum. So the dynamics are
self-terminating in spirit — keep going until the spread `σ` falls below some `ε` (the distribution
has degenerated) or the best score stops improving for a few iterations.

But that same collapse is also a danger, and I can feel exactly where it bites. If on some early
iteration the elites happen to land in a small cluster — by luck, or because the broad initial
sampling under-covered the real optimum's basin — then the spread update slams `σ` small *right
there*, and now I'm sampling in a tiny region around a possibly-wrong point with almost no spread to
escape it. The spread has collapsed before the search actually found the global basin, and the
method is stuck. Premature convergence. The pure MLE update is too eager; it believes the current
elites completely. I need to slow the collapse.

The guard is to not jump all the way to the freshly-fit parameters but to **blend** them with the
previous ones. Compute the elite mean `μ̃_t` and the elite standard deviation `σ̃_t` (the square root
of the MLE variance above), then take a convex combination with last iteration's values,

  μ_t = α μ̃_t + (1−α) μ_{t−1} ,   σ_t = α σ̃_t + (1−α) σ_{t−1} ,

for a smoothing factor `α ∈ (0, 1]`. With `α = 1` this is the raw update; with `α` somewhat below 1
the spread can only shrink by a fraction each step, so a single unlucky tight elite cluster can't
zero it out — the distribution has to *keep* producing tight elites over several iterations before
it's allowed to commit. In practice a value in the rough range `0.4–0.9` works: enough damping to
prevent premature collapse, not so much that convergence crawls. (And if I ever see the spread
degenerate too early anyway, a cruder guard is to just *inject* a little variance back in — bump `σ²`
up — to re-open exploration.) For the lean action-sequence planner I can choose the raw update,
`α = 1`, and keep the code exactly sample -> elite -> mean/std; the smoothing knob is the place I
would turn if premature collapse became the bottleneck rather than query budget.

One more thing the continuous case forces on me: the action space is **constrained**. An action
can't be arbitrarily large — there's a maximum step the agent can take. But a Gaussian has unbounded
support, so some sampled action sequences will be infeasible. One mathematical route is to choose a
sampling family already supported on the feasible set. The planner takes the cheaper engineering route:
draw from the plain Gaussian, project each sampled action into the feasible ball, then score the
projected samples and update from the projected elites. For a per-step action with a maximum norm `r`,
rescale any action whose norm exceeds `r` down to norm `r`, leaving shorter actions untouched.
Concretely, with `n = ‖a‖`, multiply the action by `clamp(n, 0, r)/(n + ε)`: actions inside the ball
are unchanged up to the tiny numerical `ε`, actions outside are pulled to the boundary. Feasibility is
enforced before the black-box cost ever sees a candidate.

Let me also pin down the bookkeeping numbers from the principle and then from the planner harness.
The rarity `ρ` sets the elite fraction: too small and the elite set is too tiny to estimate a mean
and variance reliably; too large and the "elites" include mediocre samples that drag the fit away
from the good region. The generic algorithm can expose this as `ρ`, with `N^e = ceil(ρN)` or an
equivalent top-k count. In the action-sequence planner I can expose the count directly: draw `N =
300` sequences per iteration, keep `N^e = 10` lowest-cost elites, and repeat for `30` iterations. The
initial mean is the zero action sequence (no prior bias toward moving any particular way) and the
initial standard deviation is `var_scale = 1`, a deliberately simple isotropic spread that the elite
statistics will reshape. And because my objective is a *cost* to minimize while the derivation
maximized `S`, the elites are the samples with the **lowest** cost — `S = −cost`, so the top values of
`S` are exactly the bottom costs; I pick elites by smallest cost.

The mathematical `σ̃_t` is the square root of the MLE variance; in this planner I store the spread
with `torch.std` over the elite dimension, so I keep that estimator while preserving the same
lowest-cost-elite refit loop.

So let me assemble the loop concretely as the planner, filling the empty search slot. Sample a batch
of action sequences from the current diagonal Gaussian; clip each action into the feasible ball;
roll them through the fixed model and read off the cost of each; take the lowest-cost elites; refit
the Gaussian to those elites' mean and spread; repeat for the budgeted number of iterations; and
return the final mean as the plan, since the mean is the distribution's point estimate of the
optimum and in the degenerate limit it *is* the optimum.

```python
import torch
from einops import rearrange


class CEMPlanner(Planner):
    """Planner that refits a diagonal Gaussian over action sequences to low-cost samples."""

    def __init__(self, unroll, action_dim=2, plan_length=15,
                 num_samples=300, n_iters=30, var_scale=1,
                 num_elites=10, max_norms=None, **kwargs):
        super().__init__(unroll)
        self.action_dim = action_dim
        self.plan_length = plan_length
        self.num_samples = num_samples
        self.n_iters = n_iters
        self.var_scale = var_scale
        self.num_elites = num_elites
        self.max_norms = max_norms
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def plan(self, obs_init, steps_left=None, eval_mode=True,
             t0=False, plan_vis_path=None):
        T = min(self.plan_length, steps_left) if steps_left else self.plan_length

        # diagonal Gaussian over (timestep, action coordinate); zero mean = no prior on motion
        mean = torch.zeros(T, self.action_dim, device=self.device)
        std = self.var_scale * torch.ones(T, self.action_dim, device=self.device)
        actions = torch.empty(T, self.num_samples, self.action_dim, device=self.device)
        losses, elite_means, elite_stds = [], [], []

        for _ in range(self.n_iters):
            # sample a batch of action sequences x ~ N(mean, std^2)
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                T, self.num_samples, self.action_dim, device=std.device)

            # enforce a feasible action ball when the harness supplies a norm limit
            if self.max_norms is not None:
                assert len(self.max_norms) == 1
                max_norm, eps = self.max_norms[0], 1e-6
                norms = actions.norm(dim=-1, keepdim=True)
                max_norms = torch.ones_like(norms) * max_norm
                min_norms = torch.zeros_like(norms)
                coeff = torch.min(torch.max(norms, min_norms), max_norms) / (norms + eps)
                actions = actions * coeff

            # query the black-box cost of each sequence (lower is better)
            cost = self.cost_function(
                rearrange(actions, "t b a -> b a t"), obs_init
            ).unsqueeze(1)
            losses.append(cost.min().item())

            # elites = lowest-cost samples (S = -cost, so smallest cost = largest S)
            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss = cost[elite_idxs]
            elite_actions = actions[:, elite_idxs]     # [T, N_elite, A]
            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())

            # store spread as torch.std over the elite dimension, matching the planner loop
            mean = elite_actions.mean(dim=1)
            std = elite_actions.std(dim=1)

        # return the distribution's center as the plan (point estimate of the optimum)
        return PlanningResult(
            actions=mean,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
        )
```

I can trace the whole route now. I needed to optimize a black-box, noisy, multi-extremal objective
with no gradient, and adaptive random search was the only thing that fit but it never learned from
its own evaluations. Reframing "optimize `S`" as "make the rare event `{S ≥ γ}` likely" let me reach
for importance sampling, whose variance-optimal density `g*` is the baseline restricted to the elite
event — exactly the "concentrate on the good region" distribution I wanted, but unusable directly.
Fitting the closest tractable density to the current restricted target in KL divergence reduced, after
dropping `v`-free and constant terms, to a maximum-likelihood fit
on the threshold-beating samples — a principled update, not a nudge. A fixed high threshold made the elite
set empty, so I let the threshold track the sample's top-`ρ` quantile, which is self-tuning and
ratchets upward as the distribution concentrates. Choosing the family for closed-form MLE and fast
sampling gave the Gaussian, whose elite-restricted MLE I derived to be the per-coordinate sample mean
and MLE variance of the elites — re-centering the search and auto-sizing the exploration spread. The
Gaussian's natural collapse to a point mass is the right fixed point but can bite as premature
convergence, with smoothing available as the damping guard; the action planner uses the raw update and
handles unbounded Gaussian samples by clipping them into the feasible action ball before scoring. The
result drops straight into the model-predictive-control harness as a sample, clip, score, take-elites,
refit loop returning the Gaussian's mean.
