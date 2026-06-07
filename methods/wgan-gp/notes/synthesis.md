# Synthesis — WGAN-GP (gradient penalty)

## Pain point at the time

GAN training is unstable. The minimax game
  min_G max_D  E_{x~Pr}[log D(x)] + E_{x̃~Pg}[log(1-D(x̃))]
implicitly minimizes the Jensen-Shannon divergence between Pr and Pg when D is trained to optimality. But when Pr and Pg are supported on low-dimensional manifolds that don't perfectly overlap (the generic case in high-dim image space), JS is locally constant / saturated → the discriminator's gradient w.r.t. the generator vanishes. Arjovsky & Bottou (2017, "Towards principled methods") diagnosed this: a good discriminator gives the generator no usable gradient. The `-log D` trick (non-saturating loss) helps but can misbehave.

## WGAN (Arjovsky, Chintala, Bottou 2017) — the immediate ancestor

Replace JS with the Earth-Mover / Wasserstein-1 distance:
  W(Pr, Pg) = inf_{γ∈Π(Pr,Pg)} E_{(x,y)~γ} [ ||x - y|| ]
where Π is the set of couplings (joint distributions with marginals Pr, Pg). W is continuous and differentiable a.e. in the generator's parameters even when supports don't overlap — it measures *how far* you'd have to move mass, so it gives meaningful gradients everywhere.

The primal inf-over-couplings is intractable. Kantorovich-Rubinstein duality converts it to:
  W(Pr, Pg) = sup_{||f||_L ≤ 1}  E_{x~Pr}[f(x)] - E_{x~Pg}[f(x)]
the supremum over all 1-Lipschitz functions f. So WGAN parameterizes f as a neural net (the "critic", not a classifier — no sigmoid), and trains:
  min_G max_{D∈1-Lip}  E_{x~Pr}[D(x)] - E_{x̃~Pg}[D(x̃)]
The critic's gradient w.r.t. its input is better behaved than a GAN discriminator's, and the value correlates with sample quality.

**The hard part: enforcing ||D||_L ≤ 1.** WGAN clips every weight into [-c, c] after each update. This keeps the function in *some* k-Lipschitz class (k depends on c and architecture). It works but is a "terrible way to enforce a Lipschitz constraint" (their own words).

## Pathologies of weight clipping (the motivating diagnostics — these are pre-method facts)

1. **Capacity underuse.** Clipping biases the critic toward very simple functions. Trained to optimality on toy distributions (8 Gaussians, 25 Gaussians, swiss roll) with Pg held fixed at real-data-plus-Gaussian-noise, the clipped critic learns extremely simple value surfaces that ignore higher moments of the data. (Compare: the gradient-penalty critic captures the structure.) The optimal critic attains its maximum gradient norm k everywhere, so under a clipping box the network pushes weights to the extremes (Fig 1b right: weights pile up at ±c, bimodal).

2. **Exploding / vanishing gradients.** The interaction between the weight constraint and the loss makes gradients through depth either explode or vanish without delicate tuning of c. On the swiss roll with a 12-layer ReLU MLP critic, clipping at c ∈ {1e-1, 1e-2, 1e-3} gives gradient norms (of the critic loss w.r.t. successive layer activations) that grow or decay exponentially with depth (Fig 1b left). c is a knife-edge: a bit too large → explode; a bit too small → vanish. Batch norm in the critic helps somewhat but very deep clipped critics still fail to converge.

## The key theoretical fact that motivates the penalty (Proposition 1 + Corollary 1)

Setting: Pr, Pg on a compact metric space X. There exists an optimal 1-Lipschitz critic f* solving the KR dual. Let π be the optimal coupling. Then:

**Proposition 1.** If f* is differentiable, π(x=y)=0, and x_t = (1-t)x + t y for 0≤t≤1, then
  P_{(x,y)~π}[ ∇f*(x_t) = (y - x_t)/||y - x_t|| ] = 1.

**Corollary 1.** f* has gradient norm 1 almost everywhere under Pr and Pg.

So the *optimal* critic isn't merely Lipschitz with norm ≤ 1; it has gradient norm *exactly* 1 — and specifically along the straight lines connecting each coupled pair (x, y), the gradient points in the unit direction (y - x_t)/||y - x_t|| from the line toward y. This is the load-bearing insight: a soft constraint that pushes ||∇D||=1 (rather than ≤ 1) doesn't over-constrain, because the optimum already has unit-norm gradients almost everywhere; and the natural place to enforce it is precisely on the interpolation lines between real and fake.

### Proof of Proposition 1 (worked, Appendix A)

By KR theory (Villani Thm 5.10): on compact X an optimal f* exists (part iii), and for the optimal coupling π,
  P_{(x,y)~π}[ f*(y) - f*(x) = ||y - x|| ] = 1.    (1-Lipschitz must be tight on coupled pairs)

Fix (x,y) with f*(y) - f*(x) = ||y - x||, x≠y (true π-a.s.). Define ψ(t) = f*(x_t) - f*(x), where x_t = (1-t)x + ty. Claim ψ(t) = t||x-y||.

- ψ is ||x-y||-Lipschitz in t: |ψ(t)-ψ(t')| = |f*(x_t)-f*(x_{t'})| ≤ ||x_t - x_{t'}|| = |t-t'| ||x-y||.
- ψ(1)-ψ(0) = (ψ(1)-ψ(t)) + (ψ(t)-ψ(0)) ≤ (1-t)||x-y|| + t||x-y|| = ||x-y||.
- But ψ(1)-ψ(0) = f*(y)-f*(x) = ||x-y|| (and ψ(0)=0), so all the inequalities are equalities ⇒ ψ(t) = t||x-y||, i.e. f*(x_t) = f*(x) + t||x-y||.

Now let v = (y - x_t)/||y - x_t||. Compute it: y - x_t = y - ((1-t)x + ty) = (1-t)(y-x), and ||y-x_t|| = (1-t)||y-x|| (for t<1), so v = (y-x)/||y-x||. The directional derivative:
  ∂f*/∂v (x_t) = lim_{h→0} [ f*(x_t + h v) - f*(x_t) ] / h.
Note x_t + h v = x + (t + h/||y-x||)(y - x) = x_{t + h/||y-x||}. Using f*(x_s) = f*(x) + s||y-x||:
  = lim_{h→0} [ (f*(x) + (t + h/||y-x||)||y-x||) - (f*(x) + t||y-x||) ] / h
  = lim_{h→0} [ h ] / h = 1.
So the directional derivative along v is 1.

Since f* is 1-Lipschitz and differentiable at x_t, ||∇f*(x_t)|| ≤ 1. Decompose ∇f* into its component along v and perpendicular:
  1 ≤ ||∇f*(x_t)||² = ⟨v, ∇f*⟩² + ||∇f* - ⟨v,∇f*⟩v||²
                    = |∂f*/∂v|² + ||∇f* - v ∂f*/∂v||²
                    = 1 + ||∇f*(x_t) - v||²   (since ∂f*/∂v = 1)
                    ≤ 1.   (since ||∇f*|| ≤ 1)
Both ends equal 1 ⇒ ||∇f*(x_t) - v|| = 0 ⇒ ∇f*(x_t) = v = (y - x_t)/||y - x_t||. □
Corollary 1 follows: gradient norm = ||v|| = 1 a.e.

(The first "1 ≤ ||∇f*(x)||²" line uses that the directional derivative being 1 forces ||∇f*|| ≥ 1, combined with Lipschitz ≤ 1 — together with Pythagoras it pins ∇f* exactly to v.)

## The method: gradient penalty (WGAN-GP)

A differentiable function is 1-Lipschitz iff ||∇f|| ≤ 1 everywhere. Enforcing that everywhere is intractable; instead enforce a *soft* version as a penalty on the gradient norm at sampled points. New critic objective (to minimize over critic params; note sign — it's the negative of the value being maximized, plus penalty):

  L = E_{x̃~Pg}[D(x̃)] - E_{x~Pr}[D(x)] + λ E_{x̂~P_x̂}[ (||∇_x̂ D(x̂)||_2 - 1)² ].

- **Sampling distribution P_x̂**: sample x̂ = ε x + (1-ε) x̃ with ε~U[0,1], x~Pr, x̃~Pg. I.e. uniformly along straight lines between real and fake pairs. Motivated by Prop 1: the optimal critic has unit-norm gradients exactly on these lines. Enforcing everywhere is intractable; enforcing on these lines is sufficient and works.
- **Two-sided penalty (||·||-1)²** not one-sided (max(0,||·||-1))²: push toward 1, not just below 1. Doesn't over-constrain because the optimum genuinely has norm-1 gradients almost everywhere. Empirically slightly better; one-sided also works.
- **λ = 10**: works across toy → ImageNet CNNs.
- **No critic batch norm**: BN maps a batch→batch (correlates examples); the penalty is defined per-example (one input → its own gradient norm), so BN invalidates it. Use layer normalization (Ba et al. 2016) as drop-in instead — normalizes per-example, no cross-example correlation.
- **n_critic = 5, Adam(α=1e-4, β1=0, β2=0.9)**: critic trained closer to optimality per generator step (need a good critic to get a good W estimate / gradient). β1=0 because the loss is non-stationary / adversarial; momentum on the first moment hurts. (Toy code uses β1=0.5, λ=0.1 for speed; the recommended defaults are β1=0, λ=10.)

## Why each piece (design-decision → reason)

| Decision | Why / alternative rejected |
|---|---|
| Penalize ||∇D|| instead of clip weights | Clipping → capacity underuse + exploding/vanishing gradients; gradient is the direct object the Lipschitz constraint controls. |
| Target norm = 1 (not ≤1) two-sided | Optimal critic has unit-norm gradients a.e. (Prop 1/Cor 1) so it doesn't over-constrain; one-sided also works, two-sided slightly better empirically. |
| Sample on interpolation lines real↔fake | Prop 1: optimal critic's gradient is unit-norm exactly along these lines; enforcing everywhere intractable, here sufficient. |
| ε ~ U[0,1] | Uniform along the whole segment, covering the region between the distributions where it matters. |
| λ = 10 | Robust across architectures/datasets. |
| Omit critic BN, use LayerNorm | GP is per-example; BN correlates examples across the batch, breaking the per-input gradient-norm interpretation. LayerNorm normalizes per-example. |
| Adam β1=0 | Adversarial/non-stationary objective; first-moment momentum destabilizes. |
| n_critic=5 | A near-optimal critic gives a meaningful W estimate and good generator gradients. |
| Smooth activation (softplus(2x+2)/2 - 1 in place of ELU) | GP gradient involves 2nd derivatives of activations; non-smooth ones (ELU) gave undefined Hessians/training failure; ReLU/leaky-ReLU empirically fine despite measure-zero kinks. |

## Canonical implementations
- Official TF: github.com/igul222/improved_wgan_training — gan_toy.py (cleanest GP demo, fetched to code/). interpolates = alpha*real + (1-alpha)*fake; gradients = tf.gradients(D(interpolates), interpolates); slopes = sqrt(sum(grad²)); penalty = mean((slopes-1)²); disc_cost += LAMBDA*penalty.
- Canonical PyTorch reimpl: github.com/caogang/wgan-gp — calc_gradient_penalty using torch.autograd.grad(..., create_graph=True). Final answer.md code grounded in this PyTorch structure.

## Out of scope (experiments — skip)
Inception-score tables, LSUN/CIFAR sample grids, language-model samples, 200-random-architectures success counts. (Diagnostic toy-distribution observations about *clipping* ARE in scope as motivation.)
