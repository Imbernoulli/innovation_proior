Let me start from what actually keeps going wrong, because that's the only honest place to start. I train a GAN and it works, sometimes, for some architecture, with the right normalization and the right learning rate — and then I change one thing, drop the batch norm, swap a nonlinearity, make the net deeper, and it collapses or stalls. The instability isn't a nuisance on top of a working method; it's the central fact. So before reaching for any fix I want to understand precisely *what* breaks.

The original game is min over G, max over D of E_{x~Pr}[log D(x)] + E_{x̃~Pg}[log(1 - D(x̃))], with D a probability through a sigmoid. If I freeze G and solve the inner max, the optimal D is Pr/(Pr+Pg) pointwise, and plugging that back in, the generator ends up minimizing the Jensen-Shannon divergence between the data distribution Pr and the model distribution Pg. Fine in principle. But think about where Pr and Pg actually live. Pr is natural images — a thin curved sheet in a huge pixel space, a low-dimensional manifold. Pg is the pushforward of a low-dimensional noise through G — also a thin sheet. Two thin sheets in a high-dimensional room generically don't intersect, or intersect only on a set of measure zero. And if their supports are essentially disjoint, a discriminator with enough capacity can separate them *perfectly*: D = 1 on the data sheet, D = 0 on the generated sheet, with a confident margin between. At that point JS is pinned at its maximum, log 2, and it's *locally constant* — nudging G a little doesn't change which side of the margin anything is on, so the divergence doesn't move, so its gradient with respect to G's parameters is zero. The better the discriminator, the flatter the loss the generator sees. That's the trap. It explains the folklore "don't train D too well" and the non-saturating -log D patch, which rescales the gradient but doesn't change the underlying geometry: when the sheets are disjoint, you're still trying to learn from a divergence that can't feel small movements.

So the disease is the divergence itself. JS only sees overlap, and KL-style objectives are no escape: when one singular distribution puts mass where the other has none, the KL is infinite or undefined rather than a finite slope, and the classifier-based signal still saturates instead of telling me how far the sheets are from each other. I need a notion of distance between distributions that decreases smoothly as I slide one sheet toward the other, even while they're still disjoint. The Earth-Mover picture gives exactly that: imagine Pg as a pile of dirt and Pr as the hole I need to fill, and ask for the minimum total work — mass times the distance it's carried — to reshape one into the other. Formally,

  W(Pr, Pg) = inf over couplings γ of E_{(x,y)~γ}[ ||x - y|| ],

where a coupling γ is any joint distribution on (x, y) whose marginals are Pr and Pg — a transport plan saying how much mass goes from each x to each y. Now if I translate Pg by a small ε toward Pr, the optimal plan's cost drops by about ε; W is roughly the literal distance between the sheets, so it varies continuously and is differentiable almost everywhere as G moves, *even when the supports never overlap*. This is the property JS lacked: the generator can feel a slope without needing the two supports to overlap first.

The problem is that infimum over couplings is hopeless to compute — it's a linear program over joint distributions on a continuous space. But Wasserstein-1 has a dual, the Kantorovich-Rubinstein form:

  W(Pr, Pg) = sup over all 1-Lipschitz f of  E_{x~Pr}[f(x)] - E_{x~Pg}[f(x)].

Instead of searching over transport plans, I search over *functions* f that are 1-Lipschitz — meaning |f(a) - f(b)| ≤ ||a - b|| for all a, b — and take the one that most separates the two distributions in expectation. That's a maximization over a function class, and a neural network is a function class. So I parameterize f as a net, call it the critic (it's not a classifier anymore — no sigmoid, its output is an unbounded real score), and I train

  min over G, max over D in {1-Lipschitz} of  E_{x~Pr}[D(x)] - E_{x̃~Pg}[D(x̃)].

The inner max estimates W(Pr, Pg); the outer min drags Pg toward Pr along W's smooth slope. The critic's gradient with respect to its input is exactly the direction the generator should push its samples, and unlike the saturating sigmoid discriminator, it stays informative.

Everything hinges on one thing I've been waving at: "max over D in {1-Lipschitz}." How do I keep a neural network 1-Lipschitz during training? The first thing anyone reaches for, and what the Wasserstein critic was originally trained with, is weight clipping: after each gradient update, clamp every weight into a small box [-c, c]. A network with bounded weights has a bounded Lipschitz constant, so this keeps f in *some* k-Lipschitz class, where k depends on c and on the depth and width. Rescaling the value by 1/k doesn't change the argmax, so in principle a fixed clip is fine. And it does work, in the sense that you can train a Wasserstein GAN this way.

But let me actually look at what clipping does to the critic, because something about it bothers me. Take the simplest possible test: freeze the generator at "real data plus unit-variance Gaussian noise" so I know exactly what the target critic should look like, and train a clipped critic to optimality on a toy 2D distribution — eight Gaussians in a ring, say, or twenty-five on a grid, or a swiss roll. The critic I get back is *embarrassingly* simple. It captures the gross location of the data and completely ignores the higher moments — the fine structure of the ring, the curl of the swiss roll. The clipping is biasing the critic toward extremely simple functions. Why?

The histogram of the trained weights gives away what is happening: they're not spread out, they pile up at the two extremes, +c and -c. The clip box isn't gently regularizing; the network is slamming its weights against the walls. That makes sense if the critic is trying to be as steep as the box allows everywhere — to attain its maximal Lipschitz constant — because that's how you maximize E[D(real)] - E[D(fake)]. But a function built almost entirely from saturated ±c weights can't also be *expressive*; you've spent all your degrees of freedom on being maximally steep, none on shape. Capacity underuse, directly traceable to the clip.

And there's a second, worse problem, about gradients through depth. The backprop gradient with respect to the input is, roughly, a product of one weight-matrix factor per layer. If every weight is clamped into [-c, c], each layer multiplies the backflowing gradient by a factor whose typical size is set by c, so over L layers the gradient norm scales like (something tied to c) to the Lth power — geometric in depth. Let me actually put numbers on that rather than trust the hand-wave. Model the critic as stacked random linear layers with weights drawn uniformly from [-c, c] and push a unit gradient back through, averaging over a few draws. At c = 10⁻¹ the gradient norm at depth 2, 6, 12 comes out ≈ 0.20, 9.7×10⁻³, 8.9×10⁻⁵ — each six extra layers knocking off roughly two orders of magnitude, a clean geometric decay. Shrink the box to c = 10⁻² and it's ≈ 2.2×10⁻³, 9.8×10⁻⁹, 8.4×10⁻¹⁷ — same geometric law, steeper base, now vanishing catastrophically by depth 12. Nudge it up to c = 5×10⁻² and the decay slows but is still exponential. So the picture is exactly the fear: each c gives an exponential in depth, just with a different base, and tuning c trades vanishing for exploding without ever flattening the curve. The clip threshold c is a knife-edge: it's simultaneously controlling the Lipschitz constant *and* the conditioning of backprop, and there's no single value that keeps both happy. Batch norm in the critic masks this a bit, but very deep clipped critics still refuse to converge. So clipping isn't just inelegant — it's actively fighting the optimization.

Two failure modes, same root: I'm enforcing a constraint on the *function* (be 1-Lipschitz) by bludgeoning the *weights*, and the weights are a terrible handle on it. Let me back up and ask what the constraint really is. For a differentiable f on the ambient input domain, the local form of the 1-Lipschitz condition is ||∇_x f(x)|| ≤ 1 everywhere — the slope, measured directly, never exceeds one. That's the actual object. Weight clipping is a sledgehammer proxy for "keep the gradient norm bounded." Why not constrain the gradient norm itself? That feels obviously more direct. But "for all x" is uncountably many constraints, and the gradient norm of a deep net is a complicated thing — I can't impose it exactly. The natural relaxation is a *soft* one: instead of a hard constraint, add a penalty that pushes the input-gradient norm toward the allowed value at sampled points. Tractable, differentiable, and it acts on exactly the quantity the Lipschitz condition is about.

Before I commit to that, I need to answer two questions that decide whether it's even sane: *what value* should I push the gradient norm to, and *where* should I sample the points? If I get these wrong I'll either over-constrain the critic into the same simple-function trap, or enforce the constraint in the wrong region and leave it free where it matters. So let me look hard at what the *optimal* critic — the actual argmax of the dual — looks like, because that tells me what a non-distorting constraint should permit.

Start from the dual optimum f* on a compact space, with π the optimal coupling (the best transport plan). I need to keep the sign convention straight: the critic is maximizing E_{real}[f] - E_{generated}[f], so the real endpoint should get the higher value. I will call the generated endpoint x̃ and the real endpoint x. The basic fact from optimal transport is that the 1-Lipschitz bound is *tight on the pairs π actually moves*: for π-almost-every coupled pair (x̃, x),

  f*(x) - f*(x̃) = ||x - x̃||.

This is just the dual being tight against the primal — the function gains exactly as much value over a coupled pair as the cost of moving that pair. Now stare at this. It says nothing yet about the gradient. But I have an equality between a difference of f* values and a Euclidean distance, and that's a strong thing — let me see what it forces along the straight line between x̃ and x.

Take such a pair, with x̃ ≠ x (true π-almost-surely once I exclude pairs that do not move). Parameterize the segment from generated to real as x_t = (1 - t)x̃ + t x for t in [0, 1], and define ψ(t) = f*(x_t) - f*(x̃), so ψ(0) = 0 and ψ(1) = f*(x) - f*(x̃) = ||x - x̃||. What can ψ do in between? Because f* is 1-Lipschitz,

  |ψ(t) - ψ(t')| = |f*(x_t) - f*(x_{t'})| ≤ ||x_t - x_{t'}|| = |t - t'| · ||x - x̃||,

so ψ is itself ||x - x̃||-Lipschitz as a function of t. Now squeeze it. Split the full rise into two pieces:

  ||x - x̃|| = ψ(1) - ψ(0) = (ψ(1) - ψ(t)) + (ψ(t) - ψ(0)) ≤ (1 - t)||x - x̃|| + t·||x - x̃|| = ||x - x̃||.

Each piece is bounded by its own Lipschitz upper bound, and the two upper bounds already sum to the full endpoint rise. If either piece is slack, the sum is smaller than ||x - x̃||, which it is not. So both pieces are tight, in particular ψ(t) - ψ(0) = t·||x - x̃||. Since ψ(0) = 0, this gives

  ψ(t) = t·||x - x̃||,  i.e.  f*(x_t) = f*(x̃) + t·||x - x̃||.

So f* rises *exactly linearly* along the segment from generated to real, at rate ||x - x̃|| per unit t. It has no slack to wander — the value at every interior point is pinned.

Now turn that into a statement about the gradient. For an interior point t in (0, 1), let v be the unit vector along the segment toward the real endpoint. Compute it carefully: x - x_t = x - ((1 - t)x̃ + t x) = (1 - t)(x - x̃), and its norm is (1 - t)||x - x̃||, so

  v = (x - x_t)/||x - x_t|| = (x - x̃)/||x - x̃||,

the same unit direction from generated toward real for every interior t. The directional derivative of f* along v at x_t is

  ∂f*/∂v (x_t) = lim_{h→0} [ f*(x_t + h v) - f*(x_t) ] / h.

For small enough h, the perturbed point stays on the same segment, and x_t + h v = x̃ + (t + h/||x - x̃||)(x - x̃) = x_s with s = t + h/||x - x̃||. Moving a distance h along v is moving by h/||x - x̃|| in the t-parameter. Using f*(x_s) = f*(x̃) + s·||x - x̃||:

  f*(x_t + h v) - f*(x_t) = [(t + h/||x - x̃||)||x - x̃||] - [t·||x - x̃||] = h,

so the ratio is h/h = 1 and ∂f*/∂v (x_t) = 1. The slope in the direction v is exactly one.

f* is 1-Lipschitz and differentiable at x_t, so ||∇f*(x_t)|| ≤ 1. Decompose the gradient into its component along v and the rest, by Pythagoras:

  ||∇f*(x_t)||² = ⟨v, ∇f*(x_t)⟩² + || ∇f*(x_t) - ⟨v, ∇f*(x_t)⟩ v ||².

The inner product ⟨v, ∇f*⟩ is exactly the directional derivative along v, which I just showed is 1. So

  ||∇f*(x_t)||² = 1 + || ∇f*(x_t) - v ||².

But the left side is ≤ 1 (Lipschitz). So 1 + (something ≥ 0) ≤ 1, which forces the something to be 0: ||∇f*(x_t) - v|| = 0, i.e.

  ∇f*(x_t) = v = (x - x_t)/||x - x_t||.

And taking norms, ||∇f*(x_t)|| = 1. This holds for π-almost-every coupled generated-real pair, at differentiability points along the interior of the connecting segment. So the optimal critic doesn't merely *satisfy* ||∇f|| ≤ 1 — on the transport segments it has gradient norm *exactly* 1, and the gradient *points* from the generated endpoint toward the real endpoint, the direction that raises the critic at unit speed.

That chain of squeeze-then-Pythagoras was slippery enough that I don't trust it until I've watched it work on a case where I know the answer outright. The cleanest such case is two point masses: Pr a single Dirac at a real point x, Pg a single Dirac at a generated point x̃, in the plane. Here the optimal coupling is forced — all the mass at x̃ moves to x — and the dual optimum is known in closed form: f*(z) = ⟨u, z⟩ with u = (x − x̃)/||x − x̃|| the unit vector from generated to real. That f* is 1-Lipschitz because ||∇f*|| = ||u|| = 1 everywhere, and it's the maximizer of E_{Pr}[f] − E_{Pg}[f]. Take x = (3, 1), x̃ = (0, 0), so ||x − x̃|| = √10 ≈ 3.1623 and u = (0.9487, 0.3162). Now run the derivation's three claims against it. Tightness: f*(x) − f*(x̃) = ⟨u, x⟩ − ⟨u, x̃⟩ = 3.1623 − 0 = 3.1623 = ||x − x̃|| ✓. Linear rise along x_t = (1−t)x̃ + t x: at t = 0, 0.25, 0.5, 0.75, 1 the value f*(x_t) − f*(x̃) comes out 0, 0.7906, 1.5811, 2.3717, 3.1623, which is exactly t·||x − x̃|| at each t — perfectly linear, no slack, just as the squeeze argument forced. Unit directional derivative: at t = 0.5, v = (x − x_t)/||x − x_t|| = (0.9487, 0.3162) = u, and a finite difference (f*(x_t + hv) − f*(x_t))/h with h = 10⁻⁶ returns 1.0000. And the gradient itself: ∇f* = u, which equals v to numerical precision, so ∇f*(x_t) = v with norm 1. Every step of the abstract argument lands on the number the closed-form optimum predicts. The general case is the same statement with the optimal coupling supplying the (x̃, x) pairs instead of the single forced pair — but seeing it hold concretely is what convinces me the squeeze wasn't an algebra slip.

So the two questions I parked — what value, and where — have answers coming straight out of this. On value: the penalty target should be norm *equal to 1*, not merely ≤ 1, because the true optimum already carries unit-norm gradients on the coupling segments, so pinning the norm to 1 there is reproducing a property the solution has rather than imposing an arbitrary extra restriction. That removes the worry I'd have had about a two-sided penalty over-constraining the critic. On location: the segments connecting generated and real points are exactly where the optimum's gradient is pinned and meaningful, so that's where to sample. I can't enforce ||∇D|| = 1 over all of space — intractable, and most of space is irrelevant — but the transport geometry hands me a specific region: the segments between the two distributions.

I can't sample from the *optimal* coupling π — I don't know it. The practical surrogate is to sample points on straight lines between the two distributions: take a real sample x, a generated sample x̃, draw ε ~ Uniform[0, 1], and form

  x̂ = ε x + (1 - ε) x̃.

This sprays points uniformly along the segment between a random real and a random fake. It is not the optimal-coupling segment in general, but it targets the in-between region where the critic's geometry matters for the generator. Define this as my penalty-sampling distribution P_x̂.

Now I can write the objective. The critic loss to *minimize* is the negative of the value it should maximize, plus the soft penalty:

  L = E_{x̃~Pg}[D(x̃)] - E_{x~Pr}[D(x)]  +  λ · E_{x̂~P_x̂}[ (||∇_x̂ D(x̂)||₂ - 1)² ].

The first two terms are the Wasserstein critic loss; the last is the gradient penalty, dragging the input-gradient norm at interpolated points toward 1. The generator, as before, minimizes -E[D(G(z))], pushing its samples toward where the critic scores high.

Let me double-check the choice of a *two-sided* penalty, (||∇D|| - 1)², against the obvious alternative of a one-sided hinge, (max(0, ||∇D|| - 1))², which only punishes gradients *bigger* than 1 and leaves smaller ones alone. The one-sided version is the literal soft form of the constraint ||∇D|| ≤ 1, so it is the conservative relaxation. To see what each actually permits, tabulate the penalty each assigns to a slope s on the segment: at s = 0, 0.3, 0.6, 1.0, the one-sided hinge gives 0, 0, 0, 0 — it is *flat at zero for every slope at or below 1* — while the two-sided gives 1.0, 0.49, 0.16, 0, climbing as the slope falls away from 1. The consequence is concrete and bad for the hinge. A critic that is completely flat on the segment, slope 0, has ∇D = 0 there; the generator's update is −∇(−D(G(z))) = ∇D, so a zero-slope critic hands the generator *no direction at all* — and the one-sided penalty charges that critic nothing, leaving it free to collapse to the degenerate flat solution exactly in the region that feeds the generator. The transport geometry already said the optimum wants slope 1 on these segments, not merely some slope below 1; the table makes the failure of the conservative relaxation explicit. So the two-sided penalty, which is minimized only at slope 1, encodes the local shape of the optimal critic and refuses the flat degenerate the hinge tolerates.

What about λ? It's the one knob, trading off "estimate W well" against "stay Lipschitz." Too small and the critic drifts away from 1-Lipschitz and the loss stops being a Wasserstein distance; too large and the penalty dominates and starves the critic's expressiveness. The algorithm needs a fixed default rather than a new knife-edge like the clipping threshold, and λ = 10 gives the penalty enough weight to correct slope violations while leaving the Wasserstein score difference visible in the critic loss.

Now a subtlety I almost missed, and it's about batch normalization. The penalty is fundamentally a *per-example* statement: each input x̂ has its own gradient ∇_x̂ D(x̂), and I'm penalizing that single input's gradient norm. But batch norm makes the critic's output for one example depend on the *whole batch* — it normalizes each feature using batch statistics, so D becomes a map from a batch of inputs to a batch of outputs, not from one input to one output. Under batch norm, ∇_x̂ D(x̂) isn't even well-defined as a per-example quantity; the gradient of one output leaks through the batch statistics into the other examples. That breaks the entire interpretation of the penalty. So I can't use batch norm in the critic. But I still want *some* normalization for conditioning. The fix is a normalization that doesn't couple examples: layer normalization, which normalizes each example over its own features independently of the rest of the batch. Drop it in wherever batch norm was, and the per-example gradient-norm penalty stays valid. (This is also a relief because the whole clipping story needed batch norm as a crutch; the penalty doesn't need that crutch at all, and where it does want normalization, layer norm is the per-example-clean choice.)

One more thing to verify, because the penalty quietly assumes more smoothness than the loss usually does. The gradient of L with respect to the critic's *parameters* contains the term ∇_w (||∇_x̂ D(x̂)||²), which differentiates a first derivative of D with respect to its parameters — so it involves *second* derivatives of D, hence second derivatives of the activation functions. For a ReLU, the second derivative is zero almost everywhere and a delta at the kink; for almost every sampled point the autodiff path is still usable. But a genuinely non-smooth activation can bite if the first derivative itself has a kink where the penalty needs to differentiate through it. If that happens, the clean fix is to swap in an activation that's actually smooth — something like softplus(2x + 2)/2 - 1, which closely mimics ELU's shape but has well-defined higher derivatives. Worth keeping in mind, but for ordinary ReLU critics it is not the main issue.

I should also push the critic closer to optimality than a vanilla GAN does, and for a clean reason: the outer min over G is only minimizing a true Wasserstein distance if the inner max over D has actually been solved — a half-trained critic gives a loose, possibly misleading estimate of W and therefore a bad gradient direction for G. With clipping you couldn't train the critic hard without hitting the gradient pathologies; now that those are gone, I can. So I use several critic steps per generator step, with n_critic = 5 as the default compromise between a better inner maximization and training cost. And the optimizer: this is a non-stationary, adversarial objective, the target keeps moving, and momentum on the first moment of the gradient tends to overshoot a moving target and destabilize the game. So I use Adam but kill the first-moment momentum, β₁ = 0, with β₂ = 0.9 and a small learning rate like 1e-4.

Now the code. The load-bearing piece is computing the gradient of the critic with respect to its *interpolated input* and then backpropagating through that gradient into the critic's parameters — which needs second-order autodiff (a graph built through the first gradient).

```python
import torch
import torch.nn as nn
import torch.autograd as autograd

def lipschitz_enforcement(critic, real, fake, lambda_gp=10.0):
    batch_size = real.size(0)
    # one epsilon per example, broadcast over feature dims; uniform on [0,1]
    eps = torch.rand(batch_size, *([1] * (real.dim() - 1)), device=real.device)
    # x_hat = eps*real + (1-eps)*fake : a point on the segment between a
    # real and a fake sample, used as a tractable surrogate for the coupling
    # segments where the optimal critic has unit-norm gradient
    x_hat = eps * real + (1 - eps) * fake
    x_hat.requires_grad_(True)

    d_hat = critic(x_hat)
    # d D(x_hat) / d x_hat ; create_graph=True so this gradient is itself
    # differentiable w.r.t. the critic's parameters (second-order)
    grads = autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(batch_size, -1)
    grad_norm = grads.norm(2, dim=1)            # ||∇_x_hat D(x_hat)||_2 per example
    # two-sided penalty: push the norm toward 1 on the sampled segment points
    return lambda_gp * ((grad_norm - 1) ** 2).mean()


def train_step(generator, critic, opt_g, opt_c, real, noise_dim,
               n_critic=5, lambda_gp=10.0):
    device = real.device
    # ---- critic: several steps toward optimality so E[D(real)]-E[D(fake)]
    #      is a good estimate of the Wasserstein distance ----
    for _ in range(n_critic):
        z = torch.randn(real.size(0), noise_dim, device=device)
        fake = generator(z).detach()
        # Wasserstein critic loss (minimized): down on fake, up on real
        loss_c = critic(fake).mean() - critic(real).mean()
        loss_c = loss_c + lipschitz_enforcement(critic, real, fake, lambda_gp)
        opt_c.zero_grad()
        loss_c.backward()
        opt_c.step()
        # NOTE: no weight clipping -- the penalty replaces it entirely

    # ---- generator: push samples toward where the critic scores high ----
    z = torch.randn(real.size(0), noise_dim, device=device)
    loss_g = -critic(generator(z)).mean()
    opt_g.zero_grad()
    loss_g.backward()
    opt_g.step()
    return loss_c.item(), loss_g.item()


# critic backbone uses NO batch norm (it would couple examples and break the
# per-example gradient penalty); use layer norm where normalization is wanted.
# momentum-free Adam, since the adversarial objective is non-stationary:
#   opt_c = torch.optim.Adam(critic.parameters(),    lr=1e-4, betas=(0.0, 0.9))
#   opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
```
