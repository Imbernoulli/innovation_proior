Let me start from what actually goes wrong. I train a network on task 1, it does well; I train the same network on task 2, it does well on task 2 and falls back to chance on task 1. The whole game is sequential: at time μ I only have the loss `L_μ` in front of me; the data and the loss functions for everything before are gone. But what I really want to minimize is the sum over everything I've ever seen, `L = Σ_μ L_μ`. So I'm forced to descend one summand at a time, and descending `L_μ` drags the shared weights wherever lowering `L_μ` demands — and those same weights were the ones carrying task 1, so `L_1` quietly climbs back up while I'm not looking. This is exactly the failure McCloskey and Cohen pinned down for backprop networks back in 1989, and French in 1999 put his finger on why it's *catastrophic* rather than gradual: the representation is shared and distributed, so any weight change that helps the new task tends to corrupt the old one, and the more I train, the more I overwrite. The obvious fix — throw all the data back in and train jointly — is precisely what I'm not allowed to do; the old data isn't here, and stockpiling it would cost memory growing with the number of tasks.

So I can't change the data and I'd rather not change the architecture. I want to change the *objective*: add something to `L_μ` that discourages the network from clobbering what it already knew. The crude version writes itself — anchor every weight to where it was, `L_μ + Σ_k (θ_k − θ̃_k)²` with `θ̃` the weights at the end of the last task. But that's hopeless, and I can see why before even running it: one global stiffness can't be right. Make it large enough to actually hold task 1 in place and the network is too rigid to learn task 2; make it small enough to learn task 2 and it doesn't hold task 1. The stiffness has to be *per parameter*. Some weights mattered enormously for task 1 and must barely move; others were irrelevant and should be free to absorb task 2. So the real question splits in two: which parameters were important, and how do I turn "important" into a penalty.

What does "important" even mean here? The network is wildly over-parameterized — a task doesn't pin a single solution, it pins a whole low-loss manifold, many weight settings all solving task 1 equally well. That over-parameterization is my opening: somewhere near the task-1 solution there may well be a configuration that *also* solves task 2, and the job is to steer toward that one instead of toward some arbitrary task-2 minimum that happens to wreck task 1. To steer, I need to know, for each weight, how much the old task would suffer if I moved it. Locally that's a curvature statement: expand the old loss around its minimum `θ*`, `L(θ) ≈ L(θ*) + ½(θ−θ*)ᵀ H (θ−θ*)`, and the Hessian `H` tells me exactly how much loss I pay for moving in each direction. Stiff directions — large curvature — are the ones the old task cared about; flat directions are free. And there's a useful empirical fact: in these problems the relevant Hessian tends to be *low rank*, most directions flat, which is the very thing that leaves room for new tasks. So a per-parameter importance is morally a per-parameter curvature, and a curvature-weighted quadratic penalty is the natural form.

There's an existing way to get this per-parameter curvature: the elastic-weight-consolidation route, and it's worth working through because it sets the bar I have to clear. Its derivation is Bayesian and clean. Training a task is finding the most probable parameters given its data, and after task A everything A taught lives in the posterior `p(θ | D_A)`; by Bayes, `log p(θ|D) = log p(D_B|θ) + log p(θ|D_A) − log p(D_B)`, so the new-task loss plus a *prior* that is the old-task posterior. Approximate that posterior as a Gaussian (Laplace) centered at `θ*_A` with diagonal precision equal to the diagonal Fisher information `F`, and the negative-log-prior is a quadratic spring,

  `L(θ) = L_B(θ) + Σ_i (λ/2) · F_i · (θ_i − θ*_{A,i})² ,`

stiff on the weights A cared about (large `F_i`), slack elsewhere. The Fisher `F_i = E[(∂ log p(y|x)/∂θ_i)²]` is positive-semidefinite, equal near a minimum to the curvature, and computable from first-order gradients. This is good. It's the right shape — per-parameter quadratic anchor with curvature stiffness. But stare at how `F` is obtained and something bothers me. It's a *point* estimate evaluated *at the converged endpoint* `θ*_A`, and computing it means a *separate phase after the task is over*: a whole extra sweep that isn't part of training, that throws away the entire trajectory and looks only at the final point. The exact diagonal Fisher also needs a sum over all output labels, so its cost scales with the number of outputs — fine for a few classes, awkward for many. And for a third task I'm storing another Fisher, or folding them, with bookkeeping that grows with the number of tasks. What I'd really like is importance I get *for free, online, while I'm already training* — no extra phase, no extra backward pass, computed locally per parameter from the optimization itself. The question is whether the trajectory carries enough information to supply that.

So let me throw out "evaluate curvature at the endpoint" and ask the question differently. The thing I actually care about is: over the whole course of training task μ, how much did each individual parameter contribute to making that task's loss go *down*? Because the parameters that did the heavy lifting in driving the loss down are exactly the ones I must not let drift back up. And contribution-to-loss-change is something I can read off the trajectory itself, step by step, without any of EWC's after-the-fact machinery.

Make that precise. Picture the training of task μ as a path `θ(t)` through parameter space, from the starting point `θ(t_{μ-1})` to the endpoint `θ(t_μ)`. Take an infinitesimal step `δ(t)` at time `t`. To first order the loss changes by

  `L(θ(t) + δ(t)) − L(θ(t)) ≈ Σ_k g_k(t) δ_k(t) ,   g_k = ∂L/∂θ_k ,`

so each coordinate's little move `δ_k(t)` contributes the amount `g_k(t) δ_k(t)` to the change in total loss right then. Now sum these infinitesimal contributions over the entire trajectory. Writing `δ_k(t) = θ'_k(t) dt`, the total change in loss along the path is the line integral of the gradient field,

  `∫_C g(θ(t)) · dθ = ∫_{t_{μ-1}}^{t_μ} g(θ(t)) · θ'(t) dt .`

The gradient of a scalar loss is a *conservative* vector field — it's literally a gradient — so this line integral doesn't depend on the path; it should equal the difference in the potential between the endpoints, `L(θ(t_μ)) − L(θ(t_{μ-1}))`. During successful descent that number is negative, so its negative is the loss drop. If that's right, it gives me an anchor: whatever I build out of these per-step pieces, summed up it has to reproduce the total loss change with the right sign.

I don't want to take that on faith for the discrete sum I'll actually compute, because the integral identity is a continuous statement and an optimizer takes finite steps. Let me work a case I can run by hand. One parameter, quadratic loss `E(θ) = ½H θ²` with `H = 2` and minimum at `0`, start at `θ₀ = 1` so the true loss drop is `E(1) − E(0) = 1`. Gradient descent with learning rate `η`: `θ ← θ − ηg`, `g = Hθ`. The per-step piece I accumulate is `−g·Δθ` with `Δθ` the actual update, and I sum it to the minimum. With `η = 0.1` I get a running sum of `1.1111`, not `1.0` — it *overshoots* the true drop by 11%. Pushing `η` down: `η = 0.05 → 1.0526`, `η = 0.01 → 1.0101`, `η = 0.001 → 1.0010`. So it's converging to `1.0` as the step shrinks, but only in the limit. The discrete recursion is solvable in closed form: with `θ_n = (1−ηH)^n θ₀`, the running sum is `Σ_n η g_n² = Σ_n η H² θ_n² = ηH²θ₀²/(1−(1−ηH)²)`, which simplifies to `(½Hθ₀²)/(1−ηH/2)`. So the discrete path-integral overestimates the true loss drop by the factor `1/(1−ηH/2)`, exactly. Good — the conservative-field identity is right in the infinitesimal-step limit, and now I know precisely how finite steps inflate it: the overshoot grows with the step size times the curvature. That's a concrete fact I'll need when I worry about noise later, and it already tells me a finite-`η` optimizer biases importance *upward*, never down.

And crucially, that integral *decomposes coordinate by coordinate*, because the dot product is a sum:

  `∫_{t_{μ-1}}^{t_μ} g(θ(t)) · θ'(t) dt = Σ_k ∫_{t_{μ-1}}^{t_μ} g_k(θ(t)) θ'_k(t) dt .`

Each term in that sum is the contribution of one parameter `k` to the total loss change over the whole task. That per-parameter line integral is what I'll call importance,

  `∫_{t_{μ-1}}^{t_μ} g_k(θ(t)) θ'_k(t) dt ≡ − ω_k^μ .`

I put a minus sign there deliberately: I'm interested in loss *decreasing*, and a parameter that drove the loss down has `g_k θ'_k < 0` (it moves against its own gradient under descent), so flipping the sign makes `ω_k^μ` a positive number measuring how much that parameter helped. So `ω_k^μ` is the per-parameter credit for the drop in task μ's loss, accumulated *along the trajectory*, not read off the endpoint. And it's built only from quantities I already have at every step — no separate evaluation phase, which is exactly what the Fisher route forced.

Can I actually accumulate it cheaply during training? The integrand is `g_k(t) · θ'_k(t)` — the gradient times the rate of change of the parameter. At each optimizer step I already have the gradient `g_k`, and the parameter update `Δθ_k = θ_k^{new} − θ_k^{old}` is exactly the discrete stand-in for `θ'_k dt`. So I just keep a running sum, per parameter, of `−g_k · Δθ_k` as training proceeds. One multiply and one add per parameter per step, using numbers the optimizer already computed — and it's exactly the sum I just ran by hand, so I already know it reproduces the loss change in the small-step limit and overshoots by `1/(1−ηH/2)` at finite step. So two distinct things inflate the estimate above the true path integral, and I should keep them separate. The first I've now quantified: finite learning rate, biasing `ω` upward by a factor set by `η` and the local curvature. The second is stochasticity — in practice `g_k` is a minibatch gradient, not the true one. I'd expect the minibatch noise to inflate the estimate too, since the update `Δθ_k` is driven by the same noisy `g_k` and `−g_k·Δθ_k = η g_k²` is a square, so its expectation picks up the gradient-noise variance on top of the true `η ḡ_k²` — a strictly positive extra term. That's an argument, not yet a measurement; I'd want to confirm the size of it on a real run. But both effects push the same way, upward, which already tells me the eventual penalty strength will need to be *dialed down*, not up, to compensate. Hold that; it'll fix the sign of a hyperparameter later.

Now, how do I turn `ω_k^μ` into a penalty? I want the penalty added while training a future task to *re-create*, as far as the descent dynamics can tell, the effect of the unavailable past losses — to pull the weights as if those losses were still present. So let me build a surrogate for the past contribution and add it to the current task loss. The simplest faithful surrogate is a quadratic in `(θ̃_k − θ_k)`, anchored at the reference weights `θ̃_k = θ_k(t_{μ-1})`,

  `L̃_μ = L_μ + c Σ_k Ω_k^μ (θ̃_k − θ_k)² ,`

with some per-parameter strength `Ω_k^μ` I still have to set, and a single scalar `c` trading old against new. What should `Ω_k^μ` be? I want the surrogate to be faithful to the descent in a strong sense: if I had trained on the surrogate quadratic *instead of* the real old loss, I'd want it to end at the same parameters and produce the same per-parameter loss drop over the same motion. Take one task and one parameter and ask what strength makes a quadratic `E_k(θ) = s_k (θ̃_k − θ_k)²` reproduce the right contribution. The motion over the task is `Δ_k = θ_k(t_μ) − θ_k(t_{μ-1})`. The loss drop that quadratic produces over that motion is, up to the constant, `s_k Δ_k²`. I want that to equal the credit `ω_k^μ` the real loss actually accrued. So `s_k Δ_k² = ω_k^μ`, which forces

  `s_k = ω_k^μ / Δ_k² .`

There's my normalization: divide the path-integral importance by the square of how far the parameter actually moved. Accumulating over all past tasks ν < μ,

  `Ω_k^μ = Σ_{ν<μ} ω_k^ν / ((Δ_k^ν)² + ξ) .`

The `Δ²` in the denominator isn't a fudge — it's what makes the surrogate quadratic yield the *same* `ω` over the *same* distance, so descent on the surrogate mimics descent on the true past loss. It does a second nice thing: it fixes the units. `ω` has units of loss; `Δ²` has units of parameter-squared; so `ω/Δ²` times `(θ̃−θ)²` comes out in units of loss, matching `L_μ` exactly, so `c` is a clean dimensionless knob. The `ξ` I bolt on for a concrete reason: if a parameter barely moved over a task, `Δ_k → 0`, and `ω/Δ²` blows up — a weight that sat still gets infinite importance, which is nonsense. The small damping constant `ξ` in the denominator floors that, bounding the expression when `Δ_k → 0`. The reference `θ̃_k` is the weight value at the end of the previous task, and `Ω` and `θ̃` only update at task boundaries; during a task I keep accumulating the running `ω`, and once I've folded it into `Ω` at the end I reset `ω` to zero for the next task. The bookkeeping is now constant in the number of tasks — one running `Ω` per parameter, one reference per parameter, no per-task list.

Now the overestimate I flagged twice pays off in setting `c`. If the path integral were exact, `c = 1` is what "equal weight to old and new" would mean. But I've established that the estimate runs high — by the computed factor `1/(1−ηH/2)` from finite steps, plus the positive variance term from minibatch noise — so at `c = 1` the penalty comes out too stiff and over-constrains the network. The fix is to take `c` below one to absorb the inflation; it's an empirical knob trading old memories against capacity for the new task, and since the noise part grows with problem difficulty, `c` should come down further on noisier or harder problems. So the direction is forced, not guessed: `c = 1` is the in-the-limit value and `c < 1` is the correction for an overestimate I can point to the source of.

One more thing the noise forces. The path integral, estimated with noisy gradients, can come out slightly *negative* for some parameter — the running sum of `−g_k Δθ_k` dips below zero if, over the task, that coordinate net-*increased* its own loss locally (or the noise just pushed it there). A negative final stiffness in the penalty `Ω_k (θ̃_k − θ_k)²` would have the wrong sign — instead of holding the weight near `θ̃`, it would *reward* moving it away, actively encouraging forgetting on that coordinate. That's clearly not what I want; importance should be nonnegative. The clean place to floor is the running stiffness after adding the new increment, `Ω_k ← ReLU(Ω_k + W_k/(Δ_k²+ξ))`, not necessarily the increment alone. Then a noisy negative increment can correct an earlier overestimate, but the stored stiffness can never cross below zero. If the surrounding loop only asks my function for the current task's normalized increment and separately sums those returns across tasks, that function should return the raw `W_k/(Δ_k²+ξ)`; the floor belongs at the boundary where the running stiffness is updated.

I should pressure-test whether this thing is actually measuring curvature, the way EWC's Fisher claims to — or whether I've just built a plausible-looking running sum. Let me work the one case I can solve exactly: a quadratic loss. Put

  `E(θ) = ½ (θ − θ*)ᵀ H (θ − θ*) ,`

minimum at `θ*`, constant Hessian `H`. Run continuous gradient descent on it. The dynamics are

  `τ dθ/dt = − ∂E/∂θ = − H (θ − θ*) ,`

with `τ` setting the timescale (it's just the inverse learning rate). This is a linear ODE; its solution from initial `θ(0)` is

  `θ(t) = θ* + e^{−H t/τ} (θ(0) − θ*) ,`

which I can check by differentiating: `dθ/dt = −(1/τ) H e^{−Ht/τ}(θ(0)−θ*) = −(1/τ)H(θ(t)−θ*)`, matching the ODE. So the velocity along the path is

  `θ'(t) = dθ/dt = − (1/τ) H e^{−H t/τ} (θ(0) − θ*) .`

Now I have to keep the sign straight. The line-integral contribution is `∫ g_k θ'_k dt`, but the importance is the *negative* of that contribution, `ω_k = -∫ g_k θ'_k dt`, because descent makes `g_k θ'_k` negative. Under gradient descent the ODE says `τ dθ/dt = -g`, so `-g_k θ'_k = τ (dθ_k/dt)(dθ_k/dt)`. The per-parameter `ω` are therefore the diagonal entries of the positive matrix

  `Q = τ ∫_0^∞ (dθ/dt)(dθ/dt)ᵀ dt .`

Let me compute `Q` in the eigenbasis of `H`. Let `λ^α, u^α` be the eigenpairs, and let `d^α = u^α · (θ(0) − θ*)` be the component of the initial displacement along eigenvector `α`. Then `e^{−Ht/τ}(θ(0)−θ*) = Σ_α e^{−λ^α t/τ} d^α u^α`, so `dθ/dt = −(1/τ) Σ_α λ^α e^{−λ^α t/τ} d^α u^α`. Form the outer product and integrate term by term; the time integral of `e^{−(λ^α+λ^β)t/τ}` from 0 to ∞ is `τ/(λ^α+λ^β)`. Carrying the `(1/τ)²` from the two velocities and the leading `τ`, the prefactor on the `(α,β)` term is `τ · (1/τ²) · λ^α λ^β · τ/(λ^α+λ^β) = λ^α λ^β/(λ^α+λ^β)`. So

  `Q_ij = Σ_{αβ} u_i^α d^α (λ^α λ^β / (λ^α + λ^β)) d^β u_j^β .`

Notice `τ` has dropped out entirely — `Q` is a steady-state, time-integrated quantity, independent of how fast I descend, which is reassuring (importance shouldn't depend on the learning rate). But `Q` still depends in a tangled way on the eigenvectors, the eigenvalues, *and* the particular initial condition through the `d^α`. Does it relate to the Hessian? Let me first average over random initial conditions, where the displacement components are zero-mean i.i.d. with variance `σ²`, so `⟨d^α d^β⟩ = σ² δ_{αβ}`. The off-diagonal `αβ` terms vanish in the average, and on the diagonal `λ^α λ^α/(λ^α+λ^α) = λ^α/2`, giving

  `⟨Q_ij⟩ = ½ σ² Σ_α u_i^α λ^α u_j^α = ½ σ² H_ij ,`

since `Σ_α u_i^α λ^α u_j^α` is just the spectral expansion of `H`. So *on average, the path-integral matrix reduces to one half of the Hessian, up to the displacement scale `σ²`*. The half is not a nuisance I can ignore; it is exactly the half in the loss drop of a quadratic bowl, `½ dᵀHd`. And look what the `Δ²` normalization does here — at zero damping the denominator `Δ_k²` averages to `σ²` (the squared net motion in a coordinate), so dividing by it removes the trajectory scale and leaves an average coefficient `Ω_k = ½ H_kk`. Because I write the consolidation penalty as `Ω_k(θ_k−θ̃_k)²`, with no leading `½`, that coefficient has local curvature `2Ω_k = H_kk`. So the normalization I introduced to make the surrogate reproduce the loss drop is also the thing that turns the path integral into the right quadratic curvature convention.

I should check what survives for a *single* initial condition, since in practice I don't average. I have to be precise about what I store: the algorithm only keeps the diagonal entries `Q_kk` as per-parameter importances, not the whole matrix. If the Hessian is diagonal, `u_i^α = δ_{αi}`, so the stored diagonal entry has only `α = β = i`, and

  `Q_ii = ½ (d^i)² H_ii .`

So the `Δ²` normalization removes `(d^i)²` and leaves `Ω_i = ½ H_ii`, again giving the quadratic penalty curvature `H_ii`. The full off-diagonal entries are not the object I store; for `i ≠ j` they are `d^i d^j H_ii H_jj/(H_ii+H_jj)` for one particular trajectory, so the clean claim is about the diagonal entries that become `ω_i`. This is a strong enough claim that I should run it once rather than trust the algebra blind, especially since it's a *single*-trajectory statement with no averaging. Take a diagonal Hessian with entries `(0.5, 2, 8)`, start at `(1, −0.7, 0.3)`, accumulate `−g·Δθ` per coordinate down to the minimum with a small step, then form `ω_k/Δ_k²`. I get `(0.25, 1.00, 4.03)` against `½H_kk = (0.25, 1.0, 4.0)`. The first two coordinates land essentially exactly; the third sits a touch high — and that's the finite-step overshoot I already characterized, which scales with `ηH` and so bites hardest on the stiffest coordinate, exactly the one that overshoots here. So the curvature correspondence is real and the only deviation is the one bias I've already accounted for. That's the confirmation I wanted; it isn't just a plausible running sum. Second, the rank-1 case, `λ^1` the only nonzero eigenvalue. Then only `α = β = 1` contributes and `λ^1 λ^1/(λ^1+λ^1) = λ^1/2`, giving

  `Q_ij = ½ (d^1)² u_i^1 λ^1 u_j^1 = ½ (d^1)² H_ij ,`

so the full path-integral matrix is the low-rank Hessian times the same half-displacement scale. Its diagonal entries are `Q_ii = ½(d^1)² H_ii`. If I then divide literally by the coordinate motion `Δ_i² = (d^1 u_i^1)²` at zero damping, the eigenvector factor cancels on coordinates that moved and I get `λ^1/2`; with damping the cancellation is softened. So the rank-1 statement I should trust is the matrix statement about `Q`, not an exact per-coordinate normalized-Hessian identity. The useful point remains: the trajectory integral concentrates on the active low-rank curvature direction while the rest of parameter space is flat, which is precisely the interesting continual-learning geometry. For a general loss with a Hessian that varies along the trajectory I shouldn't expect any exact correspondence between importance and the endpoint Hessian; but the quadratic analysis tells me the importance is a sensible, curvature-flavored quantity, computed over the whole path rather than at one point.

That last clause is the deepest contrast with EWC, and the quadratic case makes it sharp. EWC's stiffness is the (empirical) Fisher `F̄ = E[g gᵀ]`, evaluated *at the endpoint* `θ*`. But at a minimum the gradient vanishes, so for a quadratic the empirical Fisher *at the minimum is zero* — it has thrown away all the curvature information by the time it looks. The path integral, by contrast, accumulated curvature-flavored information *along the way*, while the gradients were still nonzero. So the two measures genuinely differ: the endpoint Fisher sees a flat point and reports nothing; the trajectory integral remembers the curvature it descended through. And it does so with no extra backward pass — the empirical Fisher needs gradients computed with labels sampled from the model (or at least the data) in a separate evaluation, whereas my `ω` reuses the gradients training already produced.

Now I want this faithful for more than two tasks, and the surrogate-loss argument I gave is, strictly, a two-task argument — for the first past task, the quadratic with strength `ω/Δ²` exactly reproduces the descent. For three or more I'm summing such quadratics, `Ω_k = Σ_{ν<μ} ω_k^ν/((Δ_k^ν)²+ξ)`, and a sum of quadratic springs is itself one quadratic spring with summed stiffness, so structurally it stays a single penalty with constant memory; I don't have a clean exactness proof beyond two, but the construction extends mechanically and the per-parameter credit just keeps accumulating. That's the right tradeoff: I'd rather have one running `Ω` than a perfect-but-unbounded per-task ledger.

Let me also be clean about *when* each piece updates, because the timing is what keeps it online and local. The reference `θ̃_k` is snapshotted at the end of a task. During a task, every optimizer step I add `−g_k · Δθ_k` into a running accumulator `W_k` (this is the discrete path integral). At the end of the task I form the raw increment `W_k/(Δ_k² + ξ)` with `Δ_k = θ_k^{end} − θ̃_k` the net motion over the task, add it into the running `Ω_k`, and floor the running result at zero. That order matters: `Ω_k ← ReLU(Ω_k + W_k/(Δ_k²+ξ))`, so a negative noisy increment can reduce a previously overestimated importance but cannot make the final stiffness negative. Then I snapshot a new `θ̃_k` and zero out `W_k` and the per-step machinery for the next task. The penalty `c Σ_k Ω_k (θ_k − θ̃_k)²` is added to the loss at every step of every subsequent task and costs only a sum over parameters — no forward pass through an old network, no extra gradients, no growth with task count.

So let me write it as the two slots the training loop exposes — one that runs once per task to produce importance, one that runs every step to produce the penalty. The per-step path-integral accumulation (`W_k += −g_k Δθ_k`) is maintained by the loop and handed to me as the accumulated `W`; the net motion `Δ_k` is the current parameter minus the snapshot from the start of this task. When I control the optimizer protocol directly, I form `W` with the unregularized task gradient, normalize it at the task boundary, apply the running nonnegative floor, refresh the reference weights, and reset `W`. When the loop handles cross-task summation outside this function, the function returns the normalized increment.

```python
import torch


def estimate_importance(model, dataset, prev_params, device):
    """Run once after a task finishes; return the normalized path-integral increment."""
    epsilon = getattr(model, 'epsilon', 0.1)     # damping ξ: bounds ω when Δ_k -> 0
    omega = {}
    W = getattr(model, '_custom_W', {})          # accumulated Σ -g·Δθ over the task

    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                theta = p.detach().clone()
                theta_ref = prev_params.get(n, theta)        # θ̃: snapshot at task start
                delta = theta - theta_ref                    # Δ_k: net motion over the task
                w = W.get(n, torch.zeros_like(theta))
                omega[n] = w / (delta ** 2 + epsilon)        # W/(Δ^2+ξ)

    return omega


def compute_regularization_loss(model, importance_dict, prev_params_dict):
    """Called every training step; must be cheap. Quadratic consolidation penalty
    Σ_k Ω_k (θ_k - θ̃_k)^2 anchored at the previous task's reference weights θ̃."""
    losses = []
    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                if n in importance_dict and n in prev_params_dict:
                    omega = importance_dict[n]               # Ω_k: consolidation strength
                    ref = prev_params_dict[n]                # θ̃_k: reference weights
                    losses.append((omega * (p - ref) ** 2).sum())
    if losses:
        return sum(losses)                                   # scaled by c (reg_strength) outside
    return torch.tensor(0.0, device=next(model.parameters()).device)
```

Sequential training forgets catastrophically because descending the current task's loss moves shared weights that encoded earlier tasks, and the old data is gone so I can't just retrain jointly. The fix is a per-parameter quadratic penalty anchored at where each weight was — but a uniform stiffness can't work, so the stiffness must be each parameter's *importance* for past tasks. Importance is morally curvature (how much loss is paid for moving a weight), and the curvature route already taken — a diagonal-Fisher spring evaluated at the converged endpoint in a separate phase — throws away the trajectory and costs an extra pass that grows with the output count. So I redefined importance as each parameter's contribution to the loss *decrease over the whole training path*, the per-coordinate piece of the line integral of the gradient against the parameter velocity; because the gradient field is conservative this integral is the net loss drop, and I can accumulate its per-parameter pieces online as a running sum of `−g·Δθ`, for free, from gradients training already computes. To turn importance into a penalty I demanded a quadratic surrogate that reproduces the same loss drop over the same motion, which forced dividing the path integral by `Δ²` (and adding a damping `ξ` so a stationary weight doesn't get infinite importance) — and that very normalization also fixes the units and, on a quadratic, strips the displacement scale so the no-`½` penalty has the right Hessian curvature. Both finite learning rate (by the computed factor `1/(1−ηH/2)`) and minibatch noise inflate the running estimate above the true integral, so the strength `c` is taken below its in-the-limit value of one, and the noise can flip an increment's sign, so I floor the running stiffness at zero after adding each task's increment. Accumulating importance as a running sum across tasks keeps memory constant. The whole thing drops into the training loop as one per-step accumulation, one end-of-task normalization, and a cheap quadratic penalty.
