OK, let me think this through from scratch. The thing that keeps biting me when I train networks is the learning rate. With plain SGD, or with momentum, or with Adam, the single most fragile knob is the step size. Too big and the iterates oscillate in the sharp directions and the loss bounces around or blows up; too small and I wait forever. And it's not a one-time tax — every new architecture, every new dataset, I'm back to sweeping learning rates and weight decays. The two big lines of work people lean on both try to fix this by being smarter about the *gradient*: adaptive methods like AdaGrad and Adam keep a running estimate of the per-coordinate gradient scale and divide by it, and accelerated methods like heavy-ball and Nesterov keep a velocity that accumulates past gradients to push along consistent directions and damp the rest. Both genuinely help. But both are still living inside the same loop — they reshape the step, and they still need that step tuned. I want to ask a different question: is there something I can do that sits *outside* whatever optimizer I'm already running, that doesn't touch how it computes its step at all, but makes the whole trajectory better behaved?

Let me look at where the pain actually comes from, mechanically. In the stochastic setting each step uses a noisy minibatch gradient. So even sitting at the optimum, the iterate doesn't stay there; it gets kicked around by gradient noise. If I crank the learning rate up to make fast progress, those kicks get bigger and the steady-state jitter gets worse. There's a real tension between "move fast toward the solution" and "don't rattle around once you're near it." Let me make that precise on the one model where I can actually compute things — a noisy quadratic. Take L̂(x) = ½(x−c)ᵀA(x−c) with the optimum itself noisy, c ~ N(0, Σ), and A, Σ diagonal so every coordinate evolves on its own with curvature aᵢ and noise variance σᵢ². The expected loss splits cleanly: E[L̂] = ½ Σᵢ aᵢ(E[θᵢ]² + V[θᵢ] + σᵢ²). Three pieces — a bias (how far the mean iterate is from zero), a variance (how much it jitters), and an irreducible floor σᵢ². I can't touch the floor. The interesting fight is bias versus variance.

What does SGD do to each piece? One step is θ ← θ − γ∇ = θ − γA(θ−c). Take expectations over the noise c (mean zero): E[x⁽ᵗ⁺¹⁾] = (I−γA)E[x⁽ᵗ⁾]. So the bias contracts geometrically with factor (I−γA) — and that contracts *faster* the bigger γ is, right up to the stability limit. Good. Now the variance. Since x⁽ᵗ⁺¹⁾ = (I−γA)x⁽ᵗ⁾ + γAc and c has variance Σ, I get V[x⁽ᵗ⁺¹⁾] = (I−γA)²V[x⁽ᵗ⁾] + γ²A²Σ. That recursion doesn't go to zero — it relaxes to a fixed point. Set V* = (I−γA)²V* + γ²A²Σ and solve: V*_SGD = γ²A²Σ / (I − (I−γA)²). And that *grows* with γ. So there it is, written out: large γ kills the bias quickly but inflates the steady-state variance. The learning-rate fragility isn't a tuning artifact, it's this trade-off. If I could knock down the variance term without shrinking γ, I'd get to keep the fast bias contraction *and* a low floor. That's the prize.

How do people knock down variance? The classical answer is averaging the iterates. Ruppert in '88 and Polyak & Juditsky in '92 proved that for stochastic approximation, averaging the iterates gives you the optimal asymptotic variance — much better statistical efficiency than the last iterate — basically because independent-ish noise cancels when you average. And in deep learning this came back as Stochastic Weight Averaging: run SGD with a high or cyclical learning rate, average the weights you visit, and you land in a flatter, wider basin that generalizes better. So averaging is clearly the right hammer. But when I look hard at how SWA and plain Polyak averaging are used, two things bug me. First, they're *equal-weight* arithmetic averages — every visited point counts the same, including stale early ones. Second, and worse for my purposes, they need me to *decide when to start averaging*: start too early and I'm folding in garbage from the beginning of training, start too late and I've averaged too few points to cut the variance. And there's a third, more conceptual thing: the average is a *readout* — a final model I produce at the end — it doesn't feed back into the trajectory that SGD is actually walking. The optimizer never benefits from the smoothing while it's still running.

So let me try to design the averaging I actually want. I want it to (1) run from initialization with no start decision, (2) weight recent proposals more than ancient ones — there's a remark I trust from Martens that an exponentially-decayed moving average works much better in practice than a flat Polyak average, and it makes sense: late iterates are nearer the solution, why give epoch-1 weights equal say — and (3) actually steer the optimizer, not just be read off at the end.

Here's a first attempt. Keep a second set of weights — call them slow weights φ — alongside the ones the optimizer is grinding on, the fast weights θ. Every step, nudge φ a little toward θ: φ ← φ + α(θ − φ) for some small α. That's an EMA of θ, it runs from the start, no start decision, recent-weighted because old contributions decay by (1−α) each step. Good so far. But if I update φ every single step, φ just trails θ by a fixed lag and inherits all of θ's jitter scaled down — and meanwhile θ is off doing its own thing, oblivious. The slow weights aren't influencing the optimization at all; they're exactly the passive readout I was trying to get away from.

So the slow weights have to feed *back*. What if, periodically, I reset the fast weights to the slow weights? Run the inner optimizer for a while from φ, let it explore, then jump φ toward where it ended up, then restart the fast weights from the new φ. Concretely: synchronize θ ← φ, run the inner optimizer A for k steps, θ ← θ + A(L, θ, d) each step on fresh minibatches, then do the interpolation φ ← φ + α(θₖ − φ) and loop. k steps forward on the fast weights, then one step back toward… no — one step of the slow weights *toward* the fast weights. The fast weights go exploring k steps; the slow weights take a single committed step in the direction the exploration found, θₖ − φ.

Let me sanity-check the "feed back" worry on this version. Now φ does drive everything: each new exploration starts from the latest φ, so a good interpolation step changes where the *next* k steps of the inner optimizer begin. It's not a passive trailing average anymore — it's in the loop. And notice what I get for free: the inner optimizer A can be literally anything — SGD, momentum, Adam — because all I use from it is "give me θₖ after k steps." This is the orthogonality I wanted. It's not competing with adaptive or accelerated methods; it wraps them. In code it should be one extra line around an existing optimizer.

Now, why pull back only at the *end* of the k steps, and not every step like some inner/outer schemes do? There's a method, Katyusha — an accelerated SVRG — that keeps an outer checkpoint and pulls the iterate back toward it on *every* inner step, plus an SVRG gradient correction. But pulling back every step would defeat the purpose here: I *want* the fast weights to range out and oscillate, because the whole point is to let them probe and then jump across the oscillation in one clean interpolation. If I yanked them back every step they'd never explore. And the SVRG correction, which gives Katyusha its convex guarantees, is known to behave badly on neural nets, so I'm not going to borrow that. Let me also rule out the most powerful version of "use the exploration": Anderson acceleration and the nonlinear-extrapolation methods keep *all* the inner iterates and solve a little least-squares to find the best linear combination extrapolating to the fixed point. That's strictly more information than I'm using. But it costs memory proportional to k — you have to store all k iterates — and it makes you solve a sub-problem for the combination each time. I'll take the dumbest possible version of that idea: use only the *first and last* iterate, θ₀ = φ and θₖ, and a *fixed* combination weight α. One extra copy of the parameters, no sub-problem. If that already reduces variance, the complexity isn't worth it.

Let me check the cost. One extra buffer the size of the parameters (the slow weights). The interpolation and the copy are O(parameters) but happen once every k inner steps, so amortized it's O((k+1)/k) times the inner optimizer's work — essentially free for k=5 or 10. Memory: one extra copy. Fine.

Now I should make sure this thing actually is an EMA of something meaningful, because that was the design goal. Unroll the slow-weight update across outer iterations. After each window I have φₜ₊₁ = φₜ + α(θ_{t,k} − φₜ) = (1−α)φₜ + α θ_{t,k}. That's a linear recursion in φₜ driven by the sequence of *final* fast weights θ_{t,k}. Unrolling: φₜ₊₁ = α θ_{t,k} + α(1−α)θ_{t−1,k} + … + α(1−α)^{t−1} θ_{0,k} + (1−α)ᵗ φ₀. So the slow weights are exactly an exponential moving average of the final fast weight of each inner loop, decay (1−α), heavily weighting the most recent window but keeping a fading tail of older ones. Exactly the recent-weighted, start-free average I wanted, and it falls straight out of the interpolation rule. And note α = 1 collapses it to "just take the fast weights," i.e. recovers the inner optimizer — a clean sanity limit.

Does it reduce the variance though? I argued it should by analogy to averaging, but let me actually do it on the noisy quadratic, because that's where I can compute the steady state and see whether the first factor really comes out below one. I already have SGD's dynamics. I need the dynamics of φ under the slow update, with SGD as the inner optimizer. Work one coordinate at a time since everything's diagonal; write the per-step contraction as (1−γa) and stack k of them.

Expectation first. E[φₜ₊₁] = (1−α)E[φₜ] + α E[θ_{t,k}]. The inner loop is k SGD steps starting from φₜ, and SGD contracts the mean by (1−γa) per step, so E[θ_{t,k}] = (1−γa)ᵏ E[φₜ] (using E starts at φₜ since θ_{t,0}=φₜ). Therefore E[φₜ₊₁] = [(1−α) + α(1−γa)ᵏ] E[φₜ], or in matrix form [1−α + α(I−γA)ᵏ]E[φₜ]. So the bias contracts with factor 1−α + α(I−γA)ᵏ. Hold that thought — compare it to SGD's per-window bias factor (I−γA)ᵏ. Since (I−γA)ᵏ ≺ 1 and we're taking a convex combination of 1 and (I−γA)ᵏ with weight α on the small one, 1−α+α(I−γA)ᵏ is *larger* than (I−γA)ᵏ for α < 1. So Lookahead contracts the *bias* more slowly than raw SGD over the same number of steps. Noted — there's a cost on the bias side. I'll come back to whether it matters.

Now the variance, which is the whole point. V[φₜ₊₁] = (1−α)²V[φₜ] + α²V[θ_{t,k}] + 2α(1−α)cov(φₜ, θ_{t,k}). I need V[θ_{t,k}] and that covariance. For the variance of θ after k SGD steps from a fixed start, iterate V[θ_{t,i}] = (1−γa)²V[θ_{t,i−1}] + γ²a²σ² starting from V[θ_{t,0}] = V[φₜ]; unrolling gives V[θ_{t,k}] = (1−γa)^{2k}V[φₜ] + Σ_{i=0}^{k−1}(1−γa)^{2i}γ²a²σ². The covariance I have to be careful with. Take cov(φₜ, θ_{t,k}). Since θ_{t,k} = (1−γa)θ_{t,k−1} + (noise independent of the past), and stepping down, the cleanest is the one-step relation: cov(θ_{t,k−1}, θ_{t,k}) = E[(θ_{t,k−1}−E)(θ_{t,k}−E)]. Substitute θ_{t,k} − E[θ_{t,k}] = (1−γa)(θ_{t,k−1} − E[θ_{t,k−1}]) + γa(c − E[c]); the noise term is independent of θ_{t,k−1}, so cov(θ_{t,k−1}, θ_{t,k}) = (1−γa)V[θ_{t,k−1}]. Chaining this back from θ_{t,k} all the way to θ_{t,0}=φₜ multiplies one factor of (1−γa) per step, so cov(φₜ, θ_{t,k}) = (1−γa)ᵏ V[φₜ], i.e. (I−γA)ᵏ V[φₜ].

Substitute everything and collect the V[φₜ] terms:
V[φₜ₊₁] = [(1−α)² + 2α(1−α)(1−γa)ᵏ + α²(1−γa)^{2k}] V[φₜ] + α² Σ_{i=0}^{k−1}(1−γa)^{2i}γ²a²σ².
And the bracket is a perfect square: (1−α) + α(1−γa)ᵏ, squared. Nice — consistent with the mean's contraction factor. So
V[φₜ₊₁] = [1−α+α(1−γa)ᵏ]² V[φₜ] + α² Σ_{i=0}^{k−1}(1−γa)^{2i} γ²a²σ².
Set the fixed point V*_LA = [1−α+α(1−γa)ᵏ]² V*_LA + α² Σ_{i=0}^{k−1}(1−γa)^{2i}γ²a²σ² and solve:
V*_LA = α² Σ_{i=0}^{k−1}(1−γa)^{2i} / (1 − [(1−α)+α(1−γa)ᵏ]²) · γ²a²σ².

Now I want to compare this to V*_SGD = γ²a²σ² / (1−(1−γa)²). Use the geometric sum Σ_{i=0}^{k−1}rⁱ = (1−rᵏ)/(1−r) with r = (1−γa)²: the numerator's sum becomes (1 − (1−γa)^{2k})/(1−(1−γa)²). So
V*_LA = α²(1 − (1−γa)^{2k}) / [(1−(1−γa)²)(1 − [(1−α)+α(1−γa)ᵏ]²)] · γ²a²σ².
Pull out the γ²a²σ²/(1−(1−γa)²) = V*_SGD:
V*_LA = α²(1 − (1−γa)^{2k}) / (1 − [(1−α)+α(1−γa)ᵏ]²) · V*_SGD.
Let me simplify the denominator. Write β = (1−γa)ᵏ for brevity. Denominator = 1 − [(1−α)+αβ]² = 1 − [1 − α(1−β)]² = 2α(1−β) − α²(1−β)² = α(1−β)[2 − α(1−β)] = α(1−β)[2(1−α) + α(1+β) ]… let me just expand cleanly: 2α(1−β) − α²(1−β)² . And the numerator factor is α²(1−β²) = α²(1−β)(1+β). So
V*_LA / V*_SGD = α²(1−β)(1+β) / [2α(1−β) − α²(1−β)²] = α²(1+β) / [2α − α²(1−β)] = α(1+β) / [2 − α(1−β)].
Let me rewrite the denominator as α(1+β) + 2(1−α) + … hmm, check: 2 − α(1−β) = 2 − α + αβ = (1−α) + (1 + αβ)… let me instead match the published-looking form by multiplying num and denom by (1−β): V*_LA/V*_SGD = α²(1−β²) / [α²(1−β²) + 2α(1−α)(1−β)]. (Multiply my α(1+β)/[2−α(1−β)] top and bottom by α(1−β): top α²(1−β²); bottom α(1−β)[2−α(1−β)] = 2α(1−β) − α²(1−β)², matches the denominator above. Good, both forms agree.) So, restoring β = (1−γa)ᵏ and per-coordinate to matrix form:
V*_LA = α²(I − (I−γA)^{2k}) / [α²(I − (I−γA)^{2k}) + 2α(1−α)(I − (I−γA)ᵏ)] · V*_SGD.

Now stare at that ratio. Both terms in the denominator are nonnegative for α ∈ (0,1) and 0 < γ < 2/aᵢ (so 0 < β < 1, both (1−βᵏ) factors positive). The ratio is (something) / (that same something + a strictly positive 2α(1−α)(1−β) term). So the ratio is strictly less than 1. The slow weights have a strictly smaller steady-state variance than the inner SGD at the *same* learning rate. There it is — exactly the prize I set out for: lower the variance floor without touching γ.

But I already saw the cost: the bias contracts more slowly, factor 1−α+α(1−γa)ᵏ versus (1−γa)ᵏ. So is this actually a net win, or did I just trade variance for bias? Two ways to think about it. First, fairness: compare the two methods at hyperparameters that reach the *same* steady-state risk, and ask who gets there faster — and on the noisy quadratic, sweeping α over (0,1] at fixed k, Lookahead reaches the matched steady state in fewer updates across a broad range of α. Second, and more to the point for real networks: which term actually dominates? When I train networks I deliberately use a *large* learning rate early — that's the regime where the bias is already contracting fast and the *variance* is the thing holding the loss up. There's the short-horizon-bias result (Wu et al.) that SGD is pulled toward exactly these large step sizes. In that high-γ, variance-limited regime, knocking down the variance term is the binding constraint, and the slight bias slowdown is cheap. That squares with what I'd expect to see inside a real run: if I watch the loss after every inner step, the fast weights overshoot and degrade accuracy within a window — high variance, just as the model predicts — and the interpolation step snaps it back. The slow weights are doing variance reduction in real time.

Let me push on the geometry too, because "variance reduction" is the stochastic story but there's a deterministic one. Drop the noise; take pure gradient descent with momentum on a quadratic, in the under-damped regime where momentum is high enough that the iterate spirals/oscillates toward the optimum instead of crawling straight in. Lookahead wraps k momentum steps and then interpolates between the start and the end of the window. If the trajectory is oscillating, the start and end of a window sit on different phases of the oscillation, and the straight-line interpolation between them cuts the corner — it lands closer to the optimum than either endpoint, skipping across the oscillation. So in the under-damped regime Lookahead should *speed up* convergence; in the over-damped regime, where momentum is too low and the iterate is already crawling monotonically inward, there's nothing to cut across and interpolating back can only slow you slightly. I can make this exact: stack the fast-weight iterates into a vector and write one slow-weight cycle as a linear map — an interpolation matrix L applied after k−1 plain momentum-update matrices B and one realignment matrix T, so the whole cycle is L·B^{(k−1)}·T. The spectral radius of that product bounds the contraction; since one application corresponds to k inner steps, I take the k-th root of its eigenvalues to get the per-step rate. Computing it confirms the picture: better rate than bare momentum exactly in the under-damped, oscillating regime.

Now, the one knob I introduced that I haven't justified is α. I picked "small-ish, fixed." Can I do better — choose α optimally? On a quadratic L(x) = ½xᵀAx − bᵀx with optimum x* = A⁻¹b, given the window endpoints θ_{t,0} and θ_{t,k}, the interpolation θ_{t,0} + α(θ_{t,k} − θ_{t,0}) is a 1-D line search, so I can minimize exactly. Differentiate: d/dα L = (θ_{t,k} − θ_{t,0})ᵀ A (θ_{t,0} + α(θ_{t,k} − θ_{t,0})) − (θ_{t,k} − θ_{t,0})ᵀ b. Set to zero, use b = A x*:
α[(θ_{t,k} − θ_{t,0})ᵀ A (θ_{t,k} − θ_{t,0})] = (θ_{t,k} − θ_{t,0})ᵀ A (x* − θ_{t,0}),
so α* = (θ_{t,0} − x*)ᵀ A (θ_{t,0} − θ_{t,k}) / [(θ_{t,0} − θ_{t,k})ᵀ A (θ_{t,0} − θ_{t,k})]. Clean closed form. To use it I need A (a curvature) and x* (the optimum), neither of which I have. But I can approximate: use the diagonal empirical-Fisher curvature that an adaptive optimizer like Adam already maintains as Â, and approximate the optimum by one Newton-ish step from the window's end, x* ≈ θ_{t,k} − Â⁻¹∇̂L(θ_{t,k}). Then α̂* = clip( (θ_{t,0} − (θ_{t,k} − Â⁻¹∇̂L(θ_{t,k})))ᵀ Â (θ_{t,0} − θ_{t,k}) / [(θ_{t,0} − θ_{t,k})ᵀ Â (θ_{t,0} − θ_{t,k})], α_low, 1). The clip with α_low > 0 keeps it stable. This does converge well in experiments. But — it forces me to maintain a curvature estimate even when the inner optimizer is SGD (which doesn't keep one), costing memory and compute, and the fixed-α version converges about as fast and tends to generalize a bit better. So I keep α fixed. The adaptive derivation is reassurance that a fixed small α is a sensible operating point, not a thing to ship.

One more design choice: the inner optimizer carries internal state — momentum buffers, Adam's moments. When I interpolate the parameters at the end of a window, what do I do with that state? Three options: maintain it (just leave the momentum running across the reset), interpolate it the same way I interpolate the parameters, or reset it to zero. All three beat the bare inner optimizer; maintaining and interpolating are essentially tied and both edge out resetting. So I'll default to the simplest — maintain — and expose the other two.

So the algorithm, end to end: keep slow weights φ and fast weights θ. Each outer iteration, set θ ← φ; run the inner optimizer A for k steps on minibatches, updating θ; then φ ← φ + α(θ − φ); reset θ to φ and repeat. Defaults k = 5, α = 0.8. It wraps any optimizer, costs one parameter copy and O((k+1)/k) work, runs from initialization with no start decision, is an EMA of the final fast weights (recent-weighted), provably lowers the steady-state variance on the noisy quadratic at fixed learning rate, and cuts across oscillations in the under-damped deterministic case.

Let me write it as a thin wrapper around `torch.optim.Optimizer`. The fast weights are just the live `p.data` that the inner optimizer mutates; the slow weights I'll stash per-parameter as `cached_params`. A counter ticks each inner `step`; when it hits k I do the interpolation in place and re-cache.

```python
from collections import defaultdict
import torch
from torch.optim.optimizer import Optimizer


class Lookahead(Optimizer):
    def __init__(self, optimizer, la_steps=5, la_alpha=0.8, pullback_momentum="none"):
        # optimizer: any inner optimizer A (SGD, Adam, ...). We only ever call its .step().
        self.optimizer = optimizer
        self._la_step = 0                 # counts inner steps within the current window
        self.la_alpha = la_alpha          # interpolation weight alpha; alpha=1 -> just the inner optimizer
        self._total_la_steps = la_steps   # window length k
        pullback_momentum = pullback_momentum.lower()
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum
        self.state = defaultdict(dict)
        # slow weights phi: a cached copy of the parameters, taken at sync time
        for group in optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)
                if self.pullback_momentum == "pullback":
                    param_state['cached_mom'] = torch.zeros_like(p.data)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        # swap in the slow weights phi (they generalize better) for evaluation
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    def step(self, closure=None):
        # k steps forward: let the inner optimizer take one fast-weight step
        loss = self.optimizer.step(closure)
        self._la_step += 1

        if self._la_step >= self._total_la_steps:
            self._la_step = 0
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    # 1 step back: phi <- phi + alpha (theta_k - phi), in place on p.data.
                    # p = alpha*theta_k + (1-alpha)*phi  is exactly that interpolation.
                    p.data.mul_(self.la_alpha).add_(param_state['cached_params'], alpha=1.0 - self.la_alpha)
                    # commit: the new phi becomes the sync point and the next window's start
                    param_state['cached_params'].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        # interpolate the inner momentum buffer the same way as the params
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.la_alpha).add_(
                            1.0 - self.la_alpha, param_state["cached_mom"])
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)
        return loss
```

Recapping the chain: the learning-rate fragility traces to a bias/variance trade-off in the stochastic iterates, where a large step size that contracts the bias fast also inflates the steady-state variance; averaging iterates cancels that variance, but the classical Polyak/SWA averages are flat, need a start-time decision, and only read out a final model instead of steering the optimizer; so I keep a slow weight that is an exponential moving average of the *final* fast weight of each k-step window and reset the fast weights to it, which feeds the average back into the trajectory, wraps any inner optimizer for one parameter-copy of overhead, and — verified on the noisy quadratic — strictly lowers the steady-state variance at fixed learning rate while cutting across oscillations in the under-damped regime; the interpolation weight α can be set by an exact quadratic line search, but a fixed α ≈ 0.8 with k ≈ 5 is simpler, generalizes better, and is what the wrapper above ships.
