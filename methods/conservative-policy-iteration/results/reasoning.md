Let me start from what actually goes wrong, because two methods I trust both keep failing on large problems and I want to know *why* before I reach for a fix.

I have a simulator. I can't sweep the states — there are too many — so everything I do is approximate: I fit value functions, I restrict to a policy class, I estimate from rollouts. The two tools everyone uses are approximate value-function methods (do policy evaluation, then act greedily) and policy gradient (parameterize the policy, ascend the return). Both win in practice. Both have a hole I can poke a finger through.

Take approximate greedy first. Exact policy iteration is beautiful: compute Q_π, switch to the deterministic policy that picks argmax_a Q_π(s,a) at every state, and the new policy is provably at least as good, everywhere. That monotone improvement is the whole reason policy iteration converges. Now make it approximate. I only have an estimate of the values, with some L∞ error ε. What survives? Only a max-norm statement — for a greedy policy π' built off an approximation that is ε-accurate in sup norm, V_{π'}(s) ≥ V_π(s) − 2γε/(1−γ). Stare at that. It's a *lower* bound that allows the value to go *down* by 2γε/(1−γ). The same shape shows up in Williams and Baird's 1993 bound, V_π ≥ V* − 2B_J/(1−γ) with B_J the Bellman error. So approximate greedy gives me no guarantee of improvement at all; it gives me a license to degrade. Why is the penalty so brutal? Because a greedy swap throws out the old policy *at every state at once*. If my value estimate is wrong somewhere, the new policy commits to that error everywhere, and the mistake compounds over the whole horizon — that's the 1/(1−γ).

Now policy gradient. With the normalized value convention, the gradient is ∂η/∂θ = (1/(1−γ))Σ_s d_{π,D}(s) Σ_a (∂π(s,a)/∂θ) Q_π(s,a). The thing I love about it, due to Sutton, McAllester, Singh and Mansour, is that there's *no* ∂d_π/∂θ term — nudging θ a hair moves the state-visitation distribution only a hair. So unlike the greedy swap, a small gradient step can't violently relocate where I spend my time, and η is guaranteed to improve to first order. That should fix everything. It doesn't, and the reason is sample complexity. Picture the length-n chain where random actions tend to push me *away* from the goal: undirected exploration takes time exponential in n — for n = 50 that's around 10^15 steps just to reach the goal once. Any on-policy estimate of the gradient has to actually *visit* the informative states, and it can't. Worse, look at the two-state Gibbs example: crank the self-loop probability at one state and the stationary probability of the other state crashes from 0.2 down to about 10^-7. Learning at that state just stops.

Why does it stop? Because the gradient is weighted by d_{π,D} — the states the *current* policy visits. The contribution of a state to ∂η/∂θ is multiplied by how often I'm there now. A state I currently avoid contributes essentially nothing, so improvement there is invisible.

And that's the same disease the greedy method has, seen from the other side. η_D measures performance from the start distribution D, and it down-weights every state by the current policy's own visitation. But the states whose improvement I *need* in order to become optimal might be exactly the ones the current policy almost never sees. A policy that's bad precisely because it never goes somewhere useful will report a tiny gradient there and feel locally fine. The metric is blind where I most need to see.

So the real enemy isn't "greedy is too aggressive" or "gradients are noisy." It's that I'm scoring improvement under a distribution — the current policy's visitation — that systematically starves the states that matter. If I could decouple *where I measure improvement* from *where the current policy happens to go*, both pathologies might lift.

I have one capability I haven't used: a restart distribution. I can reset my next state to a draw from a fixed μ of my choosing. Not as strong as a full generative model — I can't interrogate an arbitrary (s,a) — but far stronger than being stuck on one irreversible trajectory. If I pick μ fairly uniform, the restart *is* my exploration: I can gather information about states I'd otherwise never reach.

So let me change the thing I optimize. Instead of η_D, optimize η_μ(π) = E_{s~μ}[V_π(s)] for an exploratory μ — improvement weighted more uniformly across states, not crushed by the current policy's habits. The optimal policy maximizes V_π(s) at *every* state, so it maximizes η_μ for any μ; I'm not changing the destination, only the meter I read on the way. Good. But this immediately raises a worry I'll have to pay off later: a policy that maxes out η_μ within some restricted class could still be poor under η_D, the thing I actually care about. Hold that thought.

Now, even with the better meter, how do I update without falling back into the greedy degradation? I need to understand *exactly* how performance changes between two policies, not in max-norm but as an honest accounting. Let me just compute the difference η_μ(π̃) − η_μ(π) for two arbitrary policies and see what it's made of.

Fix a start state s and expand V_{π̃}(s), the value under the new policy π̃, but I'll measure everything in terms of the *old* policy's value V_π. Write P_t(s') = Pr(s_t = s'; π̃, s_0 = s), the distribution over states at step t when I follow π̃. Then, with normalized values,

  V_{π̃}(s) = (1−γ) Σ_t γ^t E_{(s_t,a_t)~π̃P_t}[ R(s_t,a_t) ].

I want advantages of π to appear, so I add and subtract V_π at each visited state. Inside the sum, (1−γ)R(s_t,a_t) is the immediate piece; pair it with γV_π(s_{t+1}) − V_π(s_t), telescoping the discounted value along the trajectory. Take the expectation of s_{t+1} given (s_t,a_t): (1−γ)R(s_t,a_t) + γ E[V_π(s_{t+1})] = Q_π(s_t,a_t). So that grouped term is Q_π(s_t,a_t) − V_π(s_t) = A_π(s_t,a_t). The telescoping leaves exactly one boundary term, V_π(s) at t=0, uncancelled. Collecting,

  V_{π̃}(s) = V_π(s) + Σ_t γ^t E_{(s_t,a_t)~π̃P_t}[ A_π(s_t,a_t) ].

The leftover sum is the future-state distribution of π̃ folded in: Σ_t γ^t Pr(s_t = s'; π̃, s) = d_{π̃,s}(s')/(1−γ). So

  V_{π̃}(s) − V_π(s) = (1/(1−γ)) E_{s'~d_{π̃,s}} E_{a~π̃(·|s')}[ A_π(s', a) ],

and taking E_{s~μ},

  η_μ(π̃) − η_μ(π) = (1/(1−γ)) E_{s~d_{π̃,μ}} E_{a~π̃(·|s)}[ A_π(s, a) ].

There it is, and look at *whose* distribution shows up. The advantages are of the old policy π. They are averaged over the future-state distribution of the *new* policy π̃ — d_{π̃,μ}, not d_{π,μ}. A quick sanity check: if π̃ = π the inner expectation E_{a~π}[A_π] = 0 at every state (advantages are centered under π), so the difference is zero. Good.

This single identity *is* the off-distribution problem, stated exactly. When I switch to π̃, the improvement is the old advantages reweighted by the *new* policy's visitation. I can estimate A_π and I can estimate d_{π,μ}, but the formula wants d_{π̃,μ} — a distribution I don't get to see until *after* I've committed to π̃. If π̃ is far from π, its visitation is far from d_{π,μ}, and any improvement I "measured" under d_{π,μ} can evaporate, or reverse. That's precisely why the greedy swap degrades: it picks π̃ to look great under the old distribution, then runs under a new one where the picked advantages no longer hold.

So the trap is the mismatch d_{π̃,μ} vs d_{π,μ}. The fix has to be: don't let the new policy's visitation wander far from the old one's. Suppose I only nudge the policy a little. Then d_{π̃,μ} ≈ d_{π,μ}, and the identity becomes trustworthy — I can use the old distribution as a stand-in. How little is "a little," and how do I get a *first-order* improvement I can actually control?

Let me define the quantity the identity says I should care about when the change is small. Call it the **policy advantage** of a candidate π' relative to π and μ:

  𝔸_{π,μ}(π') = E_{s~d_{π,μ}} E_{a~π'(·|s)}[ A_π(s,a) ].

It's the average advantage of π' measured under the *old* policy's future-state distribution — which is the distribution I can sample. If I parameterize a move from π toward π' by a scalar α and differentiate η_μ at α = 0, the ∂d/∂α term drops out at first order (same reason the policy-gradient theorem has no ∂d term — at α=0 the visitation is still d_{π,μ}), and what's left is

  ∂η_μ/∂α |_{α=0} = (1/(1−γ)) 𝔸_{π,μ}(π'),

so to first order η_μ improves iff the policy advantage is positive. That tells me what a good candidate π' is — one with large policy advantage under d_{π,μ} — and it's a quantity I can fit from samples by regression on the advantages, never needing the new policy's distribution.

But "first order" with an unstated higher-order term is exactly what bit greedy at α = 1. I need the move itself, and I need the *exact* dependence on α so I can pick α large enough to make real progress and small enough to stay safe.

What's the move? I have an old policy π and a candidate π'. The cleanest way to interpolate between them — and the one that makes "only nudge a little" literally true at the level of action selection — is a mixture: with probability 1−α act according to π, with probability α act according to π'. So

  π_new(a|s) = (1−α) π(a|s) + α π'(a|s),   α ∈ [0,1].

At α = 0 it's the old policy; at α = 1 it's the full greedy swap and I expect the degradation to come roaring back. In between, on any given step I deviate from π with probability exactly α, so the *typical* trajectory looks like π's trajectory with an occasional π' action sprinkled in — and that is the mechanism that keeps d_{π_new,μ} close to d_{π,μ}. The question is how the off-distribution error grows as those sprinkled deviations accumulate over the horizon.

Let me bound η_μ(π_new) − η_μ(π) exactly, for all α. Two pieces I'll need. First, the per-state expected advantage of the mixture under π's own advantages:

  E_{a~π_new}[A_π(s,a)] = Σ_a ((1−α)π(a|s) + απ'(a|s)) A_π(s,a)
                        = α Σ_a π'(a|s) A_π(s,a)
                        = α E_{a~π'}[A_π(s,a)],

because Σ_a π(a|s) A_π(s,a) = 0. So at the level of the mixture, the old part of the policy contributes nothing to the advantage — only the α-fraction of π' actions does. Write A_π(s, π') := E_{a~π'}[A_π(s,a)] for brevity; then E_{a~π_new}[A_π(s,·)] = α A_π(s, π').

Second, I need to track how far the *state distribution* under π_new has drifted from π's by time t. Here's the coupling. Following π_new, on each step I independently flip a coin: with probability α I take a π' action, otherwise a π action. Let c_t be the number of times I've taken a π' action *before* time t. If c_t = 0 — I haven't deviated yet — then the path up to time t was generated purely by π, so

  Pr(s_t = s | π_new, c_t = 0) = Pr(s_t = s | π).

That's the crux: conditioned on never having deviated, I'm exactly on π's distribution. The probability of having stayed pure through t steps is Pr(c_t = 0) = (1−α)^t. Let ρ_t = Pr(c_t ≥ 1) = 1 − (1−α)^t be the probability that at least one deviation has happened by time t. The ρ_t fraction is the part of the distribution that has drifted off π.

Now bound the expected advantage at step t under π_new. Let ε_inf = max_s |E_{a~π'(·|s)}[A_π(s,a)]| = max_s |A_π(s,π')| — the largest the per-state policy-advantage of π' can be at any state. Then

  E_{s~Pr(s_t|π_new)}[ A_π(s,π_new) ]
    = α E_{s~Pr(s_t|π_new)}[ A_π(s,π') ]
    = α(1−ρ_t) E_{s~Pr(s_t|π_new, c_t=0)}[A_π(s,π')] + α ρ_t E_{s~Pr(s_t|π_new, c_t≥1)}[A_π(s,π')].

On the c_t = 0 branch I substitute the π distribution exactly; on the c_t ≥ 1 branch I have no control over where I've drifted, so I lower-bound that expectation by its worst case, −ε_inf. And on the c_t = 0 branch I also drop (1−ρ_t) ≤ 1, costing me at most another ρ_t·ε_inf of slack since |A_π(s,π')| ≤ ε_inf. So

  E_{s~Pr(s_t|π_new)}[ A_π(s,π_new) ]
    ≥ α E_{s~Pr(s_t|π)}[A_π(s,π')] − α ρ_t ε_inf − α ρ_t ε_inf
    = α E_{s~Pr(s_t|π)}[A_π(s,π')] − 2 α ρ_t ε_inf.

The good term is the advantage of π' measured under *π's* distribution — exactly what 𝔸_{π,μ}(π') is built from. The penalty 2αρ_t ε_inf is the price of the drifted fraction: it grows with ρ_t = 1−(1−α)^t, the chance I've already deviated.

Feed this into the performance-difference identity, using the equivalent time-sum form η_μ(π_new) − η_μ(π) = Σ_t γ^t E_{Pr(s_t|π_new)}[A_π(s,π_new)]:

  η_μ(π_new) − η_μ(π)
    ≥ Σ_t γ^t ( α E_{s~Pr(s_t|π)}[A_π(s,π')] − 2 α ρ_t ε_inf )
    = α Σ_t γ^t E_{s~Pr(s_t|π)}[A_π(s,π')] − 2 α ε_inf Σ_t γ^t (1 − (1−α)^t).

The first sum is just the discounted future-state distribution of π folded back up: Σ_t γ^t E_{Pr(s_t|π)}[A_π(s,π')] = (1/(1−γ)) E_{s~d_{π,μ}}[A_π(s,π')] = (1/(1−γ)) 𝔸_{π,μ}(π'). The second: Σ_t γ^t = 1/(1−γ), and Σ_t γ^t (1−α)^t = 1/(1−γ(1−α)). So

  η_μ(π_new) − η_μ(π) ≥ (α/(1−γ)) 𝔸_{π,μ}(π') − 2 α ε_inf ( 1/(1−γ) − 1/(1−γ(1−α)) ).

That difference of reciprocals — let me get it exactly. Common denominator (1−γ)(1−γ(1−α)); numerator (1−γ(1−α)) − (1−γ) = γ − γ(1−α) = γα. So 1/(1−γ) − 1/(1−γ(1−α)) = γα / ((1−γ)(1−γ(1−α))). Substituting,

  η_μ(π_new) − η_μ(π) ≥ (α/(1−γ)) ( 𝔸_{π,μ}(π') − 2αγε_inf / (1−γ(1−α)) ).

This is the lower bound I wanted, and now I can read off everything. The first term, (α/(1−γ)) 𝔸, is exactly the first-order gain ∂η_μ/∂α|_0 · α — the honest improvement. The second is the penalty for going off-distribution, and it is O(α²): it carries an explicit α and the bracket γα/(1−γ(1−α)) carries another. So for small α the gain dominates and improvement is guaranteed whenever 𝔸 > 0. The off-distribution risk that destroys the greedy swap has been demoted to second order, *purely by the mixture structure*.

Let me confirm the α = 1 endpoint, where I expect to recover the disaster. Set α = 1: 1 − γ(1−α) = 1, so the bound becomes 𝔸/(1−γ) − 2γε_inf/(1−γ). The penalty 2γε_inf/(1−γ) has the *exact same form* as the approximate-greedy max-norm penalty I started from, with ε_inf now playing the role of the value-approximation error. So full greedy is literally the α = 1 corner of this bound, and at that corner I re-derive the known license-to-degrade. The mixture is the dial between the safe first-order regime and that cliff.

Now choose α to extract the most guaranteed improvement. I'll work with normalized values where the advantage magnitude is at most R, so ε_inf ≤ R, and bound the awkward denominator: 1 − γ(1−α) ≥ 1 − γ, hence 2αγε_inf/(1−γ(1−α)) ≤ 2αR/(1−γ). That gives the cleaner quadratic-in-α lower bound

  η_μ(π_new) − η_μ(π) ≥ (α/(1−γ)) ( 𝔸 − 2αR/(1−γ) )
                       = (α 𝔸)/(1−γ) − (2 α² R)/(1−γ)².

Maximize over α: derivative 𝔸/(1−γ) − 4αR/(1−γ)² = 0 gives

  α* = (1−γ) 𝔸 / (4R).

Plug it back. The bracket 𝔸 − 2α*R/(1−γ) = 𝔸 − 2·((1−γ)𝔸/(4R))·R/(1−γ) = 𝔸 − 𝔸/2 = 𝔸/2. And α*/(1−γ) = 𝔸/(4R). So

  η_μ(π_new) − η_μ(π) ≥ (𝔸/(4R))·(𝔸/2) = 𝔸² / (8R).

A clean, closed-form, *guaranteed* improvement of 𝔸²/(8R) per step, with the step size α* = (1−γ)𝔸/(4R) read straight off the bound. (Measuring values in [0,1] with horizon H = 1/(1−γ), the same algebra gives α* = (1−γ)𝔸/4 and improvement 𝔸²/8 — the R = 1 instance.) The max-norm value-error penalty is gone; the remaining worst-case term only controls how bad the candidate's per-state advantage could be on the drifted branch, and choosing α small enough beats it.

This answers my first two questions outright. There *is* a measure — η_μ — that I can improve every step (question 1), and verifying an update improves it reduces to checking a single scalar, the sign and size of the policy advantage 𝔸, which I estimate by regression on advantages under d_{π,μ} — no need to ever realize the new policy's distribution (question 2). So the loop writes itself: get a candidate π' that maximizes the policy advantage (a "policy chooser," which I can implement with whatever value/advantage regression I like), check 𝔸; if it's appreciably positive, take the mixture step with α = (1−γ)𝔸/(4R); repeat.

But I need to know it *terminates* and what it terminates *at* — question 3. Let the halt tolerance be ε. Each accepted update raises η_μ by at least 𝔸²/(8R), and if the candidate is accepted only when 𝔸 > ε, each update buys at least ε²/(8R). The value η_μ is bounded — it lives in [0,R] — so I can't keep adding fixed positive increments forever: the exact loop stops after at most 8R²/ε² accepted updates, and the sample-based bookkeeping with estimation margins inflates only the constant, to 72R²/ε² loops. That's polynomial in 1/ε and in R, and — the whole point — it has *nothing* to do with |S|. And improvement is monotone by construction, because every accepted step has 𝔸 > 0 hence Δη_μ > 0. The algorithm halts when the best candidate's policy advantage drops below the threshold — when 𝔸_{π,μ}(π') ≤ ε, i.e. the greedy chooser can no longer find a policy that beats π on average by more than ε. That's the natural break point: just past it, the first-order term no longer dominates the α² penalty and improvement is no longer guaranteed. So I stop exactly where the guarantee stops.

Now I have to pay off the worry I parked. I've been improving η_μ, but I care about η_D. When the algorithm halts, all I know is that the policy advantage under μ is small: OPT(𝔸_{π,μ}) < ε. Does small advantage under μ imply near-optimality under D? Not for free, and the performance-difference identity tells me exactly the gap. Apply it with an *optimal* policy π* and an arbitrary measure μ̄:

  η_{μ̄}(π*) − η_{μ̄}(π) = (1/(1−γ)) E_{s~d_{π*,μ̄}} E_{a~π*}[A_π(s,a)] = (1/(1−γ)) Σ_s d_{π*,μ̄}(s) Σ_a π*(a|s) A_π(s,a).

I want to bound this by what I control, OPT(𝔸_{π,μ}) = Σ_s d_{π,μ}(s) max_a A_π(s,a) < ε. Insert d_{π,μ}/d_{π,μ} and pull out the worst ratio:

  ε > OPT(𝔸_{π,μ}) = Σ_s (d_{π,μ}(s)/d_{π*,μ̄}(s)) d_{π*,μ̄}(s) max_a A_π(s,a)
                  ≥ (min_s d_{π,μ}(s)/d_{π*,μ̄}(s)) Σ_s d_{π*,μ̄}(s) max_a A_π(s,a)
                  ≥ ||d_{π*,μ̄}/d_{π,μ}||_∞^{-1} Σ_{s,a} d_{π*,μ̄}(s) π*(a|s) A_π(s,a)
                  = ||d_{π*,μ̄}/d_{π,μ}||_∞^{-1} (1−γ) ( η_{μ̄}(π*) − η_{μ̄}(π) ),

using max_a A_π(s,a) ≥ Σ_a π*(a|s)A_π(s,a) for the second-to-last step and the performance-difference identity for the last. Rearranging,

  η_{μ̄}(π*) − η_{μ̄}(π) ≤ (ε/(1−γ)) ||d_{π*,μ̄}/d_{π,μ}||_∞.

And since d_{π,μ}(s) ≥ (1−γ)μ(s) — the start distribution contributes its mass undiscounted, the t = 0 term of the (1−γ)Σγ^t sum — I can replace the denominator by μ and get

  η_{μ̄}(π*) − η_{μ̄}(π) ≤ (ε/(1−γ)²) ||d_{π*,μ̄}/μ||_∞.

So the residual sub-optimality is ε times a *mismatch ratio* between where an optimal policy goes and the measure μ I trained under. This is the second worry, made precise and made actionable: if μ is concentrated and d_{π*,D} spreads elsewhere, the ratio blows up and small advantages under μ say nothing about D. But if I pick μ *uniform-ish* — close to the future-state distribution of a good policy — the ratio stays bounded, and small policy advantage under μ does translate into near-optimality under D. That is exactly *why* the restart distribution should be exploratory: not as a heuristic, but because it's the factor controlling whether my η_μ guarantee transfers to the η_D I care about. The two factors of 1/(1−γ) are honest, too — one is the performance-difference horizon, the other is the inherent non-uniformity of d_{π,μ} (since d_{π,μ}(s) can be as small as (1−γ)μ(s)).

The last loose end is the policy chooser itself: I claimed I could find a π' with large policy advantage by regression, and I should check the regression problem is *easier* than the one greedy needed. The policy advantage can be written 𝔸_{π,μ}(π') = E_{s~d_{π,μ}} Σ_a (π'(a|s) − π(a|s)) Q_π(s,a). To sample s ~ d_{π,μ} I just roll out from μ and stop at each step with probability (1−γ) — a geometric cutoff reproduces the discounted future-state distribution exactly. Then I sample an action uniformly among the n_a actions, importance-weight to undo the uniform sampling, and estimate Q_π(s,a) by continuing the trajectory; each sample x_i = n_a Q̂(s,a)(π'(a|s) − π(a|s)) is a nearly-unbiased estimate of the policy advantage. With samples bounded in [−n_a R, n_a R], Hoeffding gives Pr(|𝔸 − 𝔸̂| > Δ) ≤ 2 exp(−kΔ²/(2 n_a² R²)), so O(n_a² R²/Δ²) trajectories suffice — a count in 1/Δ and the reward scale, not |S|. And the regression that yields the chooser only needs to keep the *average* fitting error small, E_{s~d_{π,μ}} max_a |A_π(s,a) − f(s,a)| ≤ ε/2 — an L1 error over states. That is a far weaker demand than the L∞ accuracy approximate greedy required; an average-accurate advantage fit is enough to produce an ε-good policy chooser. The whole reason greedy was hostage to a sup-norm error is gone.

So, landing it. I improve a uniform-ish measure η_μ rather than η_D, because η_D starves the states improvement must reach. I never trust a policy under a distribution it didn't generate — the performance-difference identity shows the gain rides the *new* policy's visitation, which I can't see — so I move conservatively, by a mixture, keeping the new visitation within O(α) of the old. The exact lower bound (α/(1−γ))(𝔸 − 2αγε_inf/(1−γ(1−α))) demotes the off-distribution risk to O(α²), and optimizing α gives a guaranteed per-step gain of 𝔸²/(8R) at step α* = (1−γ)𝔸/(4R). Bounded values plus a fixed positive increment force termination in O(R²/ε²) updates, independent of |S|, with monotone improvement throughout; and the measure-mismatch bound tells me that choosing μ exploratory is what carries that guarantee back to the return I actually care about.

```python
import numpy as np

# Conservative Policy Iteration.
# Improve eta_mu under an exploratory restart distribution mu; update by a MIXTURE
# of the current policy and a candidate, with step size read off the improvement bound.

class RestartMDP:
    """Simulator with restart access: next state can be drawn from mu."""
    def __init__(self, mu, gamma, n_actions, R):
        self.mu, self.gamma, self.n_actions, self.R = mu, gamma, n_actions, R
    def restart(self): ...      # draw s ~ mu
    def step(self, s, a): ...   # -> (s_next, reward in [0, R])

def sample_future_state(mdp, policy):
    # s ~ d_{policy, mu}: roll out from mu, stop each step w.p. (1 - gamma).
    s = mdp.restart()
    while np.random.rand() >= (1 - mdp.gamma):
        s, _ = mdp.step(s, policy.sample(s))
    return s

def estimate_Q(mdp, policy, s, a):
    # Unbiased normalized Q_policy(s,a): reward observed at a geometric time.
    s, r = mdp.step(s, a)
    while np.random.rand() < mdp.gamma:
        s, r = mdp.step(s, policy.sample(s))
    return r

def policy_advantage(mdp, policy, candidate, k):
    # A_{policy, mu}(candidate) = E_{s~d}[ sum_a (cand(a|s)-pol(a|s)) Q(s,a) ],
    # estimated by uniform action sampling + importance weight n_a. (Q ~ V + A.)
    est = []
    for _ in range(k):
        s = sample_future_state(mdp, policy)
        a = np.random.randint(mdp.n_actions)      # uniform action
        Qsa = estimate_Q(mdp, policy, s, a)
        est.append(mdp.n_actions * Qsa * (candidate.prob(s, a) - policy.prob(s, a)))
    return np.mean(est)

def fit_candidate(mdp, policy):
    # the policy chooser: fit advantages (average L1 error set by the tolerance)
    # and return a policy pi' that on average picks high-advantage actions under d.
    ...                                           # regression -> candidate pi'

class MixturePolicy:
    # pi_new(a|s) = (1 - alpha) pi(a|s) + alpha pi'(a|s)
    def __init__(self, base, cand, alpha): self.base, self.cand, self.alpha = base, cand, alpha
    def prob(self, s, a):
        return (1 - self.alpha) * self.base.prob(s, a) + self.alpha * self.cand.prob(s, a)
    def sample(self, s):
        return self.cand.sample(s) if np.random.rand() < self.alpha else self.base.sample(s)

def conservative_policy_iteration(mdp, policy, eps, k):
    while True:
        candidate = fit_candidate(mdp, policy)
        A = policy_advantage(mdp, policy, candidate, k)   # scalar verification
        if A <= eps:                                       # break point: gain no longer guaranteed
            return policy
        alpha = (1 - mdp.gamma) * A / (4 * mdp.R)          # alpha* from maximizing the bound
        policy = MixturePolicy(policy, candidate, alpha)   # conservative mixture update
        # guaranteed gain this step: eta_mu rises by >= A^2 / (8 R)
```
