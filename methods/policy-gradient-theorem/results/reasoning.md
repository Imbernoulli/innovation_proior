Let me start from what actually hurts. I have an agent in a Markov decision process and I want it to get better at collecting long-run reward — average reward per step in a continuing task, or discounted return from a start state in an episodic one. For anything realistic the state space is too big to tabulate, so I have to use a function approximator and share parameters across states. And the thing everyone does — estimate a value function, then act greedily with respect to it — keeps biting people. Two specific bites. One: the greedy rule always commits to a single action, so it's built to find deterministic policies, but I know there are problems where the best stationary policy *has* to randomize; the moment two states look identical to my features, a deterministic rule can be arbitrarily bad and a coin flip is optimal. Two, and this is the one that really worries me: the greedy map is *discontinuous* in the value estimates. Nudge one estimated value by epsilon and the action it selects can flip. So the policy is a step function of the parameters I'm actually adjusting. That's exactly the property that wrecks convergence proofs — and it's not academic, people have exhibited simple MDPs with simple approximators where Q-learning and Sarsa and approximate DP just oscillate or run away, even when you solve the value-prediction problem optimally at every step before touching the policy.

So flip it. Don't represent the policy implicitly through values. Give the policy its *own* parameters θ and let it be a smooth stochastic map: π(s,a,θ) = Pr{a_t=a | s_t=s, θ}, differentiable in θ. A neural net that outputs action probabilities, say, or a soft-max over linear action preferences. Now a small change in θ is a small change in the action probabilities and therefore a small change in everything downstream. If I could just compute the gradient of performance with respect to θ, I'd do gradient ascent: θ ← θ + α ∂ρ/∂θ, and a smooth landscape like that I can hope to drive to a local optimum.

That "if I could just compute the gradient" is the whole problem, though. Let me actually try to write ∂ρ/∂θ and see why it's scary. Take the average-reward case first. Performance is ρ = Σ_s d(s) Σ_a π(s,a) R^a_s, where d(s) is the stationary distribution of states under the policy. Differentiate. There are two places θ lives: in π(s,a), fine, and in d(s) — because changing the policy changes how often I land in each state. So naively

  ∂ρ/∂θ = Σ_s [∂d(s)/∂θ] Σ_a π(s,a) R^a_s + Σ_s d(s) Σ_a [∂π(s,a)/∂θ] R^a_s.

The second term I love — it's an expectation I could sample. The first term is a catastrophe. ∂d(s)/∂θ is the derivative of the *stationary distribution* of the Markov chain, which depends on the environment's transition probabilities P^a_{ss'} that I don't know and can't differentiate. There's no way I'm estimating that from sampled trajectories. If the gradient genuinely contains ∂d/∂θ, this whole direct-policy idea is dead on arrival.

So the question sharpens to exactly one thing: when I do the differentiation properly — not term-by-term on the one-step reward, but on the actual long-run value — does the ∂d/∂θ term survive, or does it cancel? Everything rides on that.

Let me not differentiate ρ directly. Let me differentiate the value function and let the recursion do the work, because the recursion is where the environment's dynamics are *exposed* and might cancel against themselves. Define V(s) = Σ_a π(s,a) Q(s,a) and differentiate it. Product rule:

  ∂V(s)/∂θ = Σ_a [ (∂π(s,a)/∂θ) Q(s,a) + π(s,a) ∂Q(s,a)/∂θ ].

The first piece is the friendly score-weighted-by-value term. The second piece, π ∂Q/∂θ, is where the dynamics enter. So I need ∂Q/∂θ. Use the Bellman relation for the differential value in the average-reward setting: Q(s,a) = R^a_s − ρ + Σ_{s'} P^a_{ss'} V(s'). The reward R^a_s doesn't depend on θ. So

  ∂Q(s,a)/∂θ = −∂ρ/∂θ + Σ_{s'} P^a_{ss'} ∂V(s')/∂θ.

There it is — the ∂Q/∂θ folds into the very thing I'm chasing (∂ρ/∂θ) plus a transported copy of ∂V/∂θ one step into the future. Substitute back:

  ∂V(s)/∂θ = Σ_a (∂π/∂θ) Q(s,a) + Σ_a π(s,a) [ −∂ρ/∂θ + Σ_{s'} P^a_{ss'} ∂V(s')/∂θ ].

Now Σ_a π(s,a) = 1, so the −∂ρ/∂θ comes straight out of the action sum, untouched:

  ∂V(s)/∂θ = Σ_a (∂π/∂θ) Q(s,a) − ∂ρ/∂θ + Σ_a π(s,a) Σ_{s'} P^a_{ss'} ∂V(s')/∂θ.

Solve for the thing I want by moving it to the left, or rather just rearrange to isolate ∂ρ/∂θ:

  ∂ρ/∂θ = Σ_a (∂π(s,a)/∂θ) Q(s,a) + Σ_a π(s,a) Σ_{s'} P^a_{ss'} ∂V(s')/∂θ − ∂V(s)/∂θ.   (★)

This holds for *every* state s. I have a relation that ties ∂ρ/∂θ to ∂V evaluated at s and at all the one-step successors s'. The ∂V terms are still there, and they still secretly contain the dynamics. But notice the shape: on the right I have ∂V at the *successors* (transported through P) minus ∂V at *s itself*. That's a difference of the same object evaluated at the current state versus the next state. If I average this over the right distribution, those two could annihilate.

Which distribution? The stationary one, d. That's the distribution that's invariant under exactly the transport Σ_a π P appearing in (★). Sum both sides of (★) against d(s):

  Σ_s d(s) ∂ρ/∂θ = Σ_s d(s) Σ_a (∂π/∂θ) Q(s,a)
                   + Σ_s d(s) Σ_a π(s,a) Σ_{s'} P^a_{ss'} ∂V(s')/∂θ
                   − Σ_s d(s) ∂V(s)/∂θ.

Left side: ∂ρ/∂θ is constant in s and Σ_s d(s) = 1, so the left side is just ∂ρ/∂θ. Good. Now the two ∂V terms. The middle one: I can swap the order of summation and collect the coefficient of ∂V(s')/∂θ. That coefficient is Σ_s d(s) Σ_a π(s,a) P^a_{ss'}. And that is *precisely* the one-step push-forward of d under the policy's induced chain — it's the probability of being at s' after one step starting distributed as d. But d is stationary: Σ_s d(s) Σ_a π(s,a) P^a_{ss'} = d(s'). So the middle term collapses to Σ_{s'} d(s') ∂V(s')/∂θ. Rename the dummy s' to s and it's Σ_s d(s) ∂V(s)/∂θ — identical to the last term. They cancel exactly.

  ∂ρ/∂θ = Σ_s d(s) Σ_a (∂π(s,a)/∂θ) Q(s,a).

The ∂d/∂θ term is *gone*. Not approximated away — gone, identically, because the derivative of the value function transports stationarily and the stationary distribution eats the transport. The only place θ appears on the right is inside ∂π/∂θ, exactly where I can differentiate it. The state distribution d(s) is still there as a *weighting*, but I never differentiate it — and I get that weighting for free, because if I just *act* under the policy, the states I visit are drawn from d. So I sample states by living in the world, and at each one I form Σ_a (∂π/∂θ) Q. That's an estimator.

I want to make sure this isn't an artifact of the average-reward bookkeeping, so let me redo it for the start-state discounted formulation, where ρ = E{Σ_{t=1}^∞ γ^{t-1} r_t | s_0} = V(s_0). Same opening:

  ∂V(s)/∂θ = Σ_a [ (∂π/∂θ) Q(s,a) + π(s,a) ∂Q(s,a)/∂θ ],

but now the Bellman relation is Q(s,a) = R^a_s + Σ_{s'} γ P^a_{ss'} V(s'), so ∂Q/∂θ = Σ_{s'} γ P^a_{ss'} ∂V(s')/∂θ — no −∂ρ/∂θ this time. Hence

  ∂V(s)/∂θ = Σ_a (∂π(s,a)/∂θ) Q(s,a) + Σ_a π(s,a) Σ_{s'} γ P^a_{ss'} ∂V(s')/∂θ.   (7)

This is a recursion in ∂V/∂θ with a discount γ each time I step forward, so unrolling it is going to converge. Unroll: ∂V(s)/∂θ equals the local score-weighted-by-value term at s, plus γ times a policy-weighted average of ∂V/∂θ at the one-step successors, each of which expands again into its own local term plus γ times *its* successors, and so on. Collect by where I land and how many steps it took to get there. After k steps the probability of being at x is Pr(s→x, k, π), carrying a factor γ^k. So

  ∂V(s)/∂θ = Σ_x Σ_{k=0}^∞ γ^k Pr(s→x, k, π) Σ_a (∂π(x,a)/∂θ) Q(x,a).

Now set s = s_0 and use ∂ρ/∂θ = ∂V(s_0)/∂θ:

  ∂ρ/∂θ = Σ_s [ Σ_{k=0}^∞ γ^k Pr(s_0→s, k, π) ] Σ_a (∂π(s,a)/∂θ) Q(s,a).

The bracket is the discounted state weighting — the discounted count of how often the agent occupies each state when it starts at s_0 and follows π. It is not normalized like a probability distribution unless I divide by its total mass, but the theorem only needs the weighting itself. Call it d(s) = Σ_{k=0}^∞ γ^k Pr(s_0→s, k, π). Then

  ∂ρ/∂θ = Σ_s d(s) Σ_a (∂π(s,a)/∂θ) Q(s,a)

— the *same form* as the average-reward case. Different meaning of d (stationary vs. discounted occupancy), identical structure, and again no derivative of the state distribution anywhere. So this is a real theorem, not a coincidence of one formulation: for any MDP, in either the average-reward or the start-state formulation,

  ∂ρ/∂θ = Σ_s d^π(s) Σ_a [∂π(s,a)/∂θ] Q^π(s,a),

with no term of the form ∂d^π(s)/∂θ. That absence is the entire content. It's what makes the gradient sampleable.

Now turn it into something I can actually compute from a sampled trajectory, not a sum over all states and actions. In the continuing case, following π samples states from the stationary d^π after the chain has mixed. In the start-state discounted case, the same sum is realized by weighting the time-t contribution by γ^t, or equivalently by sampling a time with probability proportional to γ^t. Either way, the state weighting is obtained by acting; I do not differentiate it. I also don't enumerate actions; I *sample* one action a ∼ π(s,·). So I want the action sum written as an expectation under π. The summand is (∂π/∂θ) Q. Multiply and divide by π(s,a):

  Σ_a (∂π(s,a)/∂θ) Q^π(s,a) = Σ_a π(s,a) · [ (∂π(s,a)/∂θ) / π(s,a) ] · Q^π(s,a)
                            = E_{a∼π}[ (1/π(s,a)) (∂π(s,a)/∂θ) Q^π(s,a) ].

And (1/π) ∂π/∂θ is just ∂ log π(s,a)/∂θ. So

  ∂ρ/∂θ = Σ_s d^π(s) E_{a∼π(·|s)}[ ∇_θ log π(a|s) · Q^π(s,a) ],

with d^π interpreted as the stationary distribution in the average-reward case and as the unnormalized discounted occupancy in the start-state case. If I normalize the discounted occupancy to sample a state from it, I carry the missing normalizing constant outside; if I work along a trajectory, I carry the γ^t weight at time t.

That 1/π is doing real work: it down-weights actions that are sampled often precisely *because* they're probable, correcting the oversampling so the estimator stays unbiased. This is the score-function estimator. With it, the policy-update rule writes itself: for a transition sampled from the appropriate state weighting, move θ by ∇_θ log π(a|s) times whatever I use for Q^π(s,a). If I plug in the actual return as my estimate of Q^π — R_t = Σ_{k≥1} γ^{k-1} r_{t+k} in the discounted case, or Σ_{k≥1}(r_{t+k} − ρ) in the average-reward case — I get Δθ_t ∝ (∂π(s_t,a_t)/∂θ) R_t / π(s_t,a_t), with the extra γ^t factor when I realize the discounted occupancy by walking forward from s_0. That is exactly the kind of update Williams' REINFORCE makes, and it is known to follow the gradient in expectation. So REINFORCE isn't a separate trick; it's this theorem with the crudest possible estimate of Q substituted in.

That's also where I expect to pay. Using the raw return R_t for Q^π is unbiased but the return swings wildly from episode to episode, so the gradient estimate is high-variance and learning crawls. Everyone who's used REINFORCE has felt this. The fix the field already reaches for is a *baseline*: subtract something from the return before multiplying by the score. I want to understand exactly why that's allowed, because if subtracting changes the gradient I'm in trouble, and if it doesn't I get variance reduction for free.

The action sum exposes the same fact in two guises. Add an arbitrary function v(s) that depends on the state but not the action:

  Σ_a [∂π(s,a)/∂θ] [ Q^π(s,a) + v(s) ] = Σ_a [∂π/∂θ] Q^π(s,a) + v(s) Σ_a ∂π(s,a)/∂θ.

What is Σ_a ∂π(s,a)/∂θ? It's ∂/∂θ Σ_a π(s,a) = ∂/∂θ (1) = 0, because the action probabilities sum to one for every s, and the derivative of a constant is zero. So Σ_a ∂π(s,a)/∂θ = 0 for all s, and the v(s) term vanishes identically. The gradient is unchanged for *any* state-dependent baseline v. In the sampled, score-function form this is the statement E_{a∼π}[ ∇_θ log π(a|s) v(s) ] = v(s) E_{a∼π}[ ∇_θ log π(a|s) ] = v(s)·0 = 0, since the expected score of a distribution under itself is zero (again because the probabilities sum to one). It's the same zero. So I can replace Q^π(s,a) by Q^π(s,a) − b(s) for any b depending only on s, leave the gradient exactly where it was, and choose b to shrink the variance of the estimate — the obvious choice being b(s) ≈ V^π(s), so that what multiplies the score is the *advantage* Q^π(s,a) − V^π(s), positive for better-than-average actions and negative for worse. This is precisely the reinforcement-comparison idea — maintain a running estimate of expected reward and subtract it — and now I see it's not a heuristic, it's exact.

This is the same argument Williams makes for REINFORCE, and worth being precise about because it's the load-bearing antecedent. In his setting each stochastic unit emits an output y_i with probability mass g_i, and the update is Δw_ij = α_ij (r − b_ij) e_ij with eligibility e_ij = ∂ log g_i/∂w_ij and a baseline b_ij that's conditionally independent of the unit's own output. To show the baseline doesn't bias the update, sum the contribution of the baseline over the possible outputs ξ: it's proportional to Σ_ξ E{b_ij} ∂g_i(ξ)/∂w_ij. Because b_ij doesn't depend on ξ, pull the expectation out: E{b_ij} Σ_ξ ∂g_i(ξ)/∂w_ij. And Σ_ξ g_i(ξ) = 1 (it's a probability distribution over outputs), so Σ_ξ ∂g_i(ξ)/∂w_ij = ∂/∂w_ij (1) = 0. The baseline term is dead. Exactly the same Σ_(probabilities)=1 ⇒ Σ_(derivatives)=0 fact, just attached to a unit's output distribution instead of the policy's action distribution. Williams uses it to prove the expected REINFORCE update has nonnegative inner product with the true reward gradient (and equals it with a shared learning rate), and that it survives delayed reward by unfolding the recurrent net in time. So the baseline-invariance I just derived for the policy gradient is the multi-step lift of his single-step result.

Good — so I can ascend the gradient from sampled returns with a baseline. But the whole motivation was to *use a learned value function* to cut variance, the way actor–critic methods do, because the raw return is too noisy. The honest danger: if I substitute a learned, *approximate* Q for the true Q^π in the gradient, I've introduced bias, and there's no reason a biased direction climbs ρ. People have only managed weaker guarantees — for the special tabular-POMDP case you can assure a *positive inner product* with the gradient, which at least guarantees improvement, but not equality, and not for general approximators. I want to know: is there a class of approximators f_w I can plug in *in place of Q^π* and get the gradient *exactly*, not approximately?

Let me set it up. I'll learn f_w(s,a) to approximate Q^π by following π and doing the natural thing — minimize the squared error to some unbiased estimate Q̂^π of Q^π (say the return), updating Δw ∝ [Q̂^π(s,a) − f_w(s,a)] ∂f_w(s,a)/∂w. When this least-squares process has settled at a local optimum, the gradient of the error with respect to w is zero in expectation, i.e.

  Σ_s d^π(s) Σ_a π(s,a) [ Q^π(s,a) − f_w(s,a) ] ∂f_w(s,a)/∂w = 0.   (3)

That's all "trained to convergence" buys me: the residual Q^π − f_w is orthogonal, under the on-policy measure, to ∂f_w/∂w. Now I want the gradient Σ_s d^π Σ_a (∂π/∂θ) Q^π to equal Σ_s d^π Σ_a (∂π/∂θ) f_w. The difference between them is Σ_s d^π Σ_a (∂π/∂θ)[Q^π − f_w]. If I could show *that* is zero, I'd be done. Compare it to (3): (3) has the residual dotted with ∂f_w/∂w; the thing I need zero has the residual dotted with ∂π/∂θ. These match if and only if ∂f_w/∂w and ∂π/∂θ are *the same direction*, up to the per-action factor that turns one sum into the other.

Stare at the two sums. (3) is weighted by π(s,a) ∂f_w/∂w. My target difference is weighted by ∂π/∂θ. Write ∂π/∂θ = π(s,a) · (∂π/∂θ)/π(s,a) = π(s,a) ∇_θ log π(s,a). Then the target difference is Σ_s d^π Σ_a π(s,a) ∇_θ log π(s,a) [Q^π − f_w], and (3) is Σ_s d^π Σ_a π(s,a) ∂f_w/∂w [Q^π − f_w]. They are *identical* if

  ∂f_w(s,a)/∂w = ∇_θ log π(s,a) = (∂π(s,a)/∂θ) (1/π(s,a)).   (4)

So *demand* that. Require the critic's features — its gradient with respect to its own parameters — to equal the policy's score. Call it the compatibility condition. Under it, (3) becomes

  Σ_s d^π(s) Σ_a [∂π(s,a)/∂θ] [ Q^π(s,a) − f_w(s,a) ] = 0,   (6)

which says exactly that the approximation error is orthogonal, under d^π, to the gradient of the policy parameterization. And since (6) is zero, I can subtract it from the policy gradient theorem at no cost:

  ∂ρ/∂θ = Σ_s d^π Σ_a (∂π/∂θ) Q^π
        = Σ_s d^π Σ_a (∂π/∂θ) Q^π − Σ_s d^π Σ_a (∂π/∂θ)[Q^π − f_w]
        = Σ_s d^π Σ_a (∂π/∂θ)[ Q^π − Q^π + f_w ]
        = Σ_s d^π(s) Σ_a [∂π(s,a)/∂θ] f_w(s,a).   (5)

The learned f_w, when it's compatible and trained to the least-squares fixed point, can replace the true, unknown Q^π in the gradient *with equality*. The approximation error doesn't leak into the gradient because the error is forced orthogonal to the only direction the gradient cares about — the score. That gives me the critic I was looking for: one that earns the right to stand in for Q^π exactly.

Now, what does the compatibility condition (4) actually constrain? Let me work it out for the natural policy class, a soft-max / Gibbs distribution over linear action preferences: π(s,a) = exp(θ^T φ_sa) / Σ_b exp(θ^T φ_sb), where φ_sa is a feature vector for the state-action pair. Compute the score. log π(s,a) = θ^T φ_sa − log Σ_b exp(θ^T φ_sb), so

  ∇_θ log π(s,a) = φ_sa − Σ_b [ exp(θ^T φ_sb) / Σ_c exp(θ^T φ_sc) ] φ_sb = φ_sa − Σ_b π(s,b) φ_sb.

So (4) demands ∂f_w/∂w = φ_sa − Σ_b π(s,b) φ_sb. The right-hand side is the feature vector *centered* by its policy-average over actions. Integrating, f_w must be linear in exactly those centered features:

  f_w(s,a) = w^T [ φ_sa − Σ_b π(s,b) φ_sb ].

The critic has to be linear in the *same* features as the policy, mean-subtracted within each state. (Tsitsiklis points out this linear form may be the only way to meet the condition at all.) And look at what that mean-subtraction forces: Σ_a π(s,a) f_w(s,a) = w^T Σ_a π(s,a)[φ_sa − Σ_b π(s,b)φ_sb] = w^T[Σ_a π φ_sa − Σ_b π φ_sb] = 0. The compatible f_w has *zero mean under the policy in every state*. So it can't be estimating Q^π in absolute terms — its state-average is pinned to zero, whereas the average of Q^π over actions is V^π, generally nonzero. What it's really estimating is the *advantage* A^π(s,a) = Q^π(s,a) − V^π(s) — the relative value of each action within the state. And that's exactly right and even reassuring: the convergence condition (3) only ever asked f_w to get the *relative* ranking of actions in each state correct, not their absolute level, and not how value varies from state to state. The absolute level is precisely what the baseline argument said the gradient ignores. The two facts are the same fact: anything that's constant in the action — a per-state offset v(s) — drops out, because Σ_a ∂π/∂θ = 0. I can even add an arbitrary v(s) into (5), ∂ρ/∂θ = Σ_s d^π Σ_a (∂π/∂θ)[f_w(s,a) + v(s)], and it changes nothing in expectation while letting me pick v ≈ V^π to control variance. The compatible critic is an advantage estimator, and the freedom in its absolute level is the baseline.

Now I can close the loop the value-function approach could never close — an actual convergence guarantee for policy iteration with general function approximation. Iterate: at step k, fit the compatible critic to the least-squares fixed point (3) for the current policy π_k, then take a gradient step θ_{k+1} = θ_k + α_k Σ_s d^{π_k}(s) Σ_a (∂π_k/∂θ) f_{w_k}(s,a). By the result I just proved, that step is in the direction of the true performance gradient ∂ρ(π_k)/∂θ — not approximately, exactly. So this is honest gradient ascent on ρ. The remaining work is the standard approximation bookkeeping: I need the step sizes to satisfy α_k → 0 and Σ_k α_k = ∞ (shrink, but slowly enough to keep making progress), and I need the gradient to be smooth enough — bounded second derivatives ∂²π/∂θ_i∂θ_j, together with bounded rewards, give bounded second derivatives ∂²ρ/∂θ_i∂θ_j. Those are exactly the hypotheses of the standard convergence proposition. Apply it: the performance sequence {ρ(π_k)} converges and ∂ρ(π_k)/∂θ → 0. The guarantee is convergence to a stationary local optimum of the actual objective, with the caveat that the theorem is about the idealized iteration whose compatible critic is solved to the fixed point at each policy. That is the thing the value-function-plus-greedy approach could not deliver, because its policy was a discontinuous function of the estimates — here θ moves smoothly and the update is the true gradient.

Let me say the causal chain back to myself in one breath. The greedy-on-values policy is discontinuous in its parameters and can diverge, so I parameterize a smooth stochastic policy and try to ascend ∂ρ/∂θ; the obstacle is that ρ depends on θ through the unknown state distribution, and differentiating that is hopeless. But differentiating the value function instead of ρ exposes the dynamics as a one-step transport of ∂V/∂θ, and averaging against the on-policy state weighting — stationary in one case, discounted in the other — makes that transport cancel its own copy or unroll into a γ-weighted occupancy, so ∂d^π/∂θ disappears and the gradient is just Σ_s d^π Σ_a (∂π/∂θ)Q^π. Because I get the d^π weighting by acting, with γ^t weights in the discounted start-state case, and because (1/π)∂π/∂θ = ∇log π turns the action sum into an expectation under π, the gradient is estimable from sampled trajectories; it becomes REINFORCE when I plug in the raw return. Since Σ_a ∂π/∂θ = 0, any state-only baseline subtracts out unbiasedly, cutting variance — the same zero that makes Williams' reinforcement baseline free. And if I force a learned critic to satisfy ∂f_w/∂w = ∇log π and train it to its least-squares fixed point, its error is orthogonal to the score, so it replaces the unknown Q^π in the gradient exactly — making the compatible critic an advantage estimator and turning the idealized loop into genuine gradient ascent whose performance converges and whose gradient goes to zero.

So I can finally write the objects I was trying to get:

  Policy Gradient Theorem (average-reward or start-state):
    ∂ρ/∂θ = Σ_s d^π(s) Σ_a [∂π(s,a)/∂θ] Q^π(s,a)            — no ∂d^π/∂θ term.

  Score-function form (its sampling consequence):
    ∂ρ/∂θ = Σ_s d^π(s) E_{a∼π(·|s)}[ ∇_θ log π(a|s) ( Q^π(s,a) − b(s) ) ],   any b(s) (e.g. V^π(s)),
    trajectory update: Δθ ∝ Σ_t γ^t ∇_θ log π(a_t|s_t) ( R_t − b(s_t) ) in the discounted start-state case, with no γ^t occupancy weight in the average-reward stationary case.

  Compatible function approximation (Theorem 2):
    if  ∂f_w(s,a)/∂w = ∇_θ log π(s,a)   and   Σ_s d^π Σ_a π[Q^π − f_w]∂f_w/∂w = 0,
    then  ∂ρ/∂θ = Σ_s d^π(s) Σ_a [∂π(s,a)/∂θ] f_w(s,a),
    with the natural compatible form for a soft-max policy
      f_w(s,a) = w^T [ φ_sa − Σ_b π(s,b) φ_sb ]   (linear, state-centered ⇒ an advantage).

  Convergent policy iteration (Theorem 3): with the compatible critic at its fixed point,
    θ_{k+1} = θ_k + α_k Σ_s d^{π_k}(s) Σ_a [∂π_k(s,a)/∂θ] f_{w_k}(s,a),
    α_k→0, Σα_k=∞, bounded ∂²π and rewards  ⇒  ρ(π_k) converges, ∂ρ(π_k)/∂θ → 0.
