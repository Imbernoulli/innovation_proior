Let me start from what actually hurts. I have a fixed pile of expert demonstrations on a continuous-control task — a few dozen trajectories of some MuJoCo-ish agent walking — and I want a *reward function* out of them, not just a policy. Why insist on the reward when imitation methods already reproduce the policy cheaply? Because the thing I really care about is re-using what the expert wanted in a *different* environment: same intent, changed dynamics — a re-articulated body, a different transition structure — where I'd re-run RL from scratch under the recovered reward. A policy is useless for that; it's tied to the dynamics it was trained in. A reward, in principle, is portable. So the target is a reward I can lift out of one MDP and re-optimize in another and still get good behavior. Whether "portable" actually holds is the part I'm least sure of, so let me first get *any* reward at all out of demonstrations, then come back and stress-test transfer rather than assume it.

The framework I trust for "infer reward from demos" is maximum causal entropy IRL. The reason I trust it: plain IRL is hopelessly ill-posed — infinitely many rewards make a given demonstration set optimal — and the max-entropy principle picks a unique answer by saying the demonstrations come from a policy that is as random as possible subject to achieving the observed return. Concretely it models trajectories as an energy-based distribution where higher reward means exponentially more probable: with the dynamics and initial-state distribution pinned to the real MDP and only the reward parametrized,

  p_theta(tau) ∝ p(s_0) · prod_t p(s_{t+1}|s_t,a_t) · exp{ sum_t gamma^t r_theta(s_t,a_t) },

which under deterministic dynamics is just p_theta(tau) ∝ exp{ sum_t gamma^t r_theta(s_t,a_t) }. And there's a fact I'll lean on hard later: the optimal policy in this entropy-regularized MDP is Boltzmann in the soft Q-function, pi*(a|s) ∝ exp{ Q*_soft(s,a) }, so once the soft value V* normalizes it, pi*(a|s) = exp{ Q*(s,a) - V*(s) } = exp{ A*(s,a) }. The log of the optimal policy *is* the advantage. Hold that thought.

IRL is then maximum likelihood: max_theta E_{tau~D}[ log p_theta(tau) ]. Let me actually take the gradient and see where it breaks, because I bet it breaks at the normalizer.

  d/dtheta log p_theta(tau) = sum_t gamma^t d/dtheta r_theta(s_t,a_t) - d/dtheta log Z_theta,

and the derivative of log Z is the expectation of the same quantity under the model itself:

  d/dtheta J(theta) = E_D[ sum_t gamma^t d r_theta ] - E_{p_theta}[ sum_t gamma^t d r_theta ].

A positive phase that pushes reward up on demonstrated transitions, a negative phase that pushes it down on transitions the *current model* thinks are likely. Classic energy-based model gradient. And there's the wall: the negative phase is an expectation under p_theta, and to sample p_theta I'd need to sample from exp{sum_t gamma^t r}/Z, i.e. I need Z, the partition function over all trajectories — intractable for anything continuous and high-dimensional. Every MaxEnt IRL method lives or dies by how it estimates that negative phase.

The known way around it is to learn an adaptive sampler. Marginalize the gradient per timestep first — let p_{theta,t}(s_t,a_t) be the state-action marginal of p_theta at time t — so

  d J = sum_t gamma^t ( E_D[ d r_theta(s_t,a_t) ] - E_{p_{theta,t}}[ d r_theta(s_t,a_t) ] ).

I can't draw from p_{theta,t}, so I bring in a sampling distribution mu and importance-weight. The standard, variance-reducing choice is a mixture mu = (1/2) pi + (1/2) p_hat of a learned policy pi and a rough density estimate p_hat fit to the demonstrations — the p_hat half keeps the importance estimate sane in early training when pi barely covers the expert. Then

  d J = sum_t gamma^t ( E_D[ d r_theta ] - E_{mu_t}[ (p_{theta,t}/mu_t) · d r_theta ] ).   (call this the cost gradient)

And I improve the sampler pi by pulling it toward the model, minimizing KL(pi(tau) || p_theta(tau)); when I expand that KL the dynamics and initial-state factors are identical in pi(tau) and p_theta(tau) and cancel, leaving

  max_pi E_pi[ sum_t gamma^t ( r_theta(s_t,a_t) - log pi(a_t|s_t) ) ],   (the entropy-regularized policy objective)

which is exactly "maximize the learned reward plus the policy's entropy." So the recipe is: alternate a reward (cost) update against demonstrations-vs-sampler, and a policy update that makes the sampler hug the model — the guided-cost-learning loop. It works on paper, but it estimates the cost gradient with explicit importance weights over *whole trajectories*, and I can already see where that bites. An importance weight over a length-T trajectory is a product of T per-step ratios; let me sanity-check how badly that blows up. If each per-step ratio has even a modest log-variance and the steps are roughly independent, the log of the product has variance ~T times that, so the weight's variance grows exponentially in T. For a locomotion episode with T in the hundreds, a handful of mismatched steps and the whole trajectory's weight is either astronomically large or numerically zero, and the negative-phase estimate is dominated by one or two samples. That's not a tuning issue, it's the horizon itself. I'll come back to it.

There's a cleaner way to *write* this same computation that I want to understand, because it removes the explicit importance weights: cast it as a GAN. In a GAN, for a fixed generator with density q, the optimal discriminator is D*(x) = p(x)/(p(x)+q(x)). Normally D is a free classifier that learns this ratio. But here I *know* q — the generator is my policy, and I can evaluate pi(tau). So instead of making D learn the whole ratio, I plug q in and let the discriminator only model the data density. With the data density taken as the Boltzmann form (1/Z)exp{...}, the discriminator becomes

  D_theta(tau) = exp{ f_theta(tau) } / ( exp{ f_theta(tau) } + pi(tau) ),

i.e. a sigmoid whose input is f_theta(tau) - log pi(tau): the learned term minus the *filled-in* log generator density. Two things this buys me. First, the optimal discriminator is now independent of the generator — it's optimal exactly when exp{f_theta} matches the data density up to Z — which stabilizes the adversarial game compared to a discriminator that has to chase a moving generator. Second — and this is what I want to test, not just hope for — maybe I can read a *reward* back out of f_theta, which a generic GAN classifier can't (its discriminator is 1/2 everywhere at optimum and tells me nothing about reward). Whether the structured discriminator is genuinely the same object as guided cost learning is a claim I should check, not wave at, so let me line up the gradients. Differentiating this discriminator's cross-entropy loss in theta, and minimizing over the scalar Z, the Z that minimizes the loss is the importance-sampling estimate of the partition function; substituting it back, the loss gradient equals the cost gradient I derived above with f in the reward's place. So this GAN reproduces the guided-cost-learning update, with the importance weighting absorbed into the discriminator rather than written out — I'll re-derive this carefully once I'm working with single transitions, where I actually need it to hold. The catch is that everything here is still over trajectories tau: the discriminator's logit is a sum over the whole episode, so the per-step-ratio variance I just estimated rides along, and the gradient is too noisy to be usable on continuous control.

So the obvious fix: stop discriminating trajectories, discriminate single transitions. Convert the discriminator to per state-action:

  D_theta(s,a) = exp{ f_theta(s,a) } / ( exp{ f_theta(s,a) } + pi(a|s) ).

Now every sample is one (s,a) pair, the logit is f_theta(s,a) - log pi(a|s), no product over T, variance no longer compounds with the horizon. That fixes the variance, but it might have cost me the principled objective — dropping from trajectories to transitions is exactly the kind of move that quietly turns a maximum-likelihood method into a heuristic. I'm about to build everything on this object, so I need to know whether its gradient still equals the MaxEnt IRL gradient or only resembles it. Let me grind it out. For this algebra I can suppress the outer gamma^t weights; they only reweight the time-indexed expectations and do not change the per-sample ratio. The discriminator's objective (negative loss, to make it a maximization, with mu a mixture of data and policy) is

  -L(theta) = sum_t E_D[ log D_theta(s_t,a_t) ] + E_pi[ log(1 - D_theta(s_t,a_t)) ].

Substitute the form of D. The first term: log D = log exp{f} - log(exp{f}+pi) = f - log(exp{f}+pi). The second: 1 - D = pi/(exp{f}+pi), so log(1-D) = log pi - log(exp{f}+pi). Adding the two expectations,

  -L = sum_t E_D[ f_theta(s_t,a_t) ] + E_pi[ log pi(a_t|s_t) ] - 2 E_{mu_bar}[ log( exp{f_theta(s_t,a_t)} + pi(a_t|s_t) ) ],

where mu_bar is the half-and-half mixture and the factor 2 comes from the two expectations sharing the same log-normalizer term. Differentiate in theta. The E_D[f] term gives E_D[ d f ]. The log-normalizer term: d/dtheta log(exp{f}+pi) = exp{f}/(exp{f}+pi) · d f, and the 2·(1/2)-mixture average turns the coefficient into

  d(-L) = sum_t E_D[ d f_theta ] - E_{mu_t}[ ( exp{f}/((1/2)exp{f} + (1/2)pi) ) · d f_theta ].

Now multiply the numerator and denominator of that fraction by the policy's state marginal rho_pi(s_t) = ∫_a rho_pi(s_t,a) da. Writing p_hat_{theta,t}(s,a) = exp{f_theta(s,a)} · rho_pi(s), and letting mu_hat be the mixture of p_hat_theta and policy samples, the gradient collapses to

  d(-L) = sum_t E_D[ d f_theta ] - E_{mu}[ ( p_hat_{theta,t}/mu_hat_t ) · d f_theta ].

That is, term for term, the cost gradient I derived for guided cost learning, with f_theta as the reward, holding once the policy maximizes its objective so p_hat_theta = p_theta. The substitution I was nervous about — multiplying through by rho_pi(s) to turn the bare mixture denominator into the p_hat/mu_hat importance ratio — is the only non-mechanical step, and it goes through because rho_pi(s) is exactly the state marginal that converts exp{f} into a state-action density. So dropping to single transitions did *not* cost me the objective: the single-(s,a) discriminator's gradient is the MaxEnt IRL gradient, not a lookalike, and the variance is the only thing that changed. That was the thing I actually needed to confirm before going further.

Now, what reward do I hand the policy? In the GAN game the policy should maximize "confuse the discriminator," and the natural reward is

  r_hat(s,a) = log D_theta(s,a) - log(1 - D_theta(s,a)).

Let me simplify it, because I suspect it collapses to something familiar.

  r_hat = log [ exp{f}/(exp{f}+pi) ] - log [ pi/(exp{f}+pi) ]
        = [ f - log(exp{f}+pi) ] - [ log pi - log(exp{f}+pi) ]
        = f_theta(s,a) - log pi(a|s).

The shared - log(exp{f}+pi) normalizer cancels and I'm left with r_hat = f - log pi. Worth pinning down numerically so I don't carry an algebra slip into the reward: take f = 1.3, pi = 0.2. Then exp{f} = 3.669, D = 3.669/3.869 = 0.9483, log D - log(1-D) = log(0.9483) - log(0.0517) = -0.0531 - (-2.962) = 2.909, and f - log pi = 1.3 - (-1.609) = 2.909. They match. Summed over a trajectory with the same outer discounts, E_pi[ sum_t gamma^t (f_theta - log pi) ] is precisely the entropy-regularized policy objective from guided cost learning — f as reward, minus log pi as the entropy term. So the policy step in this GAN coincides with the MaxEnt sampler update: discriminator update = reward update, policy update = sampler update, both from the same single-transition adversarial game. And the - log pi is the same "fill in the generator density" move from the trajectory case, now per state-action.

So I have a scalable, low-variance imitation algorithm whose gradient I've checked is the MaxEnt IRL one. Now the question I actually care about: what reward does it recover? I claimed the structured discriminator yields a reward where a generic one yields nothing, so let me find where f ends up at convergence. The GAN reaches its global optimum when the generator matches the data, pi = pi_E, the expert policy, and there the optimal discriminator is 1/2 for every (s,a). Setting D_theta(s,a) = 1/2 in exp{f}/(exp{f}+pi) = 1/2 forces exp{f*(s,a)} = pi_E(a|s), i.e.

  f*(s,a) = log pi_E(a|s).

And from the MaxEnt fact I parked earlier — pi*(a|s) = exp{A*(s,a)} — taking the log gives log pi_E(a|s) = A*(s,a), the soft advantage of the optimal (expert) policy. So f* lands on the advantage function. That is a *valid* reward — re-optimizing under A* reproduces the expert in the training MDP — and it is already more than GAIL gives, since GAIL's unstructured discriminator is 0.5 everywhere at optimum with nothing to extract. So the structured-discriminator claim holds: I have a reward, not just a policy. But "valid in the training MDP" and "portable" are different properties, and I deferred portability at the very start; time to see whether the advantage actually survives a dynamics change or only looks like it should.

But now I have to confront "portable," and this is where I hit the real wall. The recovered reward is the advantage, and the advantage is a deeply *entangled* object. It grades each action by how good that action is relative to the *optimal policy in the training MDP* — it has the training dynamics baked into it through V* and through the way Q* couples reward and transitions. If I take A* to a new environment with different dynamics and re-optimize, A* is no longer the advantage for *that* MDP, and the policy it induces can be wrong. So "recover the advantage" gives me a reward that reproduces the expert here but doesn't transfer. I need to understand precisely *why* it doesn't transfer, because the fix has to attack that mechanism.

Let me think about what reward learning can possibly pin down. There's an old result about reward shaping: the transformation

  r_hat(s,a,s') = r(s,a,s') + gamma · Phi(s') - Phi(s)

leaves the optimal policy unchanged for *any* potential Phi, and — crucially — without knowing the dynamics, this potential-based class is the *only* class of reward transformations that is policy-invariant. Stare at what that means for IRL. I only ever observe demonstrations from an optimal agent. Optimal behavior is invariant to adding any gamma Phi(s') - Phi(s). So from demonstrations alone I fundamentally cannot distinguish the true reward r from any shaped r_hat in this class — they explain the data identically. The advantage I recovered is exactly such a shaped reward: A*(s,a) = Q*(s,a) - V*(s) = r(s) + gamma V*(s') - V*(s) under deterministic dynamics, which is r shaped by the potential Phi = V*. So "f* = advantage" is the symptom; "IRL can't see past potential-based shaping" is the disease.

And shaping is precisely what kills transfer. Take deterministic dynamics T and a shaped reward r_hat(s,a) = r(s,a) + gamma Phi(T(s,a)) - Phi(s). It depends on the *successor state* T(s,a), i.e. on the dynamics. Change T to T' with T'(s,a) ≠ T(s,a), and r_hat is no longer of the form r + gamma Phi(s') - Phi(s) for the new MDP M' — the s' it references is the old one. So r_hat is no longer policy-invariant under M', and re-optimizing it in M' gives a different, generally worse policy. The shaping that was harmless in training becomes harmful under new dynamics. That's the mechanism. So the goal sharpens: I must restrict the learnable reward class so that the only thing I can recover is the *unshaped* reward, stripped of the dynamics-dependent potential.

What restriction does that? Let me define what I even mean by "portable." Call a reward r' *disentangled* with respect to the ground-truth r and a set of dynamics if, under every dynamics T in the set, optimizing r' gives the same optimal policy as optimizing r: pi*_{r',T} = pi*_{r,T} for all T. Under max causal entropy, policies and Q-functions are equivalent representations up to an arbitrary function of state — pi determines Q only modulo adding f(s), since adding f(s) to Q shifts every action equally and the Boltzmann normalization erases it. So "same optimal policy under T" is equivalent to Q*_{r',T}(s,a) = Q*_{r,T}(s,a) - f(s) for some state function f. That's my working condition.

Now I'll guess the restriction is "make the reward depend only on the current state," r'(s), and prove both directions: that state-only is *sufficient* for disentanglement, and that it's *necessary*. I need one structural condition on the dynamics first, because the proofs will require separating a function of s from a function of s'. Call dynamics *decomposable* if all states are "linked": two states s1, s2 are 1-step linked if some state can reach both with positive probability in one step, extend by transitivity, and require that every state links to every other. This holds, e.g., if every state has a self-transition in an ergodic MDP, and it holds in the continuous-control environments I care about. The reason I need it is a chaining lemma: if for all s, s'

  a(s) + b(s') = c(s) + d(s'),

then a(s) = c(s) + C1 and b(s) = d(s) + C2 for constants C1,C2. Proof: rearrange to a(s) - c(s) = d(s') - b(s'). The left side depends only on s, the right only on s'. Fix s; the right side d(s') - b(s') must take the *same* value for every successor s' reachable from s, since the left side is a single number. Decomposability links all states through chains of shared successors, so d(s') - b(s') is forced to be constant across all states; hence a(s) - c(s) is constant for all s, and substituting back makes b(s) - d(s) constant as well. That's the tool.

I try the easier direction first. Let r be a state-only ground truth, T decomposable, and suppose IRL recovers a state-only r'(s) that produces the optimal policy in T, i.e. Q*_{r',T}(s,a) = Q*_{r,T}(s,a) - f(s). I need this to force r' = r + const; then r' is disentangled for *all* dynamics, because a state-only reward equal to r up to a constant induces r's optimal policy under any dynamics. Write r'(s) = r(s) + phi(s) for some state function phi, and let V_r(s) = log sum_a exp Q*_r(s,a). The soft Bellman equation for r is

  Q*_r(s,a) = r(s) + gamma E_{s'}[ V_r(s') ].

Subtract f(s) from both sides:

  Q*_r(s,a) - f(s) = r(s) - f(s) + gamma E_{s'}[ V_r(s') ].

Now I want the right side to look like a soft Bellman equation for Q*_r - f, so I add and subtract gamma E_{s'}[f(s')]. If Q*_{r'}(s',a') = Q*_r(s',a') - f(s'), then its soft value is

  V_{r'}(s') = log sum_{a'} exp( Q*_r(s',a') - f(s') )
             = V_r(s') - f(s'),

because f(s') is constant across a'. Therefore

  Q*_r(s,a) - f(s)
    = [ r(s) + gamma E_{s'}[f(s')] - f(s) ] + gamma E_{s'}[ V_{r'}(s') ].

The left side is, by assumption, Q*_{r'}(s,a). This is the soft Bellman equation for Q*_{r'} with immediate reward equal to the bracket. So the reward that generates Q*_{r'} is

  r'(s) = r(s) + gamma E_{T(.|s,a)}[f(s')] - f(s).

But I also wrote r'(s) = r(s) + phi(s), so for every action from s,

  phi(s) = gamma E_{T(.|s,a)}[f(s')] - f(s).

The left side is state-only, so the right side cannot depend on which action from s I take. In the deterministic/support-wise version that the decomposability argument uses, the expectation collapses on each reachable successor and I can write phi(s) + f(s) = gamma f(s') for every linked transition. The chaining lemma sees this as a(s)=phi(s)+f(s), b(s')=0, c(s)=0, and d(s')=gamma f(s'); since b=d+C2, gamma f(s') is constant across the linked state space. Then f is constant, phi(s) = gamma f - f is constant too, and r'(s) = r(s) + const. State-only is sufficient.

For the other direction, suppose r'(s,a,s') is disentangled for *all* dynamics. The claim is that this forces it to be state-only; the way to test the claim is to take an r' that is *not* state-only but is policy-invariant in one MDP, and see whether a dynamics change can break it. If it can, action/next-state dependence is incompatible with disentanglement. Take a 3-state MDP with deterministic dynamics: a start state S and states A, B. From S, action "a" goes to A, action "b" goes to B (reward 0 either way). The true reward is state-based: leaving A pays +1, leaving B pays -1, leaving S pays 0. Both A and B then return to S, so the agent cycles S -> {A or B} -> S forever. Now an action-dependent r' that shoves the reward onto the outgoing action at S — r'(S,a) = +1, r'(S,b) = -1, zero elsewhere — is policy-invariant here (it's the shaping potential phi(S)=0, phi(A)=1, phi(B)=-1). I want to actually optimize both and compare, so let me make it finite with gamma = 0.9 and solve by value iteration rather than eyeballing "infinite return."

Under the original dynamics, the Bellman backups are: true reward gives Q(S,a) = 0 + 0.9·V(A) and Q(S,b) = 0 + 0.9·V(B), and r' gives Q(S,a) = +1 + 0.9·V(S), Q(S,b) = -1 + 0.9·V(S). Running value iteration to convergence:

  true r:  V = {S: 4.737, A: 5.263, B: 3.263},  Q(S,a)=4.737 > Q(S,b)=2.937  ->  pi(S) = a
  r':      V = {S: 5.263, A: 4.737, B: 4.737},  Q(S,a)=5.263 > Q(S,b)=3.263  ->  pi(S) = a

Same optimal action at S, so r' is indeed policy-invariant under the original dynamics — that's the trap, it looks like a fine reward. Now permute *only* the dynamics: action "a" now leads to B and "b" leads to A, nothing else changed. Re-solve:

  true r:  Q(S,a)=2.937 < Q(S,b)=4.737  ->  pi(S) = b   (correctly chases A, now reached via "b")
  r':      Q(S,a)=5.263 > Q(S,b)=3.263  ->  pi(S) = a   (still rewards action "a")

The true reward re-routes to the good state; r' keeps picking action "a", which under the new dynamics dumps the agent into B, the bad state. The two optimal policies disagree, so r' is *not* disentangled — exactly what its action-dependence let the dynamics change exploit. (I should be honest that this exhibits one offending r' rather than proving the universal claim, but it's enough to rule out "action/next-state dependence can be disentangled": at least one such reward fails, and the structure of the failure — the reward referencing an action whose consequence the new dynamics changed — is general.) Combined with the sufficiency direction, the restriction that yields a transferable reward is to make the recovered reward a function of state alone.

Now I have a target and a wall between them. Target: recover a state-only g(s) that equals the true reward up to a constant. Wall: my discriminator recovers f* = the advantage = r shaped by V*, which is *not* state-only and not transferable. If I simply force f_theta = g_theta(s) to be state-only, I break the equivalence — f* is supposed to equal the advantage A*(s,a), which genuinely depends on a, so a state-only f can't hit it and the discriminator can't reach its optimum. So I can't just amputate the action-dependence; I have to give the network a place to *put* the shaping so that what's left in g is clean.

And I already know the exact shape of "the shaping": the only policy-invariant degree of freedom is gamma Phi(s') - Phi(s). So let me give the discriminator's function f a structure that explicitly carves out a potential-shaping term with its own network h, and a reward term g:

  f_{theta,phi}(s,a,s') = g_theta(s,a) + gamma · h_phi(s') - h_phi(s).

This is precisely the Ng-shaping form, with h playing the role of the potential Phi. The idea is that whatever shaping the optimization wants to apply, it can dump entirely into h, leaving g to be the unshaped reward. And then — the move that makes it transferable — I restrict g to be a function of state only, g_theta(s). Now the discriminator has both the freedom it needs (the h(s') term restores enough action/next-state dependence for f to match the advantage) and the restriction I want (g is state-only, hence disentangled).

Does this actually recover the true reward? Let me prove it for the clean case: deterministic dynamics so s' is determined by (s,a), the decomposability condition, ground-truth reward state-only r(s), and g restricted to g(s). At the GAN optimum I still have f* = A*. Under deterministic dynamics the advantage is

  A*(s,a) = Q*(s,a) - V*(s) = [ r(s) + gamma V*(s') ] - V*(s) = r(s) + gamma V*(s') - V*(s).

So setting my structured f equal to A* gives, for all s and the determined s',

  g*(s) + gamma h*(s') - h*(s) = r(s) + gamma V*(s') - V*(s).

This is exactly the shape the chaining lemma eats. Group: a(s) = g*(s) - h*(s), b(s') = gamma h*(s'), c(s) = r(s) - V*(s), d(s') = gamma V*(s'). Then a(s) + b(s') = c(s) + d(s') for all s, s', so by the lemma a = c + C1 and b = d + C2, with C1 and C2 constants:

  gamma h*(s') = gamma V*(s') + C2   =>   h*(s) = V*(s) + C2/gamma,
  g*(s) - h*(s) = r(s) - V*(s) + C1   =>   g*(s) = r(s) + C1 + C2/gamma.

So g* recovers the true reward up to a constant and h* recovers the optimal value function up to a constant. Before I trust that, I want to run it forward on a concrete soft MDP — pick g and h equal to the claimed solution and check the structured f actually reproduces the advantage, because the proof shows g,h *exist* but I want to see the numbers close. Two states {0,1}, two actions, deterministic decomposable dynamics T(0,0)=0, T(0,1)=1, T(1,0)=0, T(1,1)=1 (every state reachable from every state), state-only reward r=(1.0, -0.5), gamma=0.9. Soft value iteration gives V* = (12.305, 10.805), and the soft advantage A*(s,a) = Q*(s,a) - V*(s) comes out to {(0,0): -0.230, (0,1): -1.580, (1,0): -0.230, (1,1): -1.580}. Now set g = r and h = V* exactly (constants zero) and evaluate the structured score f(s,a) = g(s) + gamma·h(s') - h(s) on all four (s,a):

  (0,0): 1.0 + 0.9·12.305 - 12.305 = -0.230   vs A* = -0.230
  (0,1): 1.0 + 0.9·10.805 - 12.305 = -1.580   vs A* = -1.580
  (1,0): -0.5 + 0.9·12.305 - 10.805 = -0.230   vs A* = -0.230
  (1,1): -0.5 + 0.9·10.805 - 10.805 = -1.580   vs A* = -1.580

Every f matches A* to the digit — max error 0. So plugging g=r, h=V* into the shaped form really does reconstruct the advantage the discriminator converges to; the proof isn't just an existence statement, the constructed pair lands on it numerically. (If I shift g by c1 and h by c2 the four f's all shift by the same overall constant c1 + (gamma-1)c2, which is the harmless additive freedom the discriminator has anyway.) The reason it works out: f* = r(s) + gamma V*(s') - V*(s) is just the advantage rewritten with Q = r + gamma V' and V = V* — the value function *is* the shaping that turns a reward into an advantage, so a learnable potential h placed in the discriminator absorbs all of it and leaves g clean. In stochastic environments s' isn't determined by (s,a), so f(s,a,s') is instead a single-sample estimate of A*(s,a), but the structure is the same.

One subtlety on the shaping term before I code it. The potential identity gamma Phi(s') - Phi(s) preserves the optimal policy for any Phi — but only if I'm honest about terminal states. At an episode's final transition there is no genuine s' in the MDP; if I let h(s') fire on a terminal "next state" I'm adding a phantom potential that the real value function would set to zero (a terminal state has zero future value). With variable-length episodes, a nonzero terminal potential breaks the policy-invariance guarantee. So I zero the shaping when the transition is terminal: use gamma · (1 - done) · h(s') - h(s). That keeps f a valid potential-shaped reward across episodes of different lengths.

Now the training loop, which is the GAN game I derived. Alternate: collect rollouts from the current policy pi; train the discriminator by binary logistic regression to tell expert transitions from policy transitions, where the discriminator's logit is f_{theta,phi}(s,a,s') - log pi(a|s) (label expert = 1, policy = 0); turn the discriminator into the per-step reward r_hat = log D - log(1-D) = f - log pi and hand it to a policy optimizer (PPO/TRPO with entropy regularization); repeat. The logit being f - log pi is the whole reason the optimal discriminator is generator-independent and the reward is extractable — it's the structured discriminator, not a free classifier.

A couple of practical things the math doesn't force but training demands. IRL rewards tend to "forget": the reward explains the *current* policy locally and overfits to this iteration's samples, so a reward grabbed at the end is hard to re-optimize from scratch. I mix in policy samples from the previous several iterations as additional negatives when training the discriminator, which broadens the negative distribution and stops the reward from collapsing onto the latest rollout. And because the discriminator can cheat by classifying expert vs. policy on the raw observation *scale* alone (early policies visit states with very different magnitudes than the expert), I normalize the reward network's observation inputs with a running mean/std. These are stabilizers around the structured discriminator, not changes to it.

Let me write it as code that fills the empty slots: a reward network whose forward pass is the shaped score f = g + gamma(1-done)h(s') - h(s), and an adversarial trainer whose discriminator logit is exactly f - log pi, with expert labels equal to one and policy labels equal to zero. The policy-facing sampler reward is the discriminator log-odds, log D - log(1-D) = f - log pi.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim, hidden_sizes=(32, 32), out_dim=1):
    layers = []
    last = in_dim
    for width in hidden_sizes:
        layers += [nn.Linear(last, width), nn.ReLU()]
        last = width
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class RewardNetwork(nn.Module):
    """Shaped score f(s,a,s',done) = g(s) + gamma*(1-done)*h(s') - h(s)."""

    def __init__(self, obs_dim, action_dim, gamma=0.99, use_action=False):
        super().__init__()
        self.gamma = gamma
        self.use_action = use_action
        base_in = obs_dim + action_dim if use_action else obs_dim
        self.base = mlp(base_in, hidden_sizes=(32,))          # g, state-only by default
        self.potential = mlp(obs_dim, hidden_sizes=(32, 32))  # h

    def _base_reward(self, state, action):
        if self.use_action:
            x = torch.cat([state, action], dim=-1)
        else:
            x = state
        return self.base(x).squeeze(-1)

    def _potential(self, state):
        return self.potential(state).squeeze(-1)

    def forward(self, state, action, next_state, done):
        base_reward = self._base_reward(state, action)
        old_shaping = self._potential(state)
        new_shaping = self._potential(next_state)
        new_shaping = (1.0 - done.float()) * new_shaping
        return base_reward + self.gamma * new_shaping - old_shaping

    def unshaped(self, state, action):
        return self._base_reward(state, action)


class IRLAlgorithm:
    """Adversarial reward learner with AIRL's structured discriminator."""

    def __init__(self, reward_net, optimizer):
        self.reward_net = reward_net
        self.optimizer = optimizer

    def logits_expert_is_high(
        self,
        state,
        action,
        next_state,
        done,
        log_policy_act_prob,
    ):
        if log_policy_act_prob is None:
            raise TypeError("AIRL requires log pi(a|s) for the discriminator logit")
        f = self.reward_net(state, action, next_state, done)
        return f - log_policy_act_prob

    def policy_reward(self, state, action, next_state, done, log_policy_act_prob):
        f = self.reward_net(state, action, next_state, done)
        return f - log_policy_act_prob

    def update(self, expert_batch, policy_batch):
        state = torch.cat([expert_batch["obs"], policy_batch["obs"]])
        action = torch.cat([expert_batch["acts"], policy_batch["acts"]])
        next_state = torch.cat([expert_batch["next_obs"], policy_batch["next_obs"]])
        done = torch.cat([expert_batch["dones"], policy_batch["dones"]]).float()
        logp = torch.cat([
            expert_batch["log_policy_act_prob"],
            policy_batch["log_policy_act_prob"],
        ])

        logits = self.logits_expert_is_high(state, action, next_state, done, logp)
        labels = torch.cat([
            torch.ones(len(expert_batch["obs"]), device=logits.device),
            torch.zeros(len(policy_batch["obs"]), device=logits.device),
        ])
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "disc_loss": loss.detach(),
            "disc_acc": ((logits > 0) == labels.bool()).float().mean().detach(),
        }
```

Let me retrace the chain. I wanted a reward, not a policy, because only a reward transfers to new dynamics. MaxEnt IRL gives the principled objective but stalls on the intractable partition function; guided cost learning estimates it with a learned sampler, and that whole computation can be rewritten as a GAN with a structured discriminator whose logit is the learned reward minus the filled-in log generator density, which makes the optimal discriminator generator-independent and lets a reward be read out. Trajectory-level discrimination is too high-variance, so I dropped to single transitions and re-proved that the single-(s,a) discriminator still optimizes the MaxEnt IRL gradient, with the policy reward collapsing to f - log pi. At convergence f recovers the advantage — a valid but entangled reward, because the advantage is the true reward shaped by the value function, and reward shaping is exactly the policy-invariant ambiguity IRL can't see through, and shaped rewards reference the successor state, so they break when dynamics change. Proving that disentanglement is equivalent to state-only rewards (sufficient and necessary, via the chaining lemma under decomposable dynamics) told me the fix: give the discriminator the explicit potential-shaping structure f = g(s) + gamma h(s') - h(s) so the shaping can pour into h. The chaining lemma then forces g* = r* + const and h* = V* + const — the clean reward and the value function it shed — yielding a reward that transfers across dynamics. Done-aware zeroing of the terminal potential keeps the shaping valid across variable-length episodes; previous-iteration negatives and input normalization keep the adversarial training from forgetting or cheating on scale.
