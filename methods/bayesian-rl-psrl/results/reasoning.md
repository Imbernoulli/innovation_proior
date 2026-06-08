OK, let me think this through from scratch. I have an agent in an unknown Markov decision process — finite horizon $\tau$, episodes that reset, a state space of size $S$, $A$ actions, rewards in $[0,1]$. It doesn't know the transition kernel $P^*$ or the reward distribution $R^*$; it has to learn them by acting. And the thing I actually care about is regret: over each episode $k$ I lose $\Delta_k = \sum_s \rho(s)\big(V^{M^*}_{\mu^*,1}(s)-V^{M^*}_{\mu_k,1}(s)\big)$, the gap between the optimal policy's value in the true MDP and the value of whatever policy I actually ran. I want $\sum_k \Delta_k$ to grow like $\sqrt{T}$, not linearly.

The naive thing is to keep a running estimate $\hat M$ of the MDP, plan against it, act greedily, repeat. And I know exactly why that fails. A point estimate pretends I know more than I do. If by bad luck my early samples make some genuinely-good state-action pair look mediocre, the greedy planner will simply never go back there — there's no force pulling me toward pairs I'm uncertain about, only toward pairs that *currently look* best. On something like the RiverSwim chain — small reward at the near end, a big reward at the far end reachable only by repeatedly fighting a current — greedy grabs the small reward and never discovers the big one. Linear regret, forever. So certainty-equivalence is out. The agent has to be made to care about its own ignorance.

The whole field's answer to this is optimism. Build a confidence region around the unknown rewards and transitions, then *inflate* every poorly-understood quantity to the most favorable value that's still statistically plausible, and act optimally against that rosy model. Poorly-visited pairs get the biggest inflation, so the planner is drawn to them; as you gather data the region shrinks and the inflation fades. UCRL2 does exactly this: an $L^1$ ball $\lVert \hat P_a(\cdot|s)-P_a(\cdot|s)\rVert_1 \le \sqrt{14 S\log(2SAt/\delta)/\max\{1,N\}}$ around each transition row, an interval around each reward, and then *extended value iteration* to find the best policy across that whole family of MDPs simultaneously. It gets $\tilde O(D S\sqrt{AT})$ regret, $D$ the diameter. R-max, MBIE, REGAL — all optimism, all with strong bounds.

But stare at what optimism costs. Two things bother me. First, it's statistically greedy in a bad way. To *guarantee* the optimistic model beats the truth, my confidence sets have to hold a worst-case mis-estimation in *every single* state-action pair at once. That simultaneous-worst-case event is far more pessimistic than "my model is roughly right overall" — so the agent explores more than it needs to. Second, and this nags me more, the algorithm is heavy. Planning isn't "solve one MDP" anymore — it's "optimize jointly over a whole family of plausible MDPs," and the confidence sets themselves have to be hand-crafted. REGAL gets a better bound by shrinking $D$ to the span $\Psi$, and yet there's *no tractable implementation of it* — because you can't actually solve the regularized optimization over the family. So the bound and the algorithm got tangled together: people designed the optimism mechanism partly to make the worst-case proof go through, and the proof is now driving the engineering. That feels backwards.

So let me back away from optimism entirely and ask: is there another way to make an agent explore in proportion to its uncertainty, without ever building a confidence set?

Here's an old idea worth dusting off. Thompson, in 1933, faced two treatments with unknown success rates and asked how to allocate patients. His move: let $P$ be the *posterior probability that treatment 1 is the better one*, and send a fraction $f(p)=P$ of patients to it. Not "the treatment that currently looks best" (that's greedy), and not "a hand-tuned bonus for the less-tried treatment" (that's optimism) — but *match the action frequency to the posterior probability of optimality*. And the beautiful thing is his own little calculation: the expected number of patients sacrificed to the inferior treatment under this rule is proportional to $2PQ<1$ with $Q=1-P$, strictly better than committing. The randomness is keyed to the posterior itself. When the data are ambiguous, $P\approx \tfrac12$ and you hedge; as evidence piles up and $P\to 1$, allocation concentrates automatically. No external exploration parameter anywhere.

Can I lift that to MDPs? The cleanest possible version: keep a prior over whole MDPs. At the start of each episode, draw *one* MDP $M_k$ from the posterior conditioned on everything I've seen, solve for its optimal policy $\mu_k=\mu^{M_k}$, and just follow that policy for the episode. That's it. No confidence sets, no inflation, no joint optimization — one sample, one ordinary MDP solve, one policy. Strens actually proposed exactly this in 2000 as "Bayesian Dynamic Programming," but as a heuristic — Kolter and Ng flatly said nobody knows what guarantees, if any, such Bayesian methods have, and every Bayesian method with a guarantee had bolted optimism back on (BOSS samples many MDPs and merges them into an optimistic one; BEB adds a count bonus). So the open question is sharp: does this bare one-sample scheme actually explore efficiently, and can I prove it?

First, does sampling even explore in the right *shape*? Suppose I resampled a fresh MDP, and hence a fresh policy, at *every timestep*. On a long chain where only the two far ends are informative and reaching an end takes $N$ committed steps in one direction, re-randomizing my intended direction each step means I do a random walk — I'm exponentially unlikely to string together $N$ steps the same way, so I essentially never reach an end, and I never learn. That's fatal. The fix is forced by the failure: I must draw the MDP *once per episode* and hold the policy fixed for all $\tau$ steps. Then a single sample commits me to a coherent, temporally-extended plan — swim hard right for the whole episode — which is exactly the kind of exploration a long-horizon problem rewards. Sampling once per episode isn't a detail; it's what makes the exploration "deep." Good — so the algorithm samples $M_k\sim f(\cdot\mid H_{t_k})$ once, plans, and commits.

Now: why should this control regret? Why does drawing from the posterior calibrate exploration the way optimism does, but for free?

Let me think about what I sample. Each episode I pick the policy $\mu_k$ that is optimal for a posterior draw $M_k$. So over the randomness of the draw, *I select each policy with exactly the posterior probability that it is the optimal policy.* Any policy that's still plausibly optimal keeps getting sampled; policies the data have ruled out stop getting sampled. The exploration is driven by the *variance of the posterior*, automatically. As the posterior concentrates on $M^*$, the samples concentrate, and my policy concentrates on $\mu^*$. That's the intuition. But intuition isn't a regret bound. I need to turn "identically distributed sampling" into arithmetic.

There's an obstacle that kills the direct attempt. My regret $\Delta_k$ contains $V^{M^*}_{\mu^*,1}$ — the value of the *optimal policy* under the *true* MDP. I observe neither $\mu^*$ nor $M^*$. There's no clean handle relating $\mu^*$ to the states and actions I actually visit; that's the same wall every analysis hits. I need to replace that unobservable quantity with something I can see.

And this is where the posterior buys me something an optimist has to construct by hand. Conditioned on the history $H_{t_k}$, what is the distribution of $M_k$? It's the posterior. What's the distribution of $M^*$ given the same history? By definition of "posterior" — the posterior *is* my belief about $M^*$ — it's the same distribution. So **$M_k$ and $M^*$ are identically distributed given $H_{t_k}$.** That's the entire trick. It means: for any measurable mapping $g$ from MDPs to numbers that I can choose after seeing $H_{t_k}$,

$$\mathbb{E}\big[g(M^*)\,\big|\,H_{t_k}\big]=\mathbb{E}\big[g(M_k)\,\big|\,H_{t_k}\big].$$

The sampled MDP, which I fully observe, stands in for the true MDP, which I don't — *in expectation*, exactly, for any mapping $g$ I can choose after seeing the history. Take the tower property and the unconditional version $\mathbb{E}[g(M^*)]=\mathbb{E}[g(M_k)]$ falls out too. This is the same identity Russo and Van Roy used in the bandit case — there, conditioned on the history, the sampled action and the optimal action are identically distributed, and any upper-confidence function $U_t$ is deterministic given the history, so $\mathbb{E}[U_t(A^*_t)\mid H_t]=\mathbb{E}[U_t(A_t)\mid H_t]$, and the Bayesian regret of posterior sampling collapses onto the regret form of *any* UCB algorithm. I want the MDP analogue, where the "function of the model" is the whole optimal-value functional.

Let me use it to dissolve the unobservable term. I'll introduce a surrogate tied to the policy I actually run. Define

$$\tilde\Delta_k = \sum_s \rho(s)\big(V^{M_k}_{\mu_k,1}(s) - V^{M^*}_{\mu_k,1}(s)\big).$$

Read it carefully: both terms are about $\mu_k$, the policy I actually run. The first, $V^{M_k}_{\mu_k,1}$, is the value my running policy *claims* for itself under the MDP I sampled — I computed $\mu_k$ as the optimum of $M_k$, so this is just the optimal value of $M_k$, fully known to me. The second, $V^{M^*}_{\mu_k,1}$, is the value that same policy has under the real dynamics — I only see it through the trajectory I execute. So $\tilde\Delta_k$ is "what I thought I'd get minus what the real world gives this policy," the error signal the analysis can follow. Now compare the true regret to this surrogate:

$$\Delta_k-\tilde\Delta_k = \sum_s\rho(s)\big(V^{M^*}_{\mu^*,1}(s)-V^{M_k}_{\mu_k,1}(s)\big).$$

Look at the two surviving terms. $V^{M^*}_{\mu^*,1}$ is the optimal value of the *true* MDP. $V^{M_k}_{\mu_k,1}$ is the optimal value of the *sampled* MDP — because $\mu_k$ is optimal for $M_k$. Both are the *same functional* — "the optimal value from the initial distribution" — applied once to $M^*$ and once to $M_k$. That mapping, call it $g(M)=\sum_s\rho(s)V^{M}_{\mu^M,1}(s)$, is fixed once the history has fixed the posterior. Apply the identity: $\mathbb{E}[g(M^*)\mid H_{t_k}]=\mathbb{E}[g(M_k)\mid H_{t_k}]$, i.e.

$$\mathbb{E}\big[\Delta_k-\tilde\Delta_k\,\big|\,H_{t_k}\big]=0.$$

(And the difference is bounded, $\Delta_k-\tilde\Delta_k\in[-\tau,\tau]$, since values lie in $[0,\tau]$ — I'll want that later.) Summing and taking expectations,

$$\mathbb{E}\Big[\sum_{k=1}^m\Delta_k\Big]=\mathbb{E}\Big[\sum_{k=1}^m\tilde\Delta_k\Big].$$

That's the whole game in one line. The expected regret I care about — which contained the unobservable optimal policy of the unknown MDP — *equals* the expected sum of a surrogate tied to the policy I actually run: the gap between the sampled MDP's optimal value and that policy's value under the real dynamics. The unobservable $\mu^*$ is gone. And notice the *shape* of $\tilde\Delta_k$: it's $V^{M_k}_{\mu_k,1}-V^{M^*}_{\mu_k,1}$, "my optimistic estimate minus reality," which is precisely the form an optimism term takes — the sampled MDP is playing the role the inflated model plays in OFU, except I never built it and never optimized over a family. This is optimism arising implicitly, by sampling. That's why I keep wanting to call it implicit optimism.

Now I have to bound $\mathbb{E}[\sum_k\tilde\Delta_k]$. The two value functions inside $\tilde\Delta_k$ are governed by two different MDPs with the same policy $\mu_k$, so this is a difference of dynamic-programming recursions. Let me telescope it with the Bellman operator. For an MDP $M$, policy $\mu$, value $V$, write $\mathcal{T}^M_\mu V(s)=\overline R^M_\mu(s)+\sum_{s'}P^M_\mu(s'|s)V(s')$ — one backup. Dynamic programming says $V^M_{\mu,i}=\mathcal{T}^M_{\mu(\cdot,i)}V^M_{\mu,i+1}$, with $V_{\tau+1}=0$. Abbreviate $\mathcal{T}^k=\mathcal{T}^{M_k}$, $\mathcal{T}^*=\mathcal{T}^{M^*}$, $V^k=V^{M_k}$, $V^*=V^{M^*}$. Look at the per-state difference at the first step of the episode, starting from the visited state $s_{t_k+1}$:

$$(V^k_{\mu_k,1}-V^*_{\mu_k,1})(s_{t_k+1}) = \big(\mathcal{T}^k_{\mu_k(\cdot,1)}V^k_{\mu_k,2}-\mathcal{T}^*_{\mu_k(\cdot,1)}V^*_{\mu_k,2}\big)(s_{t_k+1}).$$

Add and subtract $\mathcal{T}^*_{\mu_k(\cdot,1)}V^k_{\mu_k,2}$ to split this into "operators differ" plus "values differ":

$$=\underbrace{(\mathcal{T}^k_{\mu_k(\cdot,1)}-\mathcal{T}^*_{\mu_k(\cdot,1)})V^k_{\mu_k,2}(s_{t_k+1})}_{\text{Bellman error}} + \underbrace{\sum_{s'}P^*_{\mu_k(\cdot,1)}(s'|s_{t_k+1})\big(V^k_{\mu_k,2}-V^*_{\mu_k,2}\big)(s')}_{\text{true-dynamics expectation of next difference}}.$$

The second piece is an *expectation* over the true transition of the next-step value-difference $(V^k_{\mu_k,2}-V^*_{\mu_k,2})$. But the agent doesn't average over $s'$ — it actually lands on one realized next state $s_{t_k+2}\sim P^*_{\mu_k(\cdot,1)}(\cdot|s_{t_k+1})$. So I split the realized next difference from its conditional expectation. Define

$$d_{t_k+i}:=\big(V^k_{\mu_k,i+1}-V^*_{\mu_k,i+1}\big)(s_{t_k+i+1})-\sum_{s'}P^*_{\mu_k(\cdot,i)}(s'|s_{t_k+i})\big(V^k_{\mu_k,i+1}-V^*_{\mu_k,i+1}\big)(s'),$$

so the expectation term equals $(V^k_{\mu_k,2}-V^*_{\mu_k,2})(s_{t_k+2}) - d_{t_k+1}$. The signs now do what I need: $(V^k-V^*)_{1}(s_{t_k+1})$ = Bellman error at step $1$ $+\,(V^k-V^*)_{2}(s_{t_k+2})\,-\,d_{t_k+1}$. The $(V^k-V^*)_2(s_{t_k+2})$ term is *exactly the same kind of object*, one step deeper and at the next visited state, so I recurse on it. Unrolling all the way down to $i=\tau$ (where $V_{\tau+1}=0$ closes it):

$$(V^k_{\mu_k,1}-V^*_{\mu_k,1})(s_{t_k+1}) = \sum_{i=1}^{\tau}(\mathcal{T}^k_{\mu_k(\cdot,i)}-\mathcal{T}^*_{\mu_k(\cdot,i)})V^k_{\mu_k,i+1}(s_{t_k+i}) - \sum_{i=1}^\tau d_{t_k+i}.$$

Two factors, and they have completely different characters. The $d_{t_k+i}$ are, by construction, the realized next value-difference minus its conditional expectation under $P^*$ — so $\mathbb{E}[d_{t_k+i}\mid M^*,M_k,\text{history to }t_k{+}i]=0$. They're a martingale-difference sequence capturing the noise of the *true* MDP's transitions; conditioned on $M^*$ and $M_k$ they sum to zero in expectation and contribute nothing to expected regret. (They'd matter for a high-probability bound via Azuma, but I'm after the expectation.) So

$$\mathbb{E}\big[\tilde\Delta_k\,\big|\,M^*,M_k\big]=\mathbb{E}\Big[\sum_{i=1}^\tau(\mathcal{T}^k_{\mu_k(\cdot,i)}-\mathcal{T}^*_{\mu_k(\cdot,i)})V^k_{\mu_k,i+1}(s_{t_k+i})\,\Big|\,M^*,M_k\Big].$$

This is the payoff of the telescoping: the regret reduces to the *one-step Bellman error* between the sampled and true operators, evaluated *only along the trajectory I actually walk* and *only for the policy I actually run*. No optimal policy, no unvisited states — only the visited state-action pairs matter. As the data accumulate, the posterior should concentrate $M_k$ around $M^*$, the operators should agree on visited pairs, and this should go to zero. Now I have to make "should go to zero" quantitative.

What is a one-step Bellman error, concretely? Unpacking the operator difference at a visited pair $(s,a)=(s_{t_k+i},\mu_k(s_{t_k+i},i))$:

$$(\mathcal{T}^k-\mathcal{T}^*)V^k_{\mu_k,i+1}(s) = \underbrace{(r_k-r^*)(s,a)}_{\text{reward gap}} + \underbrace{\sum_{s'}\big(P_k-P^*\big)(s'|s,a)\,V^k_{\mu_k,i+1}(s')}_{\text{transition gap}\,\cdot\,\text{value}}.$$

Bound it by the worst case I can: the reward gap by $|r_k-r^*|$, and the transition term by Hölder, $\big|\sum_{s'}(P_k-P^*)V\big|\le \lVert P_k-P^*\rVert_1\,\lVert V\rVert_\infty$. Values are sums of at most $\tau$ rewards in $[0,1]$, so $\lVert V\rVert_\infty\le\tau$. Hence each one-step error is at most $|r_k-r^*| + \tau\,\lVert P_k-P^*\rVert_1$. I need both the reward and transition discrepancies, *between the sampled and true MDPs*, to be small on visited pairs.

This is the moment to bring back the confidence sets — but, and this is the whole philosophical difference, **only inside the analysis, never inside the algorithm.** I'll build, for episode $k$, a set of MDPs that are statistically consistent with what I've seen:

$$\mathcal{M}_k=\Big\{M:\ \lVert \hat P^t_a(\cdot|s)-P^M_a(\cdot|s)\rVert_1\le\beta_k(s,a)\ \text{and}\ |\hat R^t_a(s)-R^M_a(s)|\le\beta_k(s,a)\ \forall(s,a)\Big\},$$

with empirical estimates $\hat P,\hat R$ and visit counts $N_{t_k}(s,a)$, and radius

$$\beta_k(s,a)=\sqrt{\frac{14\,S\,\log(2SAm\,t_k)}{\max\{1,N_{t_k}(s,a)\}}}.$$

Why this exact radius? It's the $L^1$-deviation concentration for an $S$-outcome distribution. Controlling an $L^1$ ball over $S$ successor states is harder than controlling a single scalar — the empirical transition row deviates in $L^1$ by an amount whose concentration involves the possible sign patterns of the deviation and leaves a leading $\sqrt S$ cost. It's the same conservative radius UCRL2 uses after unioning over state-action pairs and times; I take the union-bound parameter to be $\delta=1/m$ over the $m$ episodes, which is the only thing that fixes the constant. I'm not trying to optimize it — and crucially I don't *have* to, because no agent decision depends on $\beta_k$. It exists purely so I can say: this radius is conservative enough that $\mathbb{P}(M^*\notin\mathcal{M}_k)\le 1/m$ (this is exactly UCRL2's Lemma 17 with their $\delta=1/m$).

And now the second, quieter place the posterior-sampling identity earns its keep. $\mathcal{M}_k$ is built entirely from the history up to $t_k$ — empirical counts and averages — so it is $H_{t_k}$-measurable. The indicator $\mathbf 1\{M\notin\mathcal{M}_k\}$ is an $H_{t_k}$-measurable function of the MDP $M$. So by the same identity,

$$\mathbb{E}\big[\mathbf 1\{M_k\notin\mathcal{M}_k\}\mid H_{t_k}\big]=\mathbb{E}\big[\mathbf 1\{M^*\notin\mathcal{M}_k\}\mid H_{t_k}\big].$$

**The sampled MDP falls inside the confidence set exactly as often as the true MDP does.** I designed the set to contain $M^*$ with probability $\ge 1-1/m$; for free it contains $M_k$ with the same probability. This is what an optimist has to engineer and I get by sampling: I never had to *construct* a valid confidence set for the agent, yet the one I build for the proof simultaneously controls both the truth and my sample. Decompose, using $\tilde\Delta_k\le\tau$ on the bad events:

$$\sum_{k=1}^m\tilde\Delta_k\le\sum_{k=1}^m\tilde\Delta_k\,\mathbf 1\{M_k,M^*\in\mathcal{M}_k\}+\tau\sum_{k=1}^m\big[\mathbf 1\{M_k\notin\mathcal{M}_k\}+\mathbf 1\{M^*\notin\mathcal{M}_k\}\big].$$

Take expectations. The two failure indicators each have expectation $\le 1/m$ per episode by the equidistribution and the choice of $\beta_k$, so the second sum contributes $\le \tau\cdot 2\cdot m\cdot(1/m)=2\tau$. On the good event, plug in the Bellman-error reduction, then the one-step bound $|r_k-r^*|+\tau\lVert P_k-P^*\rVert_1$. On $\{M_k,M^*\in\mathcal{M}_k\}$ both MDPs are within $\beta_k$ of the *same* empirical model, so by the triangle inequality the reward gap is $\le 2\beta_k$ and the transition $L^1$ gap is $\le 2\beta_k$; since values are bounded by $\tau$, each surviving Bellman-error term is bounded by a universal constant times $\tau\beta_k$. If $\beta_k$ is larger than one I cap a one-step Bellman error by a universal constant times $\tau$. Hiding only numerical constants,

$$\mathbb{E}\Big[\sum_{k=1}^m\tilde\Delta_k\Big]\le C\tau\,\mathbb{E}\sum_{k=1}^m\sum_{i=1}^\tau\min\{\beta_k(s_{t_k+i},a_{t_k+i}),1\}+2\tau.$$

Everything now hinges on one combinatorial sum: $\sum_{k}\sum_i \min\{\beta_k,1\}$ along the realized trajectory. This is the classic "sum of shrinking confidence widths" and it's where the $\sqrt{SAT}$ comes from. Let me also keep the trivial cap $\sum_k\tilde\Delta_k\le T$ on hand, so I'm really bounding $\min\{C\tau\sum_k\sum_i\min\{\beta_k,1\},\ T\}$.

Substitute $\beta_k=\sqrt{14 S\log(2SAm t_k)/\max\{1,N_{t_k}\}}$. The trouble is that $N_{t_k}$ — the count at the *start* of the episode — is frozen for the whole episode, while I keep visiting pairs *within* the episode, so within an episode the "real" count grows but $\beta_k$ doesn't shrink. I handle the early visits and the later visits separately. Split each term by whether $N_{t_k}(s,a)\le\tau$ or $>\tau$.

For the small-count part: consider a fixed $(s,a)$. The event "$(s_t,a_t)=(s,a)$ while $N_{t_k}(s,a)\le\tau$" can occur fewer than $2\tau$ times — because once you've accumulated more than $\tau$ start-of-episode visits you leave the regime, and within an episode you add at most $\tau$ more. So summed over all $(s,a)$, the number of trajectory steps with $N_{t_k}\le\tau$ is at most $2\tau SA$, each contributing $\min\{\beta_k,1\}\le 1$. That part of $\sum_k\sum_i\min\{\beta_k,1\}$ is $\le 2\tau SA$.

For the large-count part, $N_{t_k}>\tau$: within episode $k$, for any $t\in\{t_k,\dots,t_{k+1}-1\}$, the current count satisfies $N_t(s,a)+1\le N_{t_k}(s,a)+\tau\le 2N_{t_k}(s,a)$ (the $+\tau$ because an episode is $\tau$ long, the last inequality because $N_{t_k}>\tau$). So $1/\sqrt{N_{t_k}}\le\sqrt{2}/\sqrt{N_t+1}$, which lets me swap the frozen start-count for the live count at a cost of $\sqrt 2$. Then

$$\sum_{k}\sum_{t=t_k}^{t_{k+1}-1}\sqrt{\frac{\mathbf 1\{N_{t_k}>\tau\}}{N_{t_k}(s_t,a_t)}}\le\sqrt2\sum_{t=1}^{T}\big(N_t(s_t,a_t)+1\big)^{-1/2}.$$

Now reorganize the trajectory sum by counting per pair: each time I visit $(s,a)$ its live count steps through $1,2,\dots,N_{T+1}(s,a)$, so

$$\sum_{t=1}^T (N_t(s_t,a_t)+1)^{-1/2}\le\sum_{s,a}\sum_{j=1}^{N_{T+1}(s,a)}j^{-1/2}\le\sum_{s,a}\int_0^{N_{T+1}(s,a)}x^{-1/2}\,dx=\sum_{s,a}2\sqrt{N_{T+1}(s,a)}.$$

And Cauchy-Schwarz over the $SA$ pairs, with $\sum_{s,a}N_{T+1}(s,a)=T$ total steps:

$$\sum_{s,a}\sqrt{N_{T+1}(s,a)}\le\sqrt{SA\sum_{s,a}N_{T+1}(s,a)}=\sqrt{SAT}.$$

So the live-count sum $\sum_t(N_t+1)^{-1/2}\le 2\sqrt{SAT}$, and the $\sqrt2$ swap puts the large-count part of $\sum_k\sum_i 1/\sqrt{N_{t_k}}$ at $\le 2\sqrt2\,\sqrt{SAT}$. Now multiply back the constant inside $\beta_k$, which is $\sqrt{14 S\log(2SAm t_k)}\le C\sqrt{S\log(SAT)}$ after folding the extra powers of $T$ into the logarithm. The large-count contribution to $\sum_k\sum_i\beta_k$ is therefore at most $C S\sqrt{AT\log(SAT)}$.

Put the two parts together and apply the outer $\tau$ and the $\min\{\cdot,T\}$:

$$\min\Big\{C\tau\sum_k\sum_i\min\{\beta_k,1\},\ T\Big\}\le\min\Big\{C\tau^2 SA+C\tau S\sqrt{A T\log(SAT)},\ T\Big\}.$$

The $\min$ with $T$ tames the burn-in term: $\min\{C\tau^2 SA,T\}\le C\tau\sqrt{SAT}$ (since $\min\{x,T\}\le\sqrt{xT}$), and this is dominated by $C\tau S\sqrt{AT\log(SAT)}$ for the nontrivial range of parameters. Adding back the harmless $+2\tau$,

$$\boxed{\ \mathbb{E}\big[\mathrm{Regret}(T,\pi^{\rm PS}_\tau)\big]=O\big(\tau S\sqrt{AT\log(SAT)}\big).\ }$$

So the bare one-sample-per-episode scheme — no optimism, no confidence sets in the loop — has $\tilde O(\tau S\sqrt{AT})$ expected regret. Let me sanity-check the exponents against the optimistic baseline. UCRL2 gets $\tilde O(D S\sqrt{AT})$; mine has $\tau$ where it has the diameter $D$ — and in an episodic problem $\tau$ is exactly the relevant horizon, so this is the same order with the episode length playing $D$'s role, matched to the problem. The lower bound is $\sqrt{SAT}$; I'm a factor $\sqrt S$ above it, and I can see precisely where that $\sqrt S$ lives — it's inside $\beta_k$, the price of controlling an $L^1$ ball over $S$ successor states rather than a scalar. It's the same gap UCRL2 has. Not optimal, but state-of-the-art for a tractable algorithm, and obtained without ever solving an optimization over a family of MDPs.

One thing about the *kind* of regret I bounded. $\mathbb{E}[\mathrm{Regret}]$ here is an expectation under the prior $f$ — Bayes risk, or Bayesian regret. That's the natural object: the identity $\mathbb{E}[g(M^*)|H]=\mathbb{E}[g(M_k)|H]$ is a statement about *distributions*, so the bound that follows is inherently about expected, prior-averaged regret. And the prize is that it holds for *any* prior over MDPs — an immense model class — with no structural assumption. The literature usually quotes worst-case (frequentist) regret instead, so let me make sure I haven't bought generality at the cost of a vacuous guarantee for a specific true MDP. Markov's inequality on the bound gives convergence in probability: for any $\alpha>\tfrac12$, $\mathrm{Regret}(T)/T^\alpha\to_p 0$. Stronger, condition on the true MDP lying in *any* family $\mathcal{M}$ that has nonzero prior mass. Then for any $\epsilon>0$,

$$\frac{\mathbb{E}[\mathrm{Regret}(T)]}{T^\alpha}\ge\mathbb{E}\Big[\frac{\mathrm{Regret}(T)}{T^\alpha}\,\Big|\,M^*\in\mathcal{M}\Big]\,\mathbb{P}(M^*\in\mathcal{M})\ge\epsilon\,\mathbb{P}\Big(\frac{\mathrm{Regret}(T)}{T^\alpha}>\epsilon\,\Big|\,M^*\in\mathcal{M}\Big)\,\mathbb{P}(M^*\in\mathcal{M}),$$

so $\mathbb{P}\big(\mathrm{Regret}(T)/T^\alpha>\epsilon\mid M^*\in\mathcal{M}\big)\le\frac{1}{\epsilon\,\mathbb{P}(M^*\in\mathcal{M})}\cdot\frac{\mathbb{E}[\mathrm{Regret}(T)]}{T^\alpha}\to 0$ by the theorem. So as long as the true MDP isn't *impossible* under my prior, I get a frequentist statement with regret $o_p(T^\alpha)$ for every $\alpha>\tfrac12$, arbitrarily close to the $\sqrt T$ exponent. The Bayesian and frequentist notions are linked, and I haven't lost the prior-positive-instance story.

Now let me make the algorithm concrete so it actually runs. I need a posterior I can sample from cheaply. For the transitions out of each $(s,a)$ — a distribution over $S$ next states — the conjugate prior is the **Dirichlet**: a positive pseudo-count vector $\alpha\in\mathbb{R}^S_+$, and the posterior update on observing $s'$ is just "increment the $s'$ component of $\alpha$." For the rewards, model each $(s,a)$ reward as normal and put a **normal-gamma** prior on its mean and precision — conjugate to the normal, so the posterior is again closed-form. Sampling an MDP $M_k$ is then: for every $(s,a)$, draw a transition row from its Dirichlet posterior and a mean reward from its normal-gamma posterior. Solving $\mu_k=\mu^{M_k}$ is ordinary finite-horizon planning — backward induction, $V_i(s)=\max_a[r(s,a)+\sum_{s'}P(s'|s,a)V_{i+1}(s')]$ — on a single known MDP, which is cheap. Execute $\mu_k$ for $\tau$ steps, fold the observed transitions and rewards back into the Dirichlet and normal-gamma posteriors, and repeat. One sample, one MDP solve, one fixed policy per episode — and the exploration is carried entirely by how wide the posterior still is.

```python
import numpy as np

class DirichletNormalGammaBelief:
    """Conjugate posterior over an unknown finite-horizon MDP.
    Dirichlet over each transition row (conjugate to multinomial transitions);
    normal-gamma over each reward (conjugate to normal rewards)."""
    def __init__(self, S, A, horizon, alpha0=None, mu0=1.0, lam0=1.0, a0=1.0, b0=1.0):
        self.S, self.A, self.H = S, A, horizon
        # Dirichlet pseudo-counts for transitions; diffuse default alpha0 = 1/S.
        a0d = (1.0 / S) if alpha0 is None else alpha0
        self.dir = np.full((S, A, S), a0d)
        # Normal-gamma parameters per (s,a): mean mu, pseudo-obs lam, shape a, rate b.
        self.mu  = np.full((S, A), mu0)
        self.lam = np.full((S, A), lam0)
        self.a   = np.full((S, A), a0)
        self.b   = np.full((S, A), b0)

    def update(self, s, a, r, s_next):
        self.dir[s, a, s_next] += 1.0                          # increment Dirichlet count
        lam, mu = self.lam[s, a], self.mu[s, a]                # normal-gamma update
        self.mu[s, a]  = (lam * mu + r) / (lam + 1.0)
        self.b[s, a]  += 0.5 * lam / (lam + 1.0) * (r - mu) ** 2
        self.a[s, a]  += 0.5
        self.lam[s, a] = lam + 1.0

    def sample_mdp(self):
        """Draw one MDP from the posterior: P from Dirichlet, R from normal-gamma."""
        P = np.empty((self.S, self.A, self.S))
        R = np.empty((self.S, self.A))
        for s in range(self.S):
            for a in range(self.A):
                P[s, a] = np.random.dirichlet(self.dir[s, a])
                prec = np.random.gamma(self.a[s, a], 1.0 / self.b[s, a])
                R[s, a] = np.random.normal(self.mu[s, a],
                                           1.0 / np.sqrt(self.lam[s, a] * prec))
        return R, P

def finite_horizon_optimal_policy(R, P, S, A, horizon):
    """Backward induction on a single known MDP -> optimal policy per step."""
    V = np.zeros((horizon + 1, S))
    mu = np.zeros((horizon, S), dtype=int)
    for i in reversed(range(horizon)):
        Q = R + P @ V[i + 1]            # Q[s,a] = R[s,a] + sum_s' P[s,a,s'] V[i+1,s']
        mu[i] = Q.argmax(axis=1)
        V[i]  = Q.max(axis=1)
    return mu

def psrl(env, S, A, horizon, n_episodes):
    belief = DirichletNormalGammaBelief(S, A, horizon)
    for k in range(n_episodes):
        R, P = belief.sample_mdp()                              # one posterior sample
        mu   = finite_horizon_optimal_policy(R, P, S, A, horizon)  # plan for the sample
        s = env.reset()
        for i in range(horizon):                                # hold policy fixed: deep exploration
            a = mu[i][s]
            r, s_next = env.step(s, a)
            belief.update(s, a, r, s_next)                      # posterior concentrates
            s = s_next
    return belief
```

To recap the causal chain. Certainty-equivalence under-explores because a point estimate hides the agent's ignorance; optimism fixes that but at the cost of constructing confidence sets and optimizing over a family of MDPs. The alternative is to let the *posterior's own variance* drive exploration — sample one MDP per episode and act optimally for it, à la Thompson — which makes me select each policy with the probability it is optimal, and I sample once per episode so the exploration stays coherent over the horizon. The single fact that makes this provable is that, given the history, the sampled MDP and the true MDP are identically distributed; so any history-selected functional of the model has equal expectation under both. That lets me replace the unobservable optimal-policy regret $\Delta_k$ by a same-expectation surrogate $\tilde\Delta_k$ — the gap between the sampled MDP's claimed value and its value under the real dynamics — which is an optimism term arising implicitly from sampling. Telescoping that surrogate through the Bellman equation reduces it to one-step Bellman errors along the visited trajectory plus a mean-zero martingale; a confidence set — needed only for the proof, and which contains the sample exactly as often as the truth, again by the same identity — bounds those errors by shrinking widths $\beta_k$; and summing the widths via the $\sum N^{-1/2}\le 2\sqrt{SAT}$ counting argument yields $\tilde O(\tau S\sqrt{AT})$ Bayesian regret, with $o_p(T^\alpha)$ regret for every $\alpha>\tfrac12$ on any prior-positive true-MDP family.
