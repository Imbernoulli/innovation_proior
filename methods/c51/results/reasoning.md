OK, let me think this through from scratch. The thing I keep coming back to is how much we throw away in value-based RL. We train an agent to maximize expected return, and so we learn $Q(x,a)=\mathbb{E}[Z(x,a)]$, a single number per state-action. Bellman gives us the recursion $Q(x,a)=\mathbb{E}\,R(x,a)+\gamma\,\mathbb{E}\,Q(X',A')$, two operators $\mathcal{T}^\pi$ and $\mathcal{T}$, each a $\gamma$-contraction in $L_\infty$, and Banach hands us a unique fixed point. Clean. But $Z(x,a)$ — the actual return I get if I start in $(x,a)$ — is a random variable. In a game it might be "I survive and score 200" with some probability and "I die now and score 0" with the rest. That's a bimodal distribution. Q collapses it to, say, 140. And 140 is a return I will *never actually receive*. It's an average over two worlds, neither of which looks like 140.

So the question I want to sit with: why am I only learning the mean? Why not the whole distribution of $Z$? In supervised learning we'd never hesitate — if I can model a full conditional distribution I do, it strictly carries more information. The catch in RL is that there are no given targets; I bootstrap, I "learn a guess from a guess." So the real question is whether the *machinery* survives. Three things make mean value learning work: (1) a Bellman recursion the object obeys, (2) a contraction so iterating that recursion converges to a unique answer, (3) a loss I can train from sampled transitions by SGD. If I want to learn the distribution of $Z$, I need all three to survive in distribution-space. Let me check them one at a time, and let the representation and the loss fall out of what the math will and won't allow.

Start with the recursion. The return obeys $Z(x,a)=\sum_{t\ge0}\gamma^t R(x_t,a_t)$ with $x_0=x,a_0=a$. Peel off the first term: $Z(x,a)=R(x,a)+\gamma\sum_{t\ge1}\gamma^{t-1}R(x_t,a_t)$. That tail sum, given $X'=x_1, A'=a_1$, is distributed exactly like $Z(X',A')$ — same MDP, time-homogeneous. So, as random variables,
$$Z(x,a)\;\overset{D}{=}\;R(x,a)+\gamma\,Z(X',A'),\qquad X'\sim P(\cdot\,|\,x,a),\ A'\sim\pi(\cdot\,|\,X').$$
That's an equality *in distribution*, not of numbers. I have to be careful: this is not one equation, it's a statement that the law of the left side equals the law of a compound random variable built from three independent pieces — the reward $R$, the random next state-action $(X',A')$, and the next return $Z(X',A')$. Good. So a distributional Bellman recursion exists. Let me name the operators. Define the transition operator pushing distributions forward,
$$P^\pi Z(x,a)\overset{D}{:=}Z(X',A'),\qquad X'\sim P(\cdot|x,a),\,A'\sim\pi(\cdot|X'),$$
and the distributional Bellman operator
$$\mathcal{T}^\pi Z(x,a)\overset{D}{:=}R(x,a)+\gamma\,P^\pi Z(x,a).$$
It looks like the scalar operator with the expectations stripped out, but it is doing something much richer: it scales the next-state distribution by $\gamma$ (shrinking it toward 0), then convolves with the transition randomness and with the reward distribution. It's a genuine distribution-to-distribution map.

The recursion exists, but the convergence story lives or dies on the metric, and I don't yet know which one works. My instinct as a deep-learning person is KL divergence — that's what I'd minimize with a softmax. So let me ask: is $\mathcal{T}^\pi$ a contraction in KL? Think about the $\gamma$-scaling alone. Take a distribution and another one, both with support on, say, $\{1,2\}$. Apply $P^\pi$ then scale by $\gamma=0.5$: now they live on $\{0.5,1\}$. KL between the originals and KL between the scaled versions — scaling the *locations* doesn't change the *probabilities*, so KL is exactly unchanged. No contraction there at all. Worse: take two distributions on disjoint supports, scale both by $\gamma$. As $\gamma\to0$ both collapse toward the point mass at 0, they get visually closer and closer — but if their supports stay disjoint, KL stays infinite (or total variation stays at 1). KL and TV are *vertical* distances: they compare probability mass at matched locations, they're blind to how far apart the locations are. Shrinking the return toward 0 is precisely a *horizontal* operation. So KL, total variation, Kolmogorov sup-CDF distance — none of them can see the contraction that the $\gamma$-scaling is producing; my disjoint-support example above is in fact a counterexample to contraction in all three at once (scale both by $\gamma$, the supports stay disjoint, KL stays $\infty$ and TV stays $1$ while the distributions visibly converge to $\delta_0$). So $\mathcal{T}^\pi$ is *not* a contraction in total variation, nor in KL, nor in Kolmogorov distance. Dead end if I insist on those.

What I want is a distance that *does* see horizontal movement and shrinks when I scale toward a point. That's the Wasserstein / Mallows metric. For two CDFs $F,G$,
$$d_p(F,G)=\inf_{U,V}\|U-V\|_p,$$
the infimum over couplings $(U,V)$ with marginals $F,G$, attained by the quantile coupling $U=F^{-1}(\mathcal U),V=G^{-1}(\mathcal U)$ with $\mathcal U$ uniform on $[0,1]$, so
$$d_p(F,G)=\Big(\int_0^1|F^{-1}(u)-G^{-1}(u)|^p\,du\Big)^{1/p}.$$
This is a *transport* distance: how far do I have to slide mass. It is finite for disjoint supports, and crucially it scales: if I multiply both random variables by a scalar, I multiply the optimal transport by that scalar. Let me write down the properties I'll lean on. For scalar $a$ and a random variable $A$ independent of $U,V$:
$$d_p(aU,aV)\le|a|\,d_p(U,V)\ \text{(P1)},\quad d_p(A+U,A+V)\le d_p(U,V)\ \text{(P2)},\quad d_p(AU,AV)\le\|A\|_p\,d_p(U,V)\ \text{(P3)}.$$
P1 is the exact horizontal-scaling fact KL lacks. P2 says a common additive shift can't increase transport distance (in fact for a deterministic shift it's equality, but $\le$ is all I need). These two are going to do the work.

There's a subtlety: $Z$ isn't a single distribution, it's a distribution *per* state-action. So I need a metric over the whole value-distribution function. Take the sup:
$$\bar d_p(Z_1,Z_2):=\sup_{x,a}d_p\big(Z_1(x,a),Z_2(x,a)\big).$$
This is the maximal Wasserstein metric. (It's a metric — the only nontrivial axiom is the triangle inequality, and that's inherited pointwise: $d_p(Z_1(x,a),Z_2(x,a))\le d_p(Z_1(x,a),Y(x,a))+d_p(Y(x,a),Z_2(x,a))\le \bar d_p(Z_1,Y)+\bar d_p(Y,Z_2)$ for every $(x,a)$, then take the sup on the left.)

Now contract. Take $Z_1,Z_2$ and look at one state-action:
$$d_p(\mathcal{T}^\pi Z_1(x,a),\mathcal{T}^\pi Z_2(x,a))=d_p\big(R(x,a)+\gamma P^\pi Z_1(x,a),\,R(x,a)+\gamma P^\pi Z_2(x,a)\big).$$
The reward $R(x,a)$ is the *same* random variable added to both sides. By P2 the common additive term drops:
$$\le d_p\big(\gamma P^\pi Z_1(x,a),\,\gamma P^\pi Z_2(x,a)\big).$$
By P1 the scalar $\gamma$ comes out:
$$\le \gamma\, d_p\big(P^\pi Z_1(x,a),\,P^\pi Z_2(x,a)\big).$$
And $P^\pi Z(x,a)$ is by definition the law of $Z(X',A')$, a mixture over next state-actions of the distributions $Z(\cdot,\cdot)$, so its $d_p$ to the other one can't exceed the worst-case over $(x',a')$:
$$\le \gamma\,\sup_{x',a'} d_p\big(Z_1(x',a'),Z_2(x',a')\big).$$
Take the sup over $(x,a)$ on the left:
$$\bar d_p(\mathcal{T}^\pi Z_1,\mathcal{T}^\pi Z_2)\le \gamma\,\bar d_p(Z_1,Z_2).$$
A $\gamma$-contraction. So Banach applies in $\bar d_p$ (over distributions with bounded moments), there is a unique fixed point. I should check that the fixed point is actually the true return distribution $Z^\pi$ and not some other thing the operator happens to fix. The defining recursion I peeled off at the start, $Z^\pi(x,a)\overset{D}{=}R(x,a)+\gamma Z^\pi(X',A')$, is precisely the statement $\mathcal{T}^\pi Z^\pi=Z^\pi$ — so $Z^\pi$ *is* a fixed point; uniqueness then forces it to be *the* one Banach converges to. Because $\bar d_p$ controls all the $L_p$ transport — $d_p$ bounds the difference of the $p$-th quantile functions in $L_p$ — every moment that the $L_p$ quantile distance controls converges geometrically too. So I have parts (1) and (2) in the policy-evaluation case, and the load-bearing fact is the metric: it contracts in Wasserstein, and (from the $\gamma$-scaling argument above) not in KL/TV/Kolmogorov.

Let me poke at the moments, because I want to know what "converges geometrically" buys me concretely. Relate $d_2$ to a coupling $C=U-V$: $d_2^2(U,V)\le\mathbb{E}[(U-V)^2]=\mathrm{Var}(C)+(\mathbb{E}\,C)^2$. So $d_2$ does *not* cleanly upper-bound the difference of variances — the $(\mathbb{E}\,C)^2$ term gets in the way. A tempting shortcut would be to say variance simply contracts by $\gamma^2$, because
$$\mathrm{Var}\big(R(x,a)+\gamma P^\pi Z(x,a)\big)=\mathrm{Var}\big(R(x,a)\big)+\gamma^2\,\mathrm{Var}\big(P^\pi Z(x,a)\big)$$
when the reward is independent of the next return. But $P^\pi Z(x,a)$ is a mixture over next state-actions, and the law of total variance says
$$\mathrm{Var}\big(P^\pi Z(x,a)\big)=\mathbb E_{P,\pi}\mathrm{Var}\big(Z(X',A')\big)+\mathrm{Var}_{P,\pi}\big(\mathbb E Z(X',A')\big).$$
So the conditional-variance part really does carry a $\gamma^2$ factor, but the total variance also contains the variance of the next-state means. I should not claim a standalone $\gamma^2$ contraction for arbitrary approximants unless those means are held fixed. The safer complete statement is the one the Wasserstein contraction already gives me: with bounded moments, the distribution iterates converge geometrically to $Z^\pi$, and the centered moments converge with them; for the second moment the shrink of the residual spread is naturally second-order once the means have settled. Fine. Policy evaluation is in good shape, but the exact object of contraction is the distribution, not an isolated variance vector.

Now control — and here I should be suspicious, because in the scalar world the optimality operator $\mathcal T$ is the one that gives me Q-learning, and I want its distributional analogue to converge to "the optimal value distribution." But wait — what *is* the optimal value distribution? In the scalar world all optimal policies share the same $Q^*$. In distribution-space they need not: two optimal policies achieve the same *mean* return but can have genuinely different return *distributions*. So "the" optimal value distribution isn't well defined; there's a *set* $\mathcal Z^*=\{Z^{\pi^*}:\pi^*\in\Pi^*\}$, and not every distribution with mean $Q^*$ is in it — it has to be the actual return distribution of some optimal policy. That's already a warning sign.

Define the operator. A greedy policy for $Z$ maximizes the *mean*: $\mathcal G_Z=\{\pi:\sum_a\pi(a|x)\,\mathbb E\,Z(x,a)=\max_{a'}\mathbb E\,Z(x,a')\}$. A distributional optimality operator is any $\mathcal T Z=\mathcal T^\pi Z$ with $\pi\in\mathcal G_Z$. The greedy choice depends only on the mean, but it selects a whole *distribution* to bootstrap from — and that's where trouble can hide, because the mean can be insensitive to a change that flips the greedy action and thereby swaps in a completely different distribution.

The mean itself is fine, let me confirm. $\mathbb E\,\mathcal T_D Z=\mathcal T_E\,\mathbb E\,Z$ where $\mathcal T_E$ is the ordinary scalar optimality operator (taking expectation commutes through, the greedy choice is the same mean-greedy choice). The scalar operator is a $\gamma$-contraction in $L_\infty$, so
$$\|\mathbb E\,\mathcal T Z_1-\mathbb E\,\mathcal T Z_2\|_\infty\le\gamma\,\|\mathbb E\,Z_1-\mathbb E\,Z_2\|_\infty,$$
and $\mathbb E\,Z_k\to Q^*$ geometrically. The *mean* of my distributional iterates converges, exactly as in DQN. So why worry?

Because the *distribution* need not. Let me try to break it with a small example, since if it can break I want to see how. Two states. From $x_1$ there's a single forced transition to $x_2$. At $x_2$, action $a_1$ gives reward $0$, action $a_2$ gives $1+\epsilon$ or $-1+\epsilon$ with equal probability; both are terminal. The optimal action at $x_2$ is $a_2$ (mean $\epsilon>0$ vs $0$), so there's a unique optimal policy and a unique fixed point $Z^*$. Now take a $Z$ that equals $Z^*$ everywhere except at $(x_2,a_2)$, where I shift it to $-\epsilon\pm1$ (mean $-\epsilon$). The Wasserstein distance: only $(x_2,a_2)$ differs, and shifting $\epsilon\pm1$ to $-\epsilon\pm1$ moves all the mass by $2\epsilon$, so $\bar d_1(Z,Z^*)=2\epsilon$. Tiny.

Apply $\mathcal T$. At $x_2$ the means are now: $a_1$ gives $0$, $a_2$ gives $-\epsilon<0$. So greedy *flips* to $a_1$, and $\mathcal T Z(x_1)=Z(x_2,a_1)=\delta_0$, the point mass at $0$. But $Z^*(x_1)$ is the true optimal distribution, $\epsilon\pm1$. So
$$\bar d_1(\mathcal T Z,\mathcal T Z^*)=d_1(\mathcal T Z(x_1),Z^*(x_1))=d_1(\delta_0,\ \epsilon\pm1)=\tfrac12|1-\epsilon|+\tfrac12|1+\epsilon|\approx1$$
for small $\epsilon$, which is $\gg 2\epsilon$. The distance *grew* from $2\epsilon$ to about $1$. So $\mathcal T$ is *not* a non-expansion, let alone a contraction — and the same arithmetic with $\gamma<1$ still expands. A tiny perturbation to the distribution flipped the mean-greedy action and swapped in a totally different bootstrap distribution; the mean barely moved but the distribution lurched. So $\mathcal T$ is not a contraction in *any* metric over distributions. It gets worse: with a tie-breaking rule, $\mathcal T$ may have *no* fixed point at all (let $\epsilon=0$ so $a_1,a_2$ tie; a rule that picks $a_2$ when $Z(x_1)=0$ and $a_1$ otherwise makes the iterates oscillate forever between $Z(x_2,a_1)$ and $Z(x_2,a_2)$).

And even when a fixed point exists, the iterates can fail to reach $\mathcal Z^*$. Here's the cleanest way to see why — it's a nonstationarity effect. One state $x_1$, two optimal actions: $a_1$ deterministically gives reward $\tfrac12$, $a_2$ gives $0$ or $1$ with equal probability; $\gamma=\tfrac12$, and both have mean $\tfrac12$ so both are optimal. If I always take $a_1$ ($p=0$), the return is $\frac{1}{1-\gamma}\cdot\frac12=1$. If I always take $a_2$ ($p=1$), the return is $R_0.R_1R_2R_3\cdots$ read as a binary fraction with each bit i.i.d. fair — which is *uniform on $[0,2]$*. For intermediate $p$ I get intermediate distributions, all supported on $[0,2]$. But now consider the *nonstationary* policy "$a_1$ first, then $a_2$ forever": the return is uniform on $[\tfrac12,\tfrac32]$ — a distribution achievable by *no* stationary $p$. The optimality operator, which re-picks the greedy action each step based on the current distribution, can chase exactly these nonstationary mixtures. So the iterates converge — but to the larger set of *nonstationary* optimal value distributions $\mathcal Z^{**}$, not to $\mathcal Z^*$.

Let me actually nail the positive convergence result, because "it converges to $\mathcal Z^{**}$" needs a proof and the proof tells me *why* averaging will help in practice. I need two tools. A partition lemma says that if $\{A_i\}$ partition $\Omega$ (each $A_i\in\{0,1\}$, exactly one fires), then for any $U,V$,
$$d_p(U,V)\le\sum_i d_p(A_iU,A_iV).$$
Why: $d_p^p(A_iU,A_iV)=\inf\mathbb E|Y_i-Z_i|^p$, and since $A_iU-A_iV=0$ whenever $A_i=0$ I can take couplings with $|Y_i-Z_i|=0$ off the event, so $d_p^p(A_iU,A_iV)=\inf\Pr\{A_i=1\}\mathbb E[|Y_i-Z_i|^p\,|\,A_i=1]$. The CDF of $U$ factors as $\sum_i\Pr\{A_i=1\}\Pr\{U\le y|A_i=1\}$ — i.e. "first pick a partition cell, then draw within it." The full coupling of $(U,V)$ is free to reorder mass across all of $\mathbb R$; the per-cell couplings can only reorder within each cell, which is an extra constraint, so the per-cell infimum can only be larger:
$$d_p^p(U,V)\le\sum_i d_p^p(A_iU,A_iV).$$
Taking the $p$-th root and then using $(\sum_i b_i^p)^{1/p}\le\sum_i b_i$ gives the stated form. I'll also use P1, P2, P3.

Now the control-convergence argument. Take the unique-optimal-policy case first; write $\pi^*(x)$ for the optimal action, $Q_k=\mathbb E\,Z_k$. Let $B=2\sup_Z\|Z\|_\infty<\infty$ and $\epsilon_k=\gamma^kB$. From the mean contraction, $|Q_k(x,a)-Q^*(x,a)|\le\epsilon_k$. Define the "solved-by-time-$k$" states as those with a mean-gap bigger than $2\epsilon_k$:
$$\mathcal X_k:=\big\{x:Q^*(x,\pi^*(x))-\max_{a\ne\pi^*(x)}Q^*(x,a)>2\epsilon_k\big\}.$$
On $\mathcal X_k$, $Q_k(x,\pi^*(x))-Q_k(x,a)\ge Q^*(x,\pi^*(x))-Q^*(x,a)-2\epsilon_k>0$, so the greedy policy *equals* $\pi^*$ there. Because $\mathcal A$ is finite each gap $\Delta(x)>0$, and $\epsilon_k=\gamma^kB\to0$, so eventually $\gamma^kB<\Delta(x)/2$ and every $x$ enters $\mathcal X_k$ and stays: $\mathcal X_k\uparrow\mathcal X$. But I need more than the state itself being solved — I need its successors solved too, and theirs, recursively. Fix $\delta>0$, set $\mathcal X_{k,0}=\mathcal X_k$, and
$$\mathcal X_{k,i}:=\{x\in\mathcal X_k:\ P(\mathcal X_{k-1,i-1}\,|\,x,\pi^*(x))\ge1-\delta\}.$$
Since $\mathcal X_{k,i}\uparrow\mathcal X$ for $i=0$, $P(\mathcal X_{k,i}|x,\pi^*(x))\to1$, so $\mathcal X_{k,i+1}\uparrow\mathcal X$ too; by induction every $x$ eventually sits in $\mathcal X_{k,i}$ for any depth $i$.

Now bound the per-state Wasserstein error. Write $W_k(x):=Z_k(x,\pi_k(x))$, and define $W^*$ with the optimal action. Pick the time index so the current state is already in the solved set at the moment the greedy action is chosen, and so its next state lands in the previous solved-depth set with high probability. Then $\pi_k(x)=\pi^*(x)$, and by P1/P2 (the reward and $\gamma$-scaling),
$$d_p(W_{k+1}(x),W^*(x))=d_p(\mathcal T W_k(x),\mathcal T W^*(x))\le\gamma\,d_p\big(P^{\pi^*}W_k(x),P^{\pi^*}W^*(x)\big).$$
Split the random successor $X'$ into "solved" and "unsolved" with the indicators $S_i^k(X')=\mathbb 1[X'\in\mathcal X_{k,i}]$, $\bar S_i^k=1-S_i^k$, a partition of $\Omega$. By the partition lemma,
$$\le\gamma\,d_p(S_i^kW_k(X'),S_i^kW^*(X'))+\gamma\,d_p(\bar S_i^kW_k(X'),\bar S_i^kW^*(X')).$$
The unsolved term: by P3, $d_p(\bar S_i^kW_k,\bar S_i^kW^*)\le\|\bar S_i^k(X')\|_p\sup_{x'}d_p(W_k(x'),W^*(x'))$. If I want this term to be at most $\delta B$, I choose the recursive solved sets with failure probability at most $\delta^p$, so $\|\bar S_i^k(X')\|_p=\Pr\{X'\notin\mathcal X_{k,i}\}^{1/p}\le\delta$, and the worst-case Wasserstein error is bounded by $B$. So that term is $\le\gamma\delta B$. The solved term still has the same form one level down, so inducting on $i$,
$$d_p(W_{k+i}(x),W^*(x))\le\gamma^i\,d_p(S_0^kW_k(X''),S_0^kW^*(X''))+\frac{\delta B}{1-\gamma}\le\gamma^iB+\frac{\delta B}{1-\gamma},$$
summing the geometric series $\sum_{j<i}\gamma^j\,\gamma\delta B\le\delta B/(1-\gamma)$. For any target $\epsilon>0$ pick $\delta$ small, then $i$ large, then $k$ large, and the right side drops below $\epsilon$. So $d_p(Z_k(x,a),\mathcal Z^{**})\to0$ pointwise (uniformly if $\mathcal X$ is finite). With multiple optimal policies the same argument runs with $\mathcal X_{k,i}$ requiring the successor condition for *every* optimal policy, and the bootstrapped distribution $\mathcal T^{\bar\pi_k}Z^*$ — composing the greedy policies $\pi_k,\pi_{k-1},\dots$ picked along the way — is exactly the value distribution of a *nonstationary* optimal policy, which is why the limit set is $\mathcal Z^{**}$ and not $\mathcal Z^*$. A unique fixed point in $\mathcal Z^*$ only comes back if I impose a total ordering on $\Pi^*$ so $\mathcal T$ always resolves ties to the same policy, collapsing it to a single $\mathcal T^{\pi^*}$.

So control is genuinely temperamental: the operator isn't a contraction in any distribution metric, may not have a fixed point, and at best limps to a set of nonstationary distributions. But the proof is also telling me the cure. The instability is the *greedy* swap — a small mean change flips the action and lurches the distribution. The thing that makes it converge in the proof is the *averaging* over successors and over the chasing sequence of greedy policies. So if my practical algorithm uses a smooth, gradient-based update that effectively averages over the distributions it sees rather than committing hard to one greedy bootstrap each step — much like a conservative, partial policy-iteration step — it should absorb the chattering into the approximate solution instead of oscillating. That's the practical bet: model the full distribution, update it gently, and the control instability gets integrated away. With that, on to building it.

What distribution class should the network output? A Gaussian is the obvious parametric choice and has been tried, but the whole point — the win/lose bimodality I started from — is exactly what a Gaussian can't represent. I want something rich enough for multimodality, and cheap. The natural move, borrowing from how discrete autoregressive models handle continuous values, is a *categorical* distribution on a fixed grid: pick $N$ "canonical returns" — atoms — evenly spaced,
$$z_i=V_{\min}+i\,\Delta z,\quad i=0,\dots,N-1,\quad \Delta z=\frac{V_{\max}-V_{\min}}{N-1},$$
and let the network output a probability for each via a softmax:
$$Z_\theta(x,a)=z_i\ \text{w.p.}\ p_i(x,a)=\frac{e^{\theta_i(x,a)}}{\sum_j e^{\theta_j(x,a)}},\qquad Z_\theta(x,a)=\sum_i p_i(x,a)\,\delta_{z_i}.$$
This is highly expressive (any shape on the grid, multimodal included) and computationally trivial — it's just a softmax classifier per action. Bounding the support to $[V_{\min},V_{\max}]$ is also a feature: it bakes in a prior that returns beyond the range are all "equally extreme," which is a cleaner inductive bias than clipping rewards.

Now the loss. From the theory, Wasserstein is *the* metric: the operator contracts in it, and it's robust to the disjoint-support problem I'm about to hit. So my first instinct is to minimize $d_p(\mathcal T Z_\theta, Z_\theta)$ directly. Let me check it's trainable from samples — because in practice I only ever see one sampled transition $(x,a,r,x')$ at a time, and I need the expected sample gradient to equal the true gradient. Suppose the true target is a *mixture* over transitions, $P=P_I$ where $I$ indexes the sampled outcome, and I have a parametric $Q=Z_\theta$. The clean inequality I can rely on for $p=1$ is
$$d_1(P,Q)=d_1\Big(\sum_iA_iP_i,\sum_iA_iQ\Big)\le\sum_i\Pr\{I=i\}\,d_1(P_i,Q)=\mathbb E_I\,d_1(P_i,Q),$$
using the partition lemma with $A_i=\mathbb 1[I=i]$ and the fact that multiplying by an independent indicator scales $d_1$ by its probability. For general $p$ the same reasoning gives the $p$-th-power form $d_p^p(P,Q)\le\mathbb E_I d_p^p(P_i,Q)$. Either way, a sampled Wasserstein objective is an upper bound, not the true mixture distance, and the inequality is generally strict. Concrete counterexample so I believe it: $P=\tfrac12\delta_0+\tfrac12\delta_1$ (so $P_1=\delta_0,P_2=\delta_1$), and $Q=p\,\delta_0+(1-p)\delta_1$. The true distance is $d_1(P,Q)=|p-\tfrac12|$, which is *strictly less than* $\tfrac12$ for any $p\in(0,1)$ and is *minimized at $p=\tfrac12$*. But the expected sampled distance is $\mathbb E_I\,d_1(P_i,Q)=\tfrac12\,d_1(\delta_0,Q)+\tfrac12\,d_1(\delta_1,Q)=\tfrac12(1-p)+\tfrac12 p=\tfrac12$ — *constant in $p$*. Its gradient is zero everywhere, while the true mixture loss has nonzero slope away from $p=\tfrac12$. So SGD on the sampled Wasserstein loss gets a gradient that points nowhere near the true minimizer. That's a wall: the metric that makes the theory work is the one I cannot minimize from sampled transitions. I'd want to sanity-check this on something like CliffWalk against Monte-Carlo ground truth, and I expect Wasserstein-SGD to converge to a wrong fixed point with visible local minima.

So I can't minimize Wasserstein. But I have a softmax categorical and I know how to minimize *cross-entropy* from samples without any bias — that's exactly the unbiased-gradient regime. The problem is geometric, not statistical: when I apply the Bellman update to my atoms, $\mathcal T$ scales each $z_j$ by $\gamma$ and shifts by $r$, so the updated atom $r+\gamma z_j$ almost never lands on the grid $\{z_i\}$. The target distribution $\mathcal T Z_\theta$ and my parametrization $Z_\theta$ live on *disjoint* supports, so a cross-entropy / KL between them is undefined or useless (KL is exactly the vertical, support-matching quantity that ignores values). I need to put the shifted distribution *back onto my grid* before I can take cross-entropy.

The fix that respects the geometry: project each shifted atom's probability mass onto its two nearest grid neighbors by linear interpolation, clamping anything outside $[V_{\min},V_{\max}]$ to the endpoints. Concretely, the shifted atom $\hat{\mathcal T}z_j=[r+\gamma z_j]_{V_{\min}}^{V_{\max}}$ (clamped) falls at fractional grid position $b_j=(\hat{\mathcal T}z_j-V_{\min})/\Delta z\in[0,N-1]$, between $l=\lfloor b_j\rfloor$ and $u=\lceil b_j\rceil$. Linear interpolation gives the lower atom weight $(u-b_j)$ and the upper atom weight $(b_j-l)$ (they sum to 1, so mass is preserved), each times the original probability $p_j$ carried by $z_j$. Accumulating over all source atoms $j$, the $i$-th component of the projected target is
$$\big(\Phi\hat{\mathcal T}Z_\theta(x,a)\big)_i=\sum_{j=0}^{N-1}\Big[\,1-\frac{\big|\,[\hat{\mathcal T}z_j]_{V_{\min}}^{V_{\max}}-z_i\,\big|}{\Delta z}\,\Big]_0^1\,p_j(x',\pi(x')),$$
where $[\cdot]_a^b$ clamps a value to $[a,b]$ and $[\cdot]_0^1$ clamps the weight to $[0,1]$. Let me check the bracket really is the two-nearest-neighbor interpolation: the factor $[1-|\hat{\mathcal T}z_j-z_i|/\Delta z]_0^1$ is a triangular kernel of half-width $\Delta z$ centered at the shifted atom — it's zero for any grid point more than $\Delta z$ away, so only the two straddling atoms $z_l,z_u$ get nonzero weight. At $z_l$: $|\hat{\mathcal T}z_j-z_l|/\Delta z=b_j-l$, giving weight $1-(b_j-l)=u-b_j$ (since $u=l+1$). At $z_u$: $|\hat{\mathcal T}z_j-z_u|/\Delta z=u-b_j$, giving weight $1-(u-b_j)=b_j-l$. Exactly the linear-interpolation split, and only those two atoms ever get nonzero weight — which is what makes the update below linear-time in $N$.

That makes the whole Bellman update a *linear-time-in-$N$* loop: for each source atom $j$, compute $b_j$, add $p_j\,(u-b_j)$ to $m_l$ and $p_j\,(b_j-l)$ to $m_u$. One subtlety in code: if $b_j$ is exactly an integer then $l=u$ and the lower-weight $(u-b_j)=0$ would drop the mass — so I have to send the full $p_j$ to that single atom in that case. Terminal transitions are handled by $\gamma_t=0$, which collapses every shifted atom to $r$.

A projection that silently distorted the distribution would poison the bootstrap, so it's worth running an actual number through it. Take a toy grid $V_{\min}=0,V_{\max}=4,N=5$, so $z=(0,1,2,3,4)$ and $\Delta z=1$. Let the next-state distribution be a point mass at $z=2$, i.e. $p=(0,0,1,0,0)$, and let $r=0.5,\gamma=0.5$. Shift-and-scale: $\hat{\mathcal T}z_j=0.5+0.5\,z_j=(0.5,1,1.5,2,2.5)$, but only $j=2$ carries mass, so the one live shifted atom sits at $\hat{\mathcal T}z_2=1.5$. Its grid position is $b=1.5$, straddled by $l=1,u=2$; the split sends $(u-b)=0.5$ of the mass to $z_1=1$ and $(b-l)=0.5$ to $z_2=2$. So $m=(0,0.5,0.5,0,0)$ — still a probability vector (sums to 1), and its mean is $0.5\cdot1+0.5\cdot2=1.5$. The true Bellman target here is $r+\gamma\cdot2=0.5+0.5\cdot2=1.5$. The projection landed the mean exactly where it should. That's the property I actually care about: $\Phi$ may blur a delta into its two neighbors, but it preserves the mean, so the *mean* of my categorical iterate tracks the same scalar Bellman backup DQN would do — and I can check this isn't a fluke of the symmetric case. For an off-grid shift, say $b=1.3$, the split is $0.7$ to $z_1$ and $0.3$ to $z_2$, contributing $0.7\cdot1+0.3\cdot2=1.3=\Delta z\cdot b$ to the mean — linear interpolation of the *position* is exactly mean-preservation of the *value*, for every atom, so it holds for any mixture by linearity (until a shifted atom hits the clamp at $V_{\min}/V_{\max}$, which is the one place mass-position, hence mean, is deliberately distorted). So the categorical method is mean-consistent with DQN by construction, and the extra atoms are spending their capacity purely on shape.

The target distribution must be formed from a *fixed* target network $\tilde\theta$ (as in DQN, to stabilize the bootstrap), and the greedy action is chosen on the *mean* of the next-state distribution, $a^*=\arg\max_a\sum_i z_i\,p_i(x',a)$ — keeping action selection a drop-in for $\epsilon$-greedy DQN. Then the sample loss is the cross-entropy between the projected target $m=\Phi\hat{\mathcal T}Z_{\tilde\theta}(x,a)$ and the prediction $Z_\theta(x,a)$ — that's the cross-entropy term of $D_{\mathrm{KL}}(\Phi\hat{\mathcal T}Z_{\tilde\theta}(x,a)\,\|\,Z_\theta(x,a))$:
$$\mathcal L_{x,a}(\theta)=-\sum_i m_i\,\log p_i(x,a).$$
The Bellman update has become *multiclass classification* over the $N$ atoms — and that's a loss I can minimize unbiasedly by SGD/Adam, which is the whole reason I retreated from Wasserstein to projection-plus-cross-entropy. KL is insensitive to the atom *values* — it only matches mass — so a genuinely Wasserstein-aware loss would be appealing, but this sampled-gradient obstruction blocks the direct route.

The degenerate case makes the point precise. Take $N=2$, atoms $z=(V_{\min},V_{\max})$. A two-atom pmf $(1-q,q)$ has mean $V_{\min}+q\,\Delta z$, so $q$ and the mean are in bijection — two atoms can encode *only* a Bernoulli, parametrized by its mean. Run the projection: every shifted source atom lands somewhere in $[V_{\min},V_{\max}]$ with grid position $b\in[0,1]$, splits as $(1-b)$ to atom $0$ and $b$ to atom $1$, and summing over the source mass the total weight on atom $1$ is $\sum_j p_j\,b_j=\sum_j p_j(\hat{\mathcal T}z_j-V_{\min})/\Delta z=(\mathbb E[\hat{\mathcal T}Z_\theta]-V_{\min})/\Delta z$. So $\Phi\hat{\mathcal T}Z_\theta=[(\mathbb E[\hat{\mathcal T}Z_\theta]-V_{\min})/\Delta z]_0^1$ exactly — the projected target depends on the shifted distribution *only through its mean*. So the $N=2$ method is mean-tracking dressed up in a softmax: it carries no distributional information beyond what DQN already has (plus the bounded-support prior). That tells me the *whole* value of this construction lives in $N>2$ — the atoms above the first two are the only ones doing distributional work — and I'd want the Atari sweep to confirm that scores climb with $N$ until the grid is fine enough to resolve the return's shape, rather than just assuming "more is better."

Pulling the whole chain together: I refused to keep collapsing the return to its mean; wrote the distributional Bellman recursion $Z\overset{D}{=}R+\gamma Z(X',A')$; found that the right metric is Wasserstein, where $\mathcal T^\pi$ is a $\gamma$-contraction (via P1 scaling and P2 shift) with unique fixed point $Z^\pi$ — and crucially *not* a contraction in KL/TV/Kolmogorov, which are blind to the horizontal $\gamma$-shrink; saw that control is unstable (no contraction in any metric, possibly no fixed point, convergence only to the nonstationary-optimal set), which argues for a smooth averaging update; chose a fixed-support categorical representation $\sum_i p_i\delta_{z_i}$ for multimodality and cheap softmax; discovered I cannot minimize Wasserstein from samples (the sampled gradient is biased — the partition-lemma mixture inequality is strict); and so projected the shifted Bellman target back onto the grid with the linear-interpolation operator $\Phi$ and trained with cross-entropy, turning the distributional Bellman update into multiclass classification on top of the DQN torso. Now the code.

```python
import torch
import torch.nn as nn
import torch.optim as optim

N_ATOMS = 51
V_MIN, V_MAX = -10.0, 10.0  # bounded support; chosen from a small Atari sweep

class CategoricalQNetwork(nn.Module):
    # DQN's conv torso, but the head outputs N atom-logits per action,
    # softmaxed into a categorical return distribution p_i(x,a).
    def __init__(self, n_actions, n_atoms=N_ATOMS, v_min=V_MIN, v_max=V_MAX):
        super().__init__()
        self.n_actions, self.n_atoms = n_actions, n_atoms
        # the fixed grid of "canonical returns" z_i, spacing dz
        self.register_buffer("atoms", torch.linspace(v_min, v_max, n_atoms))
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, n_actions * n_atoms),
        )

    def dist(self, x):
        logits = self.net(x / 255.0).view(-1, self.n_actions, self.n_atoms)
        return torch.softmax(logits, dim=2)            # p_i(x,a)

    def greedy(self, x):
        pmfs = self.dist(x)
        q = (pmfs * self.atoms).sum(2)                 # Q(x,a) = sum_i z_i p_i(x,a) -- act on the MEAN
        a = q.argmax(1)
        return a, pmfs[torch.arange(len(x)), a]

def project(target_net, rewards, next_obs, dones, gamma,
            v_min=V_MIN, v_max=V_MAX, n_atoms=N_ATOMS):
    # Form the projected target  Phi T-hat Z_{theta~}  from the TARGET network.
    with torch.no_grad():
        _, next_pmfs = target_net.greedy(next_obs)            # a* greedy on next-state mean
        atoms = target_net.atoms
        dz = atoms[1] - atoms[0]
        # shift+scale each atom: T-hat z_j = r + gamma z_j  (gamma=0 at terminal), then clamp
        tz = (rewards + gamma * atoms * (1.0 - dones)).clamp(v_min, v_max)
        b = (tz - v_min) / dz                                 # fractional grid position in [0, N-1]
        l, u = b.floor().clamp(0, n_atoms - 1), b.ceil().clamp(0, n_atoms - 1)
        # split p_j onto the two nearest atoms: lower gets (u - b), upper gets (b - l).
        # (l == u) handles integer b_j so the mass isn't dropped to a zero weight.
        w_l = (u + (l == u).float() - b) * next_pmfs          # -> atom l
        w_u = (b - l) * next_pmfs                             # -> atom u
        m = torch.zeros_like(next_pmfs)
        for i in range(m.size(0)):
            m[i].index_add_(0, l[i].long(), w_l[i])
            m[i].index_add_(0, u[i].long(), w_u[i])
    return m                                                  # m_i = (Phi T-hat Z)_i

def loss_fn(online_net, obs, actions, target_pmfs):
    pred = online_net.dist(obs)[torch.arange(len(obs)), actions.flatten()]  # p_i(x,a) for taken action
    # cross-entropy term of KL(Phi T-hat Z_{theta~} || Z_theta): -sum_i m_i log p_i(x,a)
    return -(target_pmfs * pred.clamp(1e-5, 1 - 1e-5).log()).sum(-1).mean()

# training step (off-policy, replay, periodic target sync) -- a drop-in for DQN's squared loss:
#   data = replay.sample(batch)
#   m   = project(target_net, data.rewards, data.next_obs, data.dones, gamma)
#   loss = loss_fn(online_net, data.obs, data.actions, m)
#   optimizer.zero_grad(); loss.backward(); optimizer.step()       # Adam, eps=0.01/batch
#   periodically: target_net.load_state_dict(online_net.state_dict())
```
