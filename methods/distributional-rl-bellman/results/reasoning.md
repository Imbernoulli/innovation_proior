Let me start where it actually hurts. I have an agent collecting a random return $Z(x,a)=\sum_{t\ge0}\gamma^t R(x_t,a_t)$, and the entire apparatus I've inherited — temporal-difference learning, SARSA, Q-learning, fitted value iteration — works with a single number per state-action: the expectation $Q(x,a)=\mathbb{E}Z(x,a)$. The justification is clean and I know it cold: $Q^\pi$ satisfies $Q^\pi(x,a)=\mathbb{E}R(x,a)+\gamma\,\mathbb{E}_{P,\pi}Q^\pi(x',a')$, the operator $\mathcal{T}^\pi$ that maps $Q\mapsto \mathbb{E}R+\gamma\,\mathbb{E}_{P,\pi}Q$ is a $\gamma$-contraction in the sup norm, Banach gives me a unique fixed point and geometric convergence, and the optimality operator $\mathcal{T}Q=\mathbb{E}R+\gamma\,\mathbb{E}_P\max_{a'}Q(x',a')$ does the same for $Q^*$. That contraction is the thing the whole field stands on.

But stare at the object I threw away. The return isn't a number, it's a random variable, and a wild one. Rewards can be stochastic, transitions can be stochastic, the policy can be stochastic, and all three compound at every one of infinitely many steps. The resulting law of $Z$ is generically spread out, frequently multimodal — picture a state where one branch of the future means "I survive and rack up points" and the other means "I die and the episode ends at zero." Those are two utterly different futures, and $Q$ reports their *average*, a value that may be attained by neither. Why am I modelling only the mean? From a supervised-learning reflex it even looks perverse: if I had targets, I'd predict the whole distribution and never dream of throwing away everything but the first moment. The catch is I have no targets — I have to bootstrap, "learn a guess from a guess." So the real question is whether I can bootstrap *distributions* the way I bootstrap means.

So let me try to write a Bellman equation not for $\mathbb{E}Z$ but for $Z$ itself. The scalar one came from $Q^\pi(x,a)=\mathbb{E}[R(x,a)+\gamma Q^\pi(x',a')]$. Strip the outer expectation and keep the random variables:
$$Z(x,a) \;\overset{D}{=}\; R(x,a) + \gamma\, Z(X',A'),\qquad X'\sim P(\cdot\mid x,a),\ A'\sim\pi(\cdot\mid X').$$
The $\overset{D}{=}$ matters — this is an equality *in distribution*, not of particular random variables. The right-hand side is a compound distribution built from three sources of randomness that I'll insist are independent: the reward $R$, the next state-action $(X',A')$ drawn through the environment and policy, and the next return $Z(X',A')$ itself. Define the transition operator $P^\pi Z(x,a)\overset{D}{:=}Z(X',A')$ — it takes the value distribution and pushes it through one random step — and then the distributional Bellman operator
$$\mathcal{T}^\pi Z(x,a) \overset{D}{:=} R(x,a) + \gamma\, P^\pi Z(x,a).$$
It *looks* like the scalar operator with the expectations peeled off, but it's a different animal: it acts on a whole function from state-action pairs to distributions, mixing, scaling by $\gamma$, and shifting by $R$.

Does this even make sense — is there anything for it to converge *to*? Let me sanity-check on something small enough to compute by hand. One state, reward Bernoulli$(\tfrac12)$, $\gamma=\tfrac12$. The mean equation says $V=\tfrac12+\tfrac12 V$, so $V=1$; fine. But what's the *return distribution*? The return is $R_0+\tfrac12 R_1+\tfrac14 R_2+\cdots$ with each $R_t\in\{0,1\}$ an independent fair coin. That is $2\cdot 0.R_0R_1R_2\cdots$ in binary, apart from the usual measure-zero ambiguity of dyadic expansions, so a uniform random binary string gives a number uniform on $[0,2]$. The mean is $1$, but the return distribution is the *uniform law on $[0,2]$*. The distributional equation has real, non-trivial content; the mean was hiding an entire continuous distribution. Good — there's something here worth pinning down, and the question is whether iterating $\mathcal{T}^\pi$ pins it down uniquely.

Now, "unique fixed point reached by iteration" is a contraction statement, and the scalar proof used the sup norm. I can't use the sup norm on distributions — what's $\|Z_1-Z_2\|_\infty$ when $Z_1,Z_2$ are *distributions*? I need a metric *between distributions* in which $\mathcal{T}^\pi$ contracts. And here I get a strong nudge from outside RL. This recursion $Z\overset{D}{=}R+\gamma Z'$ is a stochastic recursive equation of sum type, and that exact shape is what shows up in the probabilistic analysis of algorithms. Rösler analyzing Quicksort had $(X_n-\mathbb{E}X_n)/n$ converging to a limit $Y$ whose law is the unique fixed point of a map $S$ on distributions — and the way he nailed it was to show $S$ is a *contraction in the Wasserstein metric* $d_2$, get the fixed point from Banach, and then read off recursive formulas for every moment. The template is precisely mine, with $\gamma$ standing in for his contraction modulus. So Wasserstein is the metric to try.

Let me write it down carefully because I'm going to need its fine structure. For two cdfs $F,G$,
$$d_p(F,G):=\inf_{U,V}\|U-V\|_p,$$
the infimum over all couplings $(U,V)$ with marginals $F,G$. The infimum is achieved by the inverse-cdf coupling: sample one uniform $\mathcal{U}\sim\mathrm{Unif}[0,1]$ and set $U=F^{-1}(\mathcal{U})$, $V=G^{-1}(\mathcal{U})$, so
$$d_p(F,G)=\|F^{-1}(\mathcal{U})-G^{-1}(\mathcal{U})\|_p=\Big(\int_0^1|F^{-1}(u)-G^{-1}(u)|^p\,du\Big)^{1/p}\quad(p<\infty).$$
The thing I care about is that this is a *horizontal* distance — it measures how far you have to slide probability mass along the value axis to turn one distribution into the other. That's the property that should make a discount act like a contraction modulus, because shrinking a random variable toward zero by $\gamma$ literally shrinks horizontal distances by $\gamma$. Let me extract the three properties I'll lean on, for a scalar $a$ and a random variable $A$ independent of $U,V$:
$$\text{(P1)}\ d_p(aU,aV)\le|a|\,d_p(U,V),\quad \text{(P2)}\ d_p(A+U,A+V)\le d_p(U,V),\quad \text{(P3)}\ d_p(AU,AV)\le\|A\|_p\,d_p(U,V).$$
P1 is the scaling I just argued for. P2 says a shared additive term doesn't increase the distance — couple $U,V$ optimally and add the same $A$ to both. P3 is the mixing-by-a-random-multiplier version. These all follow from taking the optimal coupling on the right and exhibiting it as a (suboptimal, hence upper-bounding) coupling on the left.

I also need to lift $d_p$ from single distributions to whole value distributions, which are indexed by $(x,a)$. The right object is the *maximal* (supremal) Wasserstein metric,
$$\overline{d}_p(Z_1,Z_2):=\sup_{x,a} d_p\big(Z_1(x,a),Z_2(x,a)\big).$$
A sup of metrics, with the triangle inequality surviving because each $d_p$ obeys it and then I take sup of the two pieces separately — so $\overline d_p$ is a genuine metric on value distributions. This is the analogue of the sup norm: it asks for the worst state-action pair.

Now the contraction. Take $Z_1,Z_2$. At a fixed $(x,a)$,
$$d_p\big(\mathcal{T}^\pi Z_1(x,a),\,\mathcal{T}^\pi Z_2(x,a)\big)=d_p\big(R+\gamma P^\pi Z_1,\,R+\gamma P^\pi Z_2\big).$$
The same reward $R$ sits on both sides — additive and shared — so P2 kills it:
$$\le d_p\big(\gamma P^\pi Z_1,\,\gamma P^\pi Z_2\big).$$
The same $\gamma$ multiplies both, so P1 pulls it out:
$$\le \gamma\, d_p\big(P^\pi Z_1,\,P^\pi Z_2\big).$$
And $P^\pi Z(x,a)$ is just $Z(X',A')$ for the random successor — a mixture over $(x',a')$ of the laws $Z_i(x',a')$. Couple the same successor draw on both sides, and conditional on that successor couple the two return laws optimally; the average conditional distance is then no larger than the worst successor distance:
$$\le \gamma \sup_{x',a'} d_p\big(Z_1(x',a'),Z_2(x',a')\big).$$
This bound is uniform in $(x,a)$, so taking the sup on the left,
$$\overline d_p\big(\mathcal{T}^\pi Z_1,\mathcal{T}^\pi Z_2\big)\le \gamma\,\overline d_p(Z_1,Z_2).$$
There it is: $\mathcal{T}^\pi$ is a $\gamma$-contraction in $\overline d_p$. Banach hands me a unique fixed point; by inspection the recursion $Z\overset{D}{=}R+\gamma Z'$ is solved by the actual return $Z^\pi$, so the fixed point *is* $Z^\pi$. With bounded moments the iterates $Z_{k+1}=\mathcal{T}^\pi Z_k$ converge to $Z^\pi$ in $\overline d_p$ for every $1\le p\le\infty$, and since Wasserstein convergence controls moments, every moment converges geometrically too. Distributional policy evaluation is on exactly as firm a footing as the scalar kind. And the proof is suspiciously short — it's the three metric properties and the definition of $P^\pi$, nothing more. That's because all the work was in *choosing the metric*.

Let me make sure I appreciate why the choice was forced, because it's tempting to think any distribution distance would do. Try total variation. The discount scales the support: $\gamma$ maps two point masses at $a,b$ to point masses at $\gamma a,\gamma b$. Under Wasserstein the distance went from $|a-b|$ to $\gamma|a-b|$ — it shrank. Under total variation, two disjoint atoms have distance $1$ whether they sit at $a,b$ or at $\gamma a,\gamma b$; TV doesn't see *where* the mass is, only *how much overlaps*. So $\gamma$ is not a contraction modulus in TV at all — and indeed Chung & Sobel showed the distributional operator fails to contract in total variation. The same blindness afflicts KL divergence and the Kolmogorov distance: they're vertical, likelihood-style comparisons, insensitive to the value axis that the discount acts on. Wasserstein contracts here *because* it's a transport metric and the discount is a transport. That's the whole reason the metric is non-negotiable.

One more loose end on evaluation before control: variance. I'd like to say $\mathcal{T}^\pi$ contracts variance too, and it's tempting to read it off $d_2$. But careful — for the optimal coupling $C=U-V$,
$$d_2^2(U,V)\le \mathbb{E}[(U-V)^2]=\mathrm{Var}(C)+(\mathbb{E}C)^2,$$
so $d_2$ bounds a mean-plus-variance combination, not the variance gap on its own; I can't get $|\mathrm{Var}(\mathcal{T}^\pi Z)-\mathrm{Var}(Z^\pi)|$ directly out of it. Do it head-on with the law of total variance. By independence of the reward and the next-return draw,
$$\mathrm{Var}\big(\mathcal{T}^\pi Z(x,a)\big)=\mathrm{Var}(R(x,a))+\gamma^2\,\mathbb{E}_{P,\pi}\mathrm{Var}\big(Z(X',A')\big)+\gamma^2\,\mathrm{Var}_{P,\pi}\big(\mathbb{E}Z(X',A')\big).$$
The last term is driven by the mean process. If two candidate value distributions have the same successor means, the full variance difference obeys
$$\big\|\mathrm{Var}(\mathcal{T}^\pi Z_1)-\mathrm{Var}(\mathcal{T}^\pi Z_2)\big\|_\infty\le \gamma^2\,\big\|\mathrm{Var}Z_1-\mathrm{Var}Z_2\big\|_\infty.$$
More generally, after subtracting the mean-driven term, the conditional-variance component obeys that same $\gamma^2$ bound. That is the precise Sobel-style contraction: variance does not come from $d_2$ directly, and as a standalone map over arbitrary distributions with different means it has an extra transition-mean term, but the conditional variance itself propagates with modulus $\gamma^2$. For centered moments of order $p>2$ the operator isn't a contraction, but the iterates' moments still converge geometrically, extending Rösler's argument. Evaluation is fully understood.

Now control, where I expect the same story and am about to be wrong. I want a policy that maximizes value, and the optimal *value* is unique — all optimal policies share $Q^*$. But what is an "optimal value distribution"? It has to be the value distribution *of an optimal policy*, $\mathcal{Z}^*:=\{Z^{\pi^*}:\pi^*\in\Pi^*\}$. And immediately something is off that has no scalar analogue: it is *not* enough to have mean $Q^*$. Two policies can both be optimal-in-mean yet induce different return *distributions*, so in general there are *many* optimal value distributions. In the scalar world the max collapses everything to one $Q^*$; here the max over actions hides a *choice of policy*, and the distribution remembers which choice was made.

Let me name that choice. A greedy policy for $Z$ maximizes $\mathbb{E}Z$: $\mathcal{G}_Z=\{\pi:\sum_a\pi(a\mid x)\mathbb{E}Z(x,a)=\max_{a'}\mathbb{E}Z(x,a')\}$. A distributional optimality operator is any $\mathcal{T}$ with $\mathcal{T}Z=\mathcal{T}^\pi Z$ for some $\pi\in\mathcal{G}_Z$ — a *selection rule*. In the scalar case the tie-break inside $\max$ was invisible because all maximizers give the same value; here, picking action $a$ vs. action $a'$ when both are greedy can attach a *different distribution*, so the operator isn't even uniquely defined without specifying the rule. That's the first warning sign.

The mean still behaves, at least. The distributional operator commutes with expectation — $\mathbb{E}\,\mathcal{T}_D Z=\mathcal{T}_E\,\mathbb{E}Z$, where $\mathcal{T}_E$ is the ordinary optimality operator — so
$$\|\mathbb{E}\,\mathcal{T}Z_1-\mathbb{E}\,\mathcal{T}Z_2\|_\infty=\|\mathcal{T}_E\mathbb{E}Z_1-\mathcal{T}_E\mathbb{E}Z_2\|_\infty\le\gamma\|\mathbb{E}Z_1-\mathbb{E}Z_2\|_\infty,$$
and $\mathbb{E}Z_k\to Q^*$ geometrically, exactly as before. So I might expect $Z_k$ itself to march nicely to some fixed point in $\mathcal{Z}^*$. Let me test that on the smallest example I can build.

Two states. A forced transition $x_1\to x_2$. From $x_2$: action $a_1$ gives reward $0$; action $a_2$ gives $1+\epsilon$ or $-1+\epsilon$ with equal probability — mean $\epsilon>0$, so $a_2$ is the (unique) optimal action. Both actions terminate. Unique optimal policy, so a unique $Z^*$: $Z^*(x_2,a_1)=0$, $Z^*(x_2,a_2)=\epsilon\pm1$ (shorthand for $1+\epsilon$ or $-1+\epsilon$ each w.p. $\tfrac12$), and $Z^*(x_1)=\epsilon\pm1$ since $x_1$ just inherits the optimal action's return.

Now take a $Z$ equal to $Z^*$ everywhere except I flip the sign of the offset at the one cell $(x_2,a_2)$: $Z(x_2,a_2)=-\epsilon\pm1$, i.e. $1-\epsilon$ or $-1-\epsilon$ each w.p. $\tfrac12$. How far is $Z$ from $Z^*$? Only that one cell differs, and under the quantile coupling I line up $1-\epsilon\leftrightarrow1+\epsilon$ (gap $2\epsilon$) and $-1-\epsilon\leftrightarrow-1+\epsilon$ (gap $2\epsilon$), so
$$\overline d_1(Z,Z^*)=d_1\big(Z(x_2,a_2),Z^*(x_2,a_2)\big)=\tfrac12(2\epsilon)+\tfrac12(2\epsilon)=2\epsilon.$$
Tiny — I nudged $Z$ a distance $2\epsilon$ from the fixed point. Now apply $\mathcal{T}$ and watch the mean. At $x_2$ under $Z$: $\mathbb{E}Z(x_2,a_1)=0$ but $\mathbb{E}Z(x_2,a_2)=-\epsilon<0$. So the greedy action at $x_2$ has *flipped* to $a_1$. The backup at $x_1$ therefore copies $a_1$'s distribution: $\mathcal{T}Z(x_1)=Z(x_2,a_1)=\delta_0$, a point mass at $0$. But the fixed point has $\mathcal{T}Z^*(x_1)=Z^*(x_1)=\epsilon\pm1$, a spread bimodal law. The distance after one step:
$$\overline d_1(\mathcal{T}Z,\mathcal{T}Z^*)=d_1\big(\delta_0,\ \epsilon\pm1\big)=\tfrac12|0-(-1+\epsilon)|+\tfrac12|0-(1+\epsilon)|=\tfrac12|1-\epsilon|+\tfrac12|1+\epsilon|.$$
For small $\epsilon$ that's about $1$ — and $1>2\epsilon$. So $\overline d_1(\mathcal{T}Z,\mathcal{T}Z^*)>\overline d_1(Z,Z^*)$: applying the operator *expanded* the distance. With an undiscounted backup this is already not a non-expansion; with $\gamma<1$ the same construction shows it's not a contraction in $\overline d_1$, and a little more care extends it to any metric that even separates $Z$ from $\mathcal{T}Z$. The control operator is not a contraction in *any* distribution metric.

Let me make sure I understand *why*, because the mean was so well-behaved. An arbitrarily small perturbation to the *mean* of one action's return — here, dragging $\mathbb{E}Z(x_2,a_2)$ from $+\epsilon$ to $-\epsilon$ — flips the $\arg\max$, and the flip doesn't nudge the backed-up distribution, it *swaps it wholesale*: a spread bimodal law gets replaced by a Dirac at $0$. The greedy selection is a discontinuous function of the value distribution. Look at it as a map of $\epsilon$: for $\epsilon>0$ I back up a bimodal distribution, for $\epsilon<0$ I back up $\delta_0$; the operator isn't even continuous at $\epsilon=0$. So the mean can converge as smoothly as it likes while the *distribution* attached to the chosen action jumps around. That's the instability, and it's intrinsic to greedy maximization, not an artifact of my example.

It gets worse, and I should be honest about how. First, a fixed point need not exist. Take the same example with $\epsilon=0$, so both actions at $x_2$ now have mean $0$ and the selection rule must break the tie — say it picks $a_2$ when $Z(x_1)=\delta_0$ and $a_1$ otherwise. Then the iterates $\mathcal{T}Z^*(x_1),\mathcal{T}^2Z^*(x_1),\dots$ alternate between copying $Z^*(x_2,a_1)=\delta_0$ and $Z^*(x_2,a_2)$ into $x_1$: each choice flips the tie-break condition for the next step, so the operator oscillates and never settles. A genuine limit cycle, no fixed point at all.

Second, even when a fixed point exists it needn't be where the iterates go. One state, two actions, $\gamma=\tfrac12$: $a_1$ gives reward $\tfrac12$, $a_2$ gives $0$ or $1$ each w.p. $\tfrac12$; both have mean $\tfrac12$, so both are optimal. Follow $a_1$ forever (probability $p=0$ of ever choosing $a_2$): return $=\frac{1}{1-\gamma}\cdot\tfrac12=1$, a point mass, $0.11111\cdots$ in binary. Follow $a_2$ forever ($p=1$): return $=R_0.R_1R_2\cdots$ in binary with fair bits, which is *uniform on $[0,2]$*. Any fixed mixing probability $p$ gives some law with mass spread across $[0,2]$. Now follow the *nonstationary* policy "take $a_1$ once, then $a_2$ forever": by inspection the return is uniform on $[\tfrac12,\tfrac32]$ — and that distribution equals the return of *no* stationary $p$. So an operator that switches which greedy action it selects depending on the exact incoming distribution can chase a return that no stationary optimal policy produces. The iterates converge, but to the value distribution of a *nonstationary* sequence of optimal policies.

That tells me what the right convergence claim is. Define $\mathcal{Z}^{**}$, the *nonstationary optimal value distributions*: laws of returns under sequences of optimal policies. The honest theorem is that with finitely many actions, $Z_k$ converges *pointwise* to this larger set,
$$\lim_{k\to\infty}\inf_{Z^{**}\in\mathcal{Z}^{**}}d_p\big(Z_k(x,a),Z^{**}(x,a)\big)=0\quad\forall x,a,$$
uniformly if $\mathcal{X}$ is finite, and to a *unique* $Z^*\in\mathcal{Z}^*$ only if I additionally impose a total ordering $\prec$ on optimal policies and always break greedy ties by the $\prec$-first policy — in which case $\mathcal{T}=\mathcal{T}^{\pi^*}$ for that fixed $\pi^*$ and I'm back to the contraction of evaluation.

Let me actually prove the pointwise convergence, since it's the substantive control result and it's not just hand-waving "the mean converges so the rest follows." The idea is that the mean converging fast forces the greedy policy to *become* optimal, state by state, and once enough of the future is acting optimally the distribution is squeezed toward $\mathcal{Z}^{**}$. Set $B:=2\sup_{Z}\|Z\|_\infty<\infty$ and $\epsilon_k:=\gamma^k B$. From the mean contraction, $|Q_k(x,a)-Q^*(x,a)|\le\gamma^k|Q_0-Q^*|\le\epsilon_k$. Define the states whose optimal-action *mean gap* already beats the current error,
$$\mathcal{X}_k:=\Big\{x:\ Q^*(x,\pi^*(x))-\max_{a\ne\pi^*(x)}Q^*(x,a)>2\epsilon_k\Big\}.$$
For $x\in\mathcal{X}_k$, since each $Q_k$ is within $\epsilon_k$ of $Q^*$, the greedy choice $\arg\max_a Q_k(x,a)$ must be the optimal action — the $2\epsilon_k$ margin can't be overturned by two $\epsilon_k$ errors. And because the action set is finite, every gap $\Delta(x)=Q^*(x,\pi^*(x))-\max_{a\ne\pi^*(x)}Q^*(x,a)$ is strictly positive, so for $k$ large enough that $\gamma^k B<\Delta(x)/2$, $x$ is in $\mathcal{X}_k$ and stays. Every state eventually becomes "solved."

But solving a state isn't enough — its backup pulls in successor distributions, which must also be solved, and so must theirs, several steps deep. So fix a leakage tolerance $\delta>0$ in $L_p$; for finite $p$ this means I ask the bad-successor probability to be at most $\delta^p$. Set $\mathcal{X}_{k,0}:=\mathcal{X}_k$, and recursively
$$\mathcal{X}_{k,i}:=\big\{x\in\mathcal{X}_k:\ P(\mathcal{X}_{k-1,i-1}\mid x,\pi^*(x))\ge1-\delta^p\big\}$$
— solved states whose unsolved-successor indicator has $L_p$ norm at most $\delta$. Since $\mathcal{X}_{k,0}\uparrow\mathcal{X}$ and probability is continuous, $P(\mathcal{X}_{k,i}\mid x,\pi^*(x))\to1$, so by induction $\mathcal{X}_{k,i}\uparrow\mathcal{X}$ for every depth $i$: any state is eventually $i$-deep solved. For $p=1$ this is the usual $1-\delta$ form.

Now bound the distance to the fixed point, using the indicator $S_i^k(x)=\mathbb{1}[x\in\mathcal{X}_{k,i}]$ and its complement $\bar S_i^k$. These two indicators partition $\Omega$, which is exactly the setting for a lemma I'll need: for a partition $\{A_i\}$ of $\Omega$, $d_p(U,V)\le\sum_i d_p(A_iU,A_iV)$ — Wasserstein is subadditive across a partition, because conditioning on which piece of the partition you're in only *removes* freedom to reorder mass across pieces, so the partitioned infimum is at least the joint one. Write $W_k(x):=Z_k(x,\pi_k(x))$, and for $x\in\mathcal{X}_{k+1,i+1}$ where the greedy policy is optimal, split the backed-up successor into a solved part and an unsolved part:
$$d_p(W_{k+1}(x),W^*(x))=d_p(\mathcal{T}W_k(x),\mathcal{T}W^*(x))\le\gamma\,d_p(P^{\pi_k}W_k(x),P^{\pi^*}W^*(x))$$
(using P1 and P2 to peel off $\gamma$ and the shared reward), and then by the partition lemma,
$$\le\gamma\,d_p\big(S_i^kW_k(X'),S_i^kW^*(X')\big)+\gamma\,d_p\big(\bar S_i^kW_k(X'),\bar S_i^kW^*(X')\big).$$
The unsolved term is small: let $\delta_i:=\|\bar S_i^k(X')\|_p=\Pr\{X'\notin\mathcal{X}_{k,i}\}^{1/p}$ for finite $p$, and by P3,
$$d_p\big(\bar S_i^kW_k(X'),\bar S_i^kW^*(X')\big)\le\|\bar S_i^k(X')\|_p\sup_{x'}d_p(W_k(x'),W^*(x'))\le\delta_i B\le\delta B,$$
since distributions are separated by at most the return-range diameter $B$ and $\delta_i\le\delta$ for $x\in\mathcal{X}_{k+1,i+1}$. So
$$d_p(W_{k+1}(x),W^*(x))\le\gamma\,d_p\big(S_i^kW_k(X'),S_i^kW^*(X')\big)+\gamma\delta B.$$
The solved term is itself a backup one step further along solved states, so I can recurse on $i$: peeling $i$ layers gives
$$d_p(W_{k+i}(x),W^*(x))\le\gamma^i\,d_p\big(S_0^kW_k(X''),S_0^kW^*(X'')\big)+\frac{\delta B}{1-\gamma}\le\gamma^i B+\frac{\delta B}{1-\gamma},$$
the geometric series $\sum_{j<i}\gamma^j\delta B\le\delta B/(1-\gamma)$ collecting the unsolved leakage. Given any $x$ and any target accuracy, choose $\delta$ small, then $i$ large, then $k$ large, and the right side drops below it. One more application of $\mathcal{T}$ carries the bound from $W_k$ to $Z_k(x,a)$. With multiple optimal policies I redefine $\mathcal{X}_{k,i}$ to require the successor condition for *all* $\pi^*\in\Pi^*$, run the same argument along the nonstationary product $\mathcal{T}^{\bar\pi_k}=\mathcal{T}^{\pi_k}\cdots\mathcal{T}^{\pi_{k-i+1}}$, and land at $\mathcal{Z}^{**}$ instead of a single $Z^*$. That's the convergence theorem, blemishes and all: the mean races to $Q^*$, but the distribution only limps toward the nonstationary set, and pinning a *stationary* limit needs a consistent tie-break.

So I have a clean evaluation theory and a deliberately bleak control theory, and the bleakness is itself a finding: it argues for learning algorithms that don't pretend the policy is stationary — that average over the chattering greedy choices rather than commit to one. Now I have to *implement* the operator, and the moment I try, two walls go up.

I need a finite-parameter family of distributions. The expressive, computation-friendly choice is a discrete distribution on a fixed grid of "canonical returns" $\{z_i=V_{\min}+i\,\Delta z\}_{i=0}^{N-1}$, $\Delta z=\frac{V_{\max}-V_{\min}}{N-1}$, with probabilities $p_i(x,a)=\mathrm{softmax}(\theta_i(x,a))$ — the network emits $N$ logits per action. First wall: apply the Bellman update to such a $Z_\theta$ and you get $\mathcal{T}Z_\theta$ supported on $\{r+\gamma z_i\}$, which is the grid scaled by $\gamma$ and shifted by $r$ — almost never the original grid. So $\mathcal{T}Z_\theta$ and $Z_\theta$ have *disjoint supports*, and most losses can't even compare them. My own theory says: minimize Wasserstein, which doesn't care about support mismatch and is exactly the metric the operator contracts in. That feels like the principled move.

Second wall, and it's the one that kills the principled move: I learn from *sampled* transitions, one $(x,a,r,x')$ at a time, and Wasserstein cannot be minimized from samples by stochastic gradient descent. Here's why, and it's a fact about couplings, not about RL. Suppose the true target is a mixture $P=P_I$ over a random index $I$ (the randomness in reward and next state). For any fixed prediction $Q$,
$$d_p(P,Q)\le\mathbb{E}_{i\sim I}\,d_p(P_i,Q),$$
and the inequality is in general *strict*, with the gradients also differing: $\nabla_Q d_p(P_I,Q)\ne\mathbb{E}_i\nabla_Q d_p(P_i,Q)$. So if I form the loss from a single sampled transition and average, I'm minimizing the right-hand side, an *upper bound* that bottoms out in the wrong place. Concretely: let $P$ be Bernoulli$(\tfrac12)$ on $\{0,1\}$, and let $Q$ put probability $p$ on $0$. Then $d_1(P,Q)=|p-\tfrac12|$ — a V-shape bottoming out at $p=\tfrac12$ (distance $0$), equal to $\tfrac12$ at the endpoints $p\in\{0,1\}$ and strictly less in between. But the sampled version, drawing the outcome $0$ or $1$ each w.p. $\tfrac12$ and averaging $d_1$ of a Dirac against $Q$, is $\tfrac12 p+\tfrac12(1-p)=\tfrac12$ for *every* $p$ — dead flat. The true loss has a minimum, the sampled loss has no gradient toward it. Minimizing the sampled Wasserstein doesn't minimize the Wasserstein. The principled loss is a no-go under sampling.

I need a backup that lives on the *fixed* grid and is trainable from samples. Stop fighting the support mismatch — *project* the Bellman update back onto $\{z_i\}$. For each atom $z_j$ compute its image under the sample backup, $\widehat{\mathcal{T}}z_j=[r+\gamma z_j]_{V_{\min}}^{V_{\max}}$ (clipped to the grid range), and then *smear* its probability $p_j(x',\pi(x'))$ onto the two nearest grid points by linear interpolation — the closer atom gets more, by the complementary fractional distance. The $i$-th coordinate of the projected target is
$$\big(\Phi\widehat{\mathcal{T}}Z_\theta(x,a)\big)_i=\sum_{j=0}^{N-1}\Big[1-\frac{\big|[\widehat{\mathcal{T}}z_j]_{V_{\min}}^{V_{\max}}-z_i\big|}{\Delta z}\Big]_0^1\;p_j(x',\pi(x')),$$
the bracket clipped to $[0,1]$ so only the immediate neighbours receive mass. Now $\Phi\widehat{\mathcal{T}}Z_\theta$ is a distribution on the *same* grid as $Z_\theta$, so I can compare them with a likelihood loss after all — minimize the cross-entropy of the KL divergence $D_{\mathrm{KL}}(\Phi\widehat{\mathcal{T}}Z_{\tilde\theta}(x,a)\,\|\,Z_\theta(x,a))$, treating the next-state parameters $\tilde\theta$ as a fixed target. The Bellman update has become *multiclass classification*: a soft target vector over $N$ classes, cross-entropy against the predicted logits. That's trivially minimizable by SGD on a single transition, and the projection step is computable in time linear in $N$ — for each source atom, find the two neighbours and split.

Is this projected operator *principled*, or just a convenient hack that happens to live on the grid? The KL/cross-entropy I'm now minimizing is a vertical, likelihood loss — exactly the kind I argued the discount does *not* contract. But I'm no longer asking the *raw* operator to contract in KL; I'm asking the *projected* operator $\Phi\mathcal{T}^\pi$ to behave, and the right metric for the projection turns out not to be Wasserstein at all but the Cramér distance $\ell_2(F,G)=\big(\int(F(y)-G(y))^2\,dy\big)^{1/2}$ — still a value-axis metric, so the discount still shrinks it. The scaling is different from Wasserstein: $\ell_2$ scales by $\sqrt{\gamma}$ under discounting, equivalently $\ell_2^2$ scales by $\gamma$. The piecewise-linear projection $\Phi$ onto the fixed grid is an orthogonal-style non-expansion in this geometry, so composing it with the discounted backup makes $\Phi\mathcal{T}^\pi$ a $\sqrt{\gamma}$-contraction in Cramér distance, with its own fixed point reachable by iteration. The categorical projection isn't a betrayal of the theory; it is the same contraction idea in the metric whose projection survives the fixed grid. The cross-entropy is then the sampling-friendly way to fit the projected categorical target. With $N=2$ a degenerate one-parameter version just regresses the clipped mean, a Bernoulli backup; the interesting regime is many atoms.

The concrete update now has three moving parts: `bellman_target`/projection is $\Phi\widehat{\mathcal{T}}$, the loss is the cross-entropy term of the KL, and the representation is the $N$-atom categorical that the discrete approximation demanded.

```python
import numpy as np

# Distributional policy evaluation: the operator T^pi  Z =_D R + gamma Z(X',A')
# is a gamma-contraction in the maximal Wasserstein metric  Wbar_p(Z1,Z2) =
#   sup_{x,a} d_p(Z1(x,a), Z2(x,a)),  d_p(F,G) = inf over couplings ||U-V||_p,
# so iterating it converges to the true return distribution Z^pi. The control
# operator T Z = T^pi Z for pi greedy w.r.t. E Z is NOT a contraction in any
# distribution metric (only the mean contracts): a vanishing change in one
# action's mean flips the argmax and swaps in a wholly different distribution.
# Below: the categorical / projected operator that makes this trainable from
# samples, since the Wasserstein loss cannot be minimized by sampled SGD.

V_MIN, V_MAX, N = -10.0, 10.0, 51                 # support hyperparameters
z = np.linspace(V_MIN, V_MAX, N)                  # canonical returns {z_i}
dz = (V_MAX - V_MIN) / (N - 1)

def project_bellman_update(rewards, terminals, next_probs, gamma):
    """Phi T-hat Z: apply the sample Bellman backup to each atom, then smear
    its probability onto the two nearest grid points (linear interpolation).
    next_probs[b, i] = p_i(x', pi(x')) chosen greedily by E[Z]; this is the
    soft classification target for the cross-entropy loss."""
    B = rewards.shape[0]
    m = np.zeros((B, N))
    for b in range(B):
        g = gamma * (1.0 - terminals[b])           # terminal -> gamma_t = 0
        for j in range(N):
            # T-hat z_j = [ r + gamma z_j ]_{V_min}^{V_max}
            Tz = min(V_MAX, max(V_MIN, rewards[b] + g * z[j]))
            b_idx = (Tz - V_MIN) / dz               # position on the grid, in [0, N-1]
            lo, hi = int(np.floor(b_idx)), int(np.ceil(b_idx))
            # distribute p_j(x', pi(x')) to immediate neighbours by 1 - |.|/dz
            if lo == hi:
                m[b, lo] += next_probs[b, j]
            else:
                m[b, lo] += next_probs[b, j] * (hi - b_idx)
                m[b, hi] += next_probs[b, j] * (b_idx - lo)
    return m                                        # categorical target on {z_i}

def greedy_next_action(next_logits):
    """Greedy w.r.t. the EXPECTED return  Q(x',a) = sum_i z_i p_i(x',a)."""
    p = softmax(next_logits, axis=-1)               # (B, A, N)
    q = (p * z).sum(-1)                             # (B, A)
    return q.argmax(-1)                            # the action defining T Z

def loss(online_logits_chosen, target_dist):
    """Bellman update as multiclass classification: cross-entropy of
    KL( Phi T-hat Z_target || Z_theta ).  The projected operator is a
    sqrt(gamma)-contraction in the Cramer distance."""
    logp = log_softmax(online_logits_chosen, axis=-1)   # (B, N)
    return -(target_dist * logp).sum(-1).mean()

def softmax(x, axis):     e = np.exp(x - x.max(axis, keepdims=True)); return e / e.sum(axis, keepdims=True)
def log_softmax(x, axis): return x - x.max(axis, keepdims=True) - np.log(np.exp(x - x.max(axis, keepdims=True)).sum(axis, keepdims=True))
```

The causal chain, end to end: I refused to throw away the return's distribution, wrote the Bellman recursion $Z\overset{D}{=}R+\gamma Z'$ directly on it, and — borrowing the contraction method from the analysis of recursive algorithms — proved that in the maximal Wasserstein metric $\overline d_p$ the policy-evaluation operator $\mathcal{T}^\pi$ is a $\gamma$-contraction, so distributional evaluation has a unique fixed point $Z^\pi$ reached geometrically; the metric had to be Wasserstein because the discount is a horizontal transport that likelihood metrics can't see. The variance component propagates at $\gamma^2$ once the mean-dependent transition term is fixed. Control broke the clean picture: the mean still contracts, but greedy selection is discontinuous in the distribution, so the optimality operator contracts in *no* distribution metric, may have no fixed point, and at best limps toward the nonstationary optimal set — which is itself the argument for averaging over greedy chatter rather than committing to a stationary policy. To run any of this I had to represent distributions on a fixed atom grid and abandon the Wasserstein loss (unminimizable from samples), projecting the backup onto the grid and minimizing a cross-entropy — a move that is principled because the projected operator is a $\sqrt{\gamma}$-contraction in the Cramér distance, leaving the Bellman update as a multiclass classification trainable by ordinary SGD.
