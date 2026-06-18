OK, let me think this through from scratch. I have a stochastic policy $\pi_\theta(a\mid s)$, a neural net, and I want to make it good at a long-horizon control task by climbing the expected total reward $\mathbb{E}[\sum_{t=0}^\infty r_t]$ directly. Why direct? Because the alternative — fit a $Q$-function and act greedily — bakes in all the pathologies of value-based methods with function approximation, and because I want to drop in an arbitrary neural net without worrying about whether a greedy argmax over it is even well-behaved. So I'll do gradient ascent on the reward.

The likelihood-ratio trick gives me a gradient I can actually estimate from samples. The expected return is $J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]$, and since the only $\theta$-dependence in the trajectory distribution is the product of $\pi_\theta(a_t\mid s_t)$ terms, $\nabla_\theta \log p_\theta(\tau)=\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)$ (the dynamics don't depend on $\theta$, so they drop out of the gradient of the log). That gives
$$
g=\nabla_\theta J=\mathbb{E}\Big[\sum_{t=0}^\infty \Psi_t\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\Big],
$$
where, at the crudest, $\Psi_t=R(\tau)=\sum_{t'=0}^\infty r_{t'}$, the whole trajectory's reward sitting in front of every term. So the gradient is some weighting $\Psi_t$ times the score of each action. The whole game is choosing $\Psi_t$.

Now, the trajectory return as $\Psi_t$ is obviously terrible. Each action $a_t$ gets multiplied by the *entire* return, including rewards that happened *before* $a_t$ was taken. Those earlier rewards are not consequences of $a_t$ at all — they're pure noise as far as $a_t$'s credit is concerned. Causality says the reward at time $t'<t$ can't depend on $a_t$, so in expectation $\mathbb{E}[r_{t'}\nabla_\theta\log\pi(a_t\mid s_t)]=0$ for $t'<t$; those terms contribute nothing to the gradient but plenty to the variance. So I can drop them and use the reward-to-go $\Psi_t=\sum_{t'\ge t} r_{t'}$ without changing the expectation. Good, that's free variance reduction.

But it's still bad. The reward-to-go after $a_t$ contains the effects of all the *later* actions $a_{t+1},a_{t+2},\dots$ and all the environment noise from there on. So the multiplier on $a_t$'s score is dominated by stuff $a_t$ didn't cause. Over a long horizon, this variance piles up — roughly, every future action injects its own noise into $a_t$'s credit. This is the credit assignment problem showing up as estimator variance, and it's why naive policy gradients need absurd numbers of samples.

Can I subtract something to calm it down? Here's the one structural fact I get for free: I can subtract any function of the state alone, $b(s_t)$, from $\Psi_t$ without biasing the gradient. Check it: the offending term is $\mathbb{E}[\nabla_\theta\log\pi(a_t\mid s_t)\,b(s_t)]$, and conditioning on $s_t$,
$$
\mathbb{E}_{a_t\sim\pi}[\nabla_\theta\log\pi(a_t\mid s_t)]=\int \pi(a\mid s_t)\,\frac{\nabla_\theta\pi(a\mid s_t)}{\pi(a\mid s_t)}\,da=\nabla_\theta\int\pi(a\mid s_t)\,da=\nabla_\theta 1=0.
$$
So $\mathbb{E}[\nabla_\theta\log\pi(a_t\mid s_t)\,b(s_t)]=\mathbb{E}_{s_t}[b(s_t)\cdot 0]=0$. A state-dependent baseline is invisible to the expectation but can slash the variance, because it removes the part of the return that's predictable from $s_t$ alone — the part that has nothing to do with which action I chose.

What's the best $b$? Intuitively I want $\Psi_t$ to be positive exactly when $a_t$ was a better-than-typical choice from $s_t$, and negative when it was worse, so the update pushes probability toward good actions and away from bad ones. "Better than typical from $s_t$" is exactly $Q^\pi(s_t,a_t)-V^\pi(s_t)$, where $V^\pi(s_t)=\mathbb{E}_{a\sim\pi}[Q^\pi(s_t,a)]$ is the average value of the state. So the natural choice is the advantage
$$
A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t),
$$
i.e. baseline $b=V^\pi$. This centers the multiplier at zero on average, which is about as low-variance as I can make it while staying unbiased — the score gets scaled by *how much better than average* the action was, not by the raw return. So if I could use $\Psi_t=A^\pi(s_t,a_t)$, I'd be in great shape.

I can't, of course. I don't know $A^\pi$. I have at best a learned, imperfect value function $V$. So the real problem is: estimate the advantage from samples and an imperfect $V$, with low variance, without poisoning the gradient with bias. And I'll deliberately work with the *state*-value $V(s)$ rather than a learned action-value $Q(s,a)$: $V$ takes only the state as input, so it lives in a lower-dimensional space and is easier to fit accurately than a $Q$ over the joint state-action space — and, as I'll see, building the estimator around $V$ is exactly what gives me a knob to slide between high-bias and low-bias estimators, whereas a parameterized $Q$ would pin me to a single one-step form.

Let me stop and think about bias versus variance, because they are not symmetric here and that asymmetry should drive the design. High variance is annoying but recoverable: it just means I need more samples or smaller steps; with enough data it washes out. Bias is worse — a biased gradient can point the optimizer at a policy that isn't even a local optimum of my actual objective, and no amount of data fixes that. So I want to be greedy about cutting variance but stingy about accepting bias, and I want a *knob* so I can choose where on that tradeoff to sit, rather than committing to one estimator.

Before I get to the advantage estimator, there's a separate variance lever I should think about: the horizon itself. The far-future rewards are where the variance is worst — the noise from a hundred future actions. What if I just downweight them? Introduce a discount $\gamma<1$ and weight $r_{t+l}$ by $\gamma^l$. Note I'm *not* changing the problem — the goal is still the undiscounted $\sum_t r_t$. I'm using $\gamma$ purely as an estimator-side variance-reduction parameter. The discounted value functions are
$$
V^{\pi,\gamma}(s_t)=\mathbb{E}_{a_t,s_{t+1},\dots}\Big[\sum_{l=0}^\infty \gamma^l r_{t+l}\Big],\quad Q^{\pi,\gamma}(s_t,a_t)=\mathbb{E}_{s_{t+1},\dots}\Big[\sum_{l=0}^\infty \gamma^l r_{t+l}\Big],\quad A^{\pi,\gamma}=Q^{\pi,\gamma}-V^{\pi,\gamma},
$$
and I'll aim my estimator at the discounted advantage $A^{\pi,\gamma}$, accepting whatever bias $\gamma<1$ costs against the true undiscounted gradient. Why is that bias acceptable? Because the credit for an action usually decays with delay — if an action's effect on reward is essentially forgotten after some number of steps, then dropping the heavily-delayed terms barely changes the gradient but kills a lot of variance. I'll make this precise later; for now $\gamma$ is my "how far into the future do I bother attributing credit" knob.

So the target is $A^{\pi,\gamma}(s_t,a_t)$, and I want estimators of it built from my imperfect $V$. Where do I start? The most local thing I can compute is the one-step temporal-difference residual,
$$
\delta_t^V=r_t+\gamma V(s_{t+1})-V(s_t).
$$
Stare at it. If $V$ happened to be exactly $V^{\pi,\gamma}$, what's its expectation given $s_t,a_t$? Taking the expectation over $s_{t+1}$,
$$
\mathbb{E}_{s_{t+1}}[\delta_t^{V^{\pi,\gamma}}]=\mathbb{E}_{s_{t+1}}[r_t+\gamma V^{\pi,\gamma}(s_{t+1})]-V^{\pi,\gamma}(s_t)=Q^{\pi,\gamma}(s_t,a_t)-V^{\pi,\gamma}(s_t)=A^{\pi,\gamma}(s_t,a_t),
$$
where the middle step is just the definition of $Q^{\pi,\gamma}$ as the expected immediate reward plus the discounted value of the next state. So $\delta_t$ is an *unbiased* one-step advantage estimate — but only when $V$ is exact. With a real, imperfect $V$, $\delta_t$ is biased: it inherits all the error in $V(s_t)$ and $V(s_{t+1})$, just one step deep. Its virtue is low variance — it only looks one step ahead, so almost none of the future noise leaks in. This is the high-bias/low-variance end.

What's at the other end? If I trust $V$ not at all and just use the empirical return, $\Psi_t=\sum_{l\ge0}\gamma^l r_{t+l}-V(s_t)$, that's unbiased (for the discounted advantage) regardless of $V$ — the $V(s_t)$ is just a baseline, which I proved can't bias anything — but it carries the full horizon's worth of variance. High variance, low bias.

So I have the two endpoints, and I want everything in between. Let me build the intermediate estimators explicitly by summing TD residuals. Take two:
$$
\delta_t+\gamma\delta_{t+1}=\big(r_t+\gamma V(s_{t+1})-V(s_t)\big)+\gamma\big(r_{t+1}+\gamma V(s_{t+2})-V(s_{t+1})\big).
$$
The $+\gamma V(s_{t+1})$ from the first term cancels the $-\gamma V(s_{t+1})$ from the second. Telescoping. What survives is
$$
\delta_t+\gamma\delta_{t+1}=-V(s_t)+r_t+\gamma r_{t+1}+\gamma^2 V(s_{t+2}).
$$
And in general, summing $k$ of them with the $\gamma^l$ weights, every interior $V$ cancels against its neighbor and I'm left with
$$
\hat A_t^{(k)}\;\equiv\;\sum_{l=0}^{k-1}\gamma^l \delta_{t+l}=-V(s_t)+r_t+\gamma r_{t+1}+\dots+\gamma^{k-1}r_{t+k-1}+\gamma^k V(s_{t+k}).
$$
That's a clean reading: $\hat A_t^{(k)}$ is the $k$-step return — $k$ real rewards followed by a bootstrap $\gamma^k V(s_{t+k})$ off the value function — minus the baseline $V(s_t)$. So $k$ literally interpolates between my two endpoints. At $k=1$ it's $\delta_t$. As $k\to\infty$, the bootstrap term $\gamma^k V(s_{t+k})$ is crushed by $\gamma^k\to0$, so its future-tail contribution vanishes, and I get $-V(s_t)+\sum_{l\ge0}\gamma^l r_{t+l}$, the Monte-Carlo return minus baseline. I have to be precise about "bias" here. As a point estimate of $A^{\pi,\gamma}$, an imperfect $V(s_t)$ can still shift the conditional mean, because the true advantage subtracts $V^{\pi,\gamma}(s_t)$. But in the policy-gradient estimator that $-V(s_t)$ is a state-only baseline, so it contributes zero after multiplying by $\nabla\log\pi(a_t\mid s_t)$ and averaging over $a_t$. The policy-gradient bias introduced by bootstrapping is in the future tail; larger $k$ discounts that tail more heavily, at the price of more variance. There's my spectrum, parameterized by an integer $k$.

But an integer knob is clumsy — I'd have to commit to one horizon $k$ for the bootstrap, and that's a hard, discrete bias/variance choice. I'd rather have a soft knob that blends all the $k$ together. And there's a well-known way to blend $n$-step estimators smoothly: weight them geometrically, the way the $\lambda$-return does for value estimation. There, instead of picking one $n$-step return you take $(1-\lambda)\sum_{k\ge1}\lambda^{k-1}(\text{$k$-step return})$, a single parameter $\lambda$ sliding from one-step ($\lambda=0$) to Monte-Carlo ($\lambda=1$). The exact same construction should work here, except I'm averaging $k$-step *advantage* estimators, not value targets. Let me just do it:
$$
\hat A_t^{\text{GAE}(\gamma,\lambda)}\;\equiv\;(1-\lambda)\big(\hat A_t^{(1)}+\lambda\,\hat A_t^{(2)}+\lambda^2\,\hat A_t^{(3)}+\dots\big).
$$
The $(1-\lambda)$ in front is the normalizer: the weights on the successive $k$-step estimators are $(1-\lambda)\lambda^{k-1}$, and $\sum_{k\ge1}(1-\lambda)\lambda^{k-1}=(1-\lambda)\cdot\frac{1}{1-\lambda}=1$, so I'm taking a genuine weighted average (a convex combination) of the $\hat A^{(k)}$, not just an arbitrary sum. Good — that keeps the scale right.

Now let me see if this collapses to something simple. Substitute $\hat A_t^{(k)}=\sum_{l=0}^{k-1}\gamma^l\delta_{t+l}$ and collect by which $\delta$ appears:
$$
\hat A_t^{\text{GAE}(\gamma,\lambda)}=(1-\lambda)\Big(\delta_t+\lambda(\delta_t+\gamma\delta_{t+1})+\lambda^2(\delta_t+\gamma\delta_{t+1}+\gamma^2\delta_{t+2})+\dots\Big).
$$
Group the terms by $\delta_{t+l}$. The residual $\delta_t$ appears in *every* bracket (it's in $\hat A^{(1)},\hat A^{(2)},\dots$), so it picks up $1+\lambda+\lambda^2+\dots=\frac{1}{1-\lambda}$. The residual $\gamma\delta_{t+1}$ first appears in $\hat A^{(2)}$ and in every bracket after, so it picks up $\lambda+\lambda^2+\lambda^3+\dots=\frac{\lambda}{1-\lambda}$. In general $\gamma^l\delta_{t+l}$ first shows up in $\hat A^{(l+1)}$, so its coefficient is $\lambda^l+\lambda^{l+1}+\dots=\frac{\lambda^l}{1-\lambda}$. Putting it together,
$$
\hat A_t^{\text{GAE}(\gamma,\lambda)}=(1-\lambda)\Big(\delta_t\tfrac{1}{1-\lambda}+\gamma\delta_{t+1}\tfrac{\lambda}{1-\lambda}+\gamma^2\delta_{t+2}\tfrac{\lambda^2}{1-\lambda}+\dots\Big),
$$
and the $(1-\lambda)$ cancels every denominator, leaving
$$
\boxed{\;\hat A_t^{\text{GAE}(\gamma,\lambda)}=\sum_{l=0}^\infty (\gamma\lambda)^l\,\delta_{t+l}.\;}
$$
Huh. That's almost suspiciously clean. The entire exponentially-weighted blend of every $k$-step advantage estimator is just a discounted sum of the *same* one-step TD residuals, with discount $\gamma\lambda$ instead of $\gamma$. All the messy averaging collapsed into a single geometric series over $\delta$'s. So $\lambda$ doesn't change *what* I sum (TD residuals), it changes the *rate* at which I discount them: the effective discount is the product $\gamma\lambda$.

Let me sanity-check the two limits against my endpoints. At $\lambda=0$: $(\gamma\lambda)^l=0$ for $l\ge1$ and $1$ for $l=0$, so $\hat A_t=\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$. That's the one-step TD estimate — the low-variance, high-bias end. At $\lambda=1$: $\hat A_t=\sum_{l\ge0}\gamma^l\delta_{t+l}$, which by the telescoping identity above (take $k\to\infty$) equals $\sum_{l\ge0}\gamma^l r_{t+l}-V(s_t)$ — the Monte-Carlo return minus baseline, the low-bias, high-variance end. So $\lambda$ slides exactly between them, and any $0<\lambda<1$ is a compromise. The knob I wanted, and it's a one-liner to compute.

Now I should be careful about a claim I keep making loosely — "unbiased." Unbiased *for what*? My honest target is the discounted policy gradient
$$
g^\gamma=\mathbb{E}\Big[\sum_t A^{\pi,\gamma}(s_t,a_t)\,\nabla_\theta\log\pi(a_t\mid s_t)\Big],
$$
which is itself already a biased proxy for the true undiscounted gradient (that bias I knowingly accepted with $\gamma$). The question for $\lambda$ is whether plugging $\hat A_t$ in place of $A^{\pi,\gamma}$ keeps me unbiased *for $g^\gamma$*. Let me define the property I care about: call $\hat A_t$ "$\gamma$-just" if
$$
\mathbb{E}\big[\hat A_t\,\nabla_\theta\log\pi(a_t\mid s_t)\big]=\mathbb{E}\big[A^{\pi,\gamma}(s_t,a_t)\,\nabla_\theta\log\pi(a_t\mid s_t)\big],
$$
so that if every $\hat A_t$ is $\gamma$-just, summing over $t$ recovers $g^\gamma$ exactly. When is my estimator $\gamma$-just?

Here's a sufficient condition I can prove. Suppose $\hat A_t$ splits as $\hat A_t = Q_t(s_{t:\infty},a_{t:\infty}) - b_t(s_{0:t},a_{0:t-1})$, where $Q_t$ can look at the whole future and is an *unbiased estimate of $Q^{\pi,\gamma}(s_t,a_t)$* in the sense $\mathbb{E}[Q_t\mid s_t,a_t]=Q^{\pi,\gamma}(s_t,a_t)$, and $b_t$ is *any* function of things sampled strictly before $a_t$. Claim: then $\hat A_t$ is $\gamma$-just. Let me prove it by handling the two pieces separately.

The $Q$-piece. I want $\mathbb{E}[\nabla_\theta\log\pi(a_t\mid s_t)\,Q_t]$. Condition on everything up to and including $a_t$, i.e. write the expectation as outer over $(s_{0:t},a_{0:t})$ and inner over the future $(s_{t+1:\infty},a_{t+1:\infty})$:
$$
\mathbb{E}[\nabla\log\pi(a_t\mid s_t)\,Q_t]
=\mathbb{E}_{s_{0:t},a_{0:t}}\Big[\nabla\log\pi(a_t\mid s_t)\;\mathbb{E}_{\text{future}}[Q_t\mid s_t,a_t]\Big],
$$
where I could pull $\nabla\log\pi(a_t\mid s_t)$ out of the inner expectation because it depends only on $(s_t,a_t)$, which are fixed by the outer conditioning. The inner expectation is exactly $Q^{\pi,\gamma}(s_t,a_t)$ by assumption. So this becomes $\mathbb{E}_{s_{0:t},a_{0:t}}[\nabla\log\pi(a_t\mid s_t)\,Q^{\pi,\gamma}(s_t,a_t)]$, and since the summand now depends only on $(s_t,a_t)$, that's $\mathbb{E}[\nabla\log\pi(a_t\mid s_t)\,Q^{\pi,\gamma}(s_t,a_t)]$.

The $b$-piece. I want $\mathbb{E}[\nabla_\theta\log\pi(a_t\mid s_t)\,b_t]$ to vanish. Now $b_t$ depends only on $(s_{0:t},a_{0:t-1})$ — everything before $a_t$ — so I condition the other way: take the outer expectation over $(s_{0:t},a_{0:t-1})$ and the inner expectation over $a_t$ and beyond. Since $b_t$ is determined by the outer variables, pull it out:
$$
\mathbb{E}[\nabla\log\pi(a_t\mid s_t)\,b_t]=\mathbb{E}_{s_{0:t},a_{0:t-1}}\Big[b_t\;\mathbb{E}_{a_t,\dots}[\nabla\log\pi(a_t\mid s_t)]\Big].
$$
But the inner expectation is the same one I computed at the very start: $\mathbb{E}_{a_t\sim\pi}[\nabla_\theta\log\pi(a_t\mid s_t)]=\nabla_\theta\!\int\pi\,da=0$. So the whole $b$-piece is $\mathbb{E}[b_t\cdot 0]=0$. Subtracting an arbitrary past-measurable baseline costs nothing. Done: $\hat A_t = Q_t - b_t$ is $\gamma$-just.

Now apply this to my estimators. The Monte-Carlo-flavored ones — the empirical $\gamma$-discounted return $\sum_l\gamma^l r_{t+l}$ (which has $\mathbb{E}[\cdot\mid s_t,a_t]=Q^{\pi,\gamma}$), and that return minus any $V(s_t)$ baseline — are $\gamma$-just for *any* $V$, because they're literally of the form (unbiased $Q$ estimate) minus (past-measurable baseline). That's exactly $\hat A^{\text{GAE}(\gamma,1)}$: it's $\gamma$-just regardless of how bad $V$ is. The price is variance. On the other hand $\hat A^{\text{GAE}(\gamma,0)}=\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$ is only $\gamma$-just when $V=V^{\pi,\gamma}$ (that's the calculation I did showing $\mathbb{E}[\delta]=A^{\pi,\gamma}$ in that case); for an imperfect $V$ it's biased, because $r_t+\gamma V(s_{t+1})$ is *not* an unbiased estimate of $Q^{\pi,\gamma}(s_t,a_t)$ unless $V$ is right. So the $\lambda$ knob is literally interpolating between "$\gamma$-just for any $V$, high variance" and "biased unless $V$ is exact, low variance." That's the precise statement of the tradeoff, and it tells me $\lambda<1$ only hurts to the extent $V$ is wrong.

This is a good moment to disentangle $\gamma$ and $\lambda$, because they look similar — both enter as discount-like factors, and $\gamma\lambda$ even multiplies them together — but they are doing genuinely different jobs and I shouldn't conflate them. $\gamma$ sets the scale of the value function $V^{\pi,\gamma}$ itself and, more importantly, $\gamma<1$ biases the policy gradient *even if my value function is perfect* — because it changes the very thing I'm estimating, $A^{\pi,\gamma}$ instead of $A^{\pi,1}$, regardless of $V$. By contrast $\lambda<1$ introduces bias *only* through the inaccuracy of $V$: with a perfect $V$, every $\lambda$ is $\gamma$-just. So they're not redundant, and I'd expect to want them at different settings — a fairly large $\gamma$ (I do want to attribute credit reasonably far out) but a smaller $\lambda$ (it buys variance reduction at the cost of only the $V$-error-mediated bias, which is mild for a decent $V$). One number couldn't express that; I need both.

Let me make the role of $\gamma$ precise, because I waved at "credit decays with delay" earlier and I want to know exactly what approximation I'm making. Define the response function
$$
\chi(l;s_t,a_t)=\mathbb{E}[r_{t+l}\mid s_t,a_t]-\mathbb{E}[r_{t+l}\mid s_t],
$$
the extra expected reward $l$ steps later attributable to having taken $a_t$ (over the policy's default at $s_t$). It cleanly decomposes the advantage by delay:
$$
A^{\pi,\gamma}(s_t,a_t)=\sum_{l=0}^\infty \gamma^l\,\chi(l;s_t,a_t),
$$
which I get by writing $A^{\pi,\gamma}$ as the discounted sum of (reward given the action) minus (reward not conditioning on the action), term by term. So $\chi$ quantifies the temporal credit assignment problem directly: long-range dependence between an action and its rewards means $\chi(l;\cdot)$ is nonzero for $l\gg0$. Now look at the gradient term $\nabla\log\pi(a_t\mid s_t)\,A^{\pi,\gamma}=\nabla\log\pi\cdot\sum_l\gamma^l\chi(l)$. Using $\gamma<1$ effectively drops the $\chi$-terms with $l\gg1/(1-\gamma)$ (since $\gamma^l$ is tiny there). So the bias from $\gamma$ is small precisely when the response decays within $\approx1/(1-\gamma)$ steps — when an action's influence is "forgotten" on that timescale. That's the precise version of my earlier hand-wave, and it tells me how to read $\gamma$: it's a credit-assignment horizon.

There's a second, prettier way to see the whole $\hat A^{\text{GAE}}$ formula that also explains *why* using $V$ as a baseline helps with the response function, not just with raw variance. It comes from reward shaping. Take any potential $\Phi:\mathcal S\to\mathbb R$ and transform the reward to
$$
\tilde r(s,a,s')=r(s,a,s')+\gamma\Phi(s')-\Phi(s).
$$
What does this do to a discounted trajectory sum? It telescopes:
$$
\sum_{l=0}^\infty\gamma^l\tilde r(s_{t+l},a_{t+l},s_{t+l+1})=\sum_{l=0}^\infty\gamma^l r(s_{t+l},\dots)+\sum_{l=0}^\infty\gamma^l\big(\gamma\Phi(s_{t+l+1})-\Phi(s_{t+l})\big),
$$
and the second sum is $\sum_l(\gamma^{l+1}\Phi(s_{t+l+1})-\gamma^l\Phi(s_{t+l}))=-\Phi(s_t)$ (everything cancels except the very first $-\Phi(s_t)$). So shaping changes the discounted return only by the constant $-\Phi(s_t)$, which means $\tilde Q^{\pi,\gamma}=Q^{\pi,\gamma}-\Phi$, $\tilde V^{\pi,\gamma}=V^{\pi,\gamma}-\Phi$, and therefore $\tilde A^{\pi,\gamma}=A^{\pi,\gamma}$ — shaping leaves the advantage *completely unchanged*. That's a strong invariance: I can rewrite the rewards however I like via a potential and not move the advantage I'm trying to estimate.

Now choose $\Phi=V$, my value function. Then the shaped reward is
$$
\tilde r=r+\gamma V(s')-V(s)=\delta^V,
$$
the TD residual itself. So the residuals I've been summing *are* the shaped rewards under $\Phi=V$. And what is $\hat A^{\text{GAE}(\gamma,\lambda)}=\sum_l(\gamma\lambda)^l\delta_{t+l}$ in this light? It's the $\gamma\lambda$-discounted sum of shaped rewards — the return of the shaped MDP, but with a *steeper* discount $\gamma\lambda$ rather than $\gamma$. So the recipe reads cleanly: reshape the reward with $\Phi=V$ to absorb value into the reward, then apply an extra discount $\lambda$ on top of $\gamma$.

Why is that the right thing to do? Because of what shaping with $V$ does to the response function. Suppose $\Phi=V^{\pi,\gamma}$ exactly. Then for the shaped reward, $\mathbb{E}[\tilde r_{t+l}\mid s_t,a_t]=\mathbb{E}[\tilde r_{t+l}\mid s_t]=0$ for all $l>0$ — the shaped response function is zero everywhere except $l=0$. Intuitively, a perfect value baseline reports all of an action's future consequence *immediately*, as the one-step residual, leaving nothing for later steps. So perfect shaping converts temporally-spread credit into a single immediate signal. With an imperfect $V\approx V^{\pi,\gamma}$, it doesn't fully collapse, but it *shrinks* the temporal spread of the response. And once the response is concentrated near $l=0$, I can afford a steeper discount $\gamma\lambda$: it cuts off the residual terms $\nabla\log\pi(a_t\mid s_t)\,\delta_{t+l}$ for $l\gg1/(1-\gamma\lambda)$, throwing away the long-delay noise that, after shaping, carries almost no signal anyway. So $V$-shaping and the $\lambda$-discount work as a team: shaping concentrates the signal, then $\lambda$ trims the now-mostly-noise tail. That's the deeper "why" behind the estimator, and it's why $\lambda$ can be aggressive (small) without much bias as long as $V$ is decent.

Good. I'm confident in the estimator. Now the practical apparatus around it.

First, I need a value function $V$, and I need to fit it. The obvious target is the Monte-Carlo discounted return $\hat V_t=\sum_{l\ge0}\gamma^l r_{t+l}$, and I just regress $V_\phi$ onto it by least squares, $\min_\phi\sum_n\|V_\phi(s_n)-\hat V_n\|^2$. (I could instead use a TD($\lambda$)-style target $V_{\phi_\text{old}}(s_n)+\sum_l(\gamma\lambda)^l\delta_{t+l}$, mirroring the advantage estimator — but in practice it doesn't beat the simple $\lambda=1$ Monte-Carlo target, so I'll keep the simpler one.) But there's a danger with a powerful neural-net $V$ in this loop: if I let the regression overfit the current batch, the fitted residuals $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$ collapse toward zero — and then my whole advantage signal, which is built from those residuals, vanishes and the policy gradient goes to zero. So I must not overfit $V$ to the latest batch.

The fix is to limit how much $V$ is allowed to move per iteration — a trust region on the value function. Interpret $V_\phi(s)$ as the mean of a Gaussian with fixed variance $\sigma^2$ (where $\sigma^2=\frac1N\sum_n\|V_{\phi_\text{old}}(s_n)-\hat V_n\|^2$ from before the update); then constraining the average KL between old and new value "distributions" is the same as constraining $\frac1N\sum_n\frac{\|V_\phi(s_n)-V_{\phi_\text{old}}(s_n)\|^2}{2\sigma^2}\le\epsilon$. So I solve
$$
\min_\phi\sum_n\|V_\phi(s_n)-\hat V_n\|^2\quad\text{s.t.}\quad \frac1N\sum_n\frac{\|V_\phi(s_n)-V_{\phi_\text{old}}(s_n)\|^2}{2\sigma^2}\le\epsilon.
$$
This keeps the value function from lurching to fit noise in the current batch, which is exactly the overfitting that would zero out my residuals. To solve it for a $\sim10^4$-parameter net I linearize the objective and quadratize the constraint, getting a QP $\min_\phi g^\top(\phi-\phi_\text{old})$ s.t. $\frac1N\sum_n(\phi-\phi_\text{old})^\top H(\phi-\phi_\text{old})\le\epsilon$, with $H=\frac1N\sum_n j_n j_n^\top$, $j_n=\nabla_\phi V_\phi(s_n)$ — the Gauss-Newton approximation to the Hessian, which is also the Fisher information (up to the $\sigma^2$) under the Gaussian reading. I don't form $H$; I run conjugate gradient using only matrix-vector products $v\mapsto Hv$ to get a step $s\approx-H^{-1}g$, then rescale $s\to\alpha s$ so that $\frac12(\alpha s)^\top H(\alpha s)=\epsilon$ and step to $\phi_\text{old}+\alpha s$.

Second, the policy update. I have low-variance, low-(enough)-bias advantage estimates $\hat A_t$; I could just do SGD on $\sum_t\hat A_t\nabla\log\pi$. But the data is nonstationary — every policy step changes the distribution I'm sampling from — and a step that's too big collapses the policy. So I want the largest step that still keeps the new policy close to the old one in distribution. That's a KL trust region on the policy:
$$
\max_\theta\ \frac1N\sum_n\frac{\pi_\theta(a_n\mid s_n)}{\pi_{\theta_\text{old}}(a_n\mid s_n)}\hat A_n\quad\text{s.t.}\quad \frac1N\sum_n D_{\mathrm{KL}}\!\big(\pi_{\theta_\text{old}}(\cdot\mid s_n)\,\|\,\pi_\theta(\cdot\mid s_n)\big)\le\epsilon.
$$
Same machinery as for the value function: linearize the (importance-weighted) objective, quadratize the KL, and the ascent step for this maximization is $\theta-\theta_\text{old}\propto F^{-1}g$ with $F$ the average Fisher information and $g$ the policy gradient estimate. If I hand the optimizer a loss $\ell(\theta)=-\frac1N\sum_n\frac{\pi_\theta(a_n\mid s_n)}{\pi_{\theta_\text{old}}(a_n\mid s_n)}\hat A_n$, then the same update is a descent step $-F^{-1}\nabla\ell$. That is exactly the sign convention implementation code usually uses. The direction $F^{-1}g$ is the natural policy gradient; it falls out here not because I went looking for it but because the KL constraint *is* the Fisher metric to second order. Reparameterization-invariant steps, for free, as a consequence of "stay close in distribution."

One ordering subtlety bites here, and it's the same overfitting trap in a different guise. In each iteration I have both a policy update and a value-function update; which first? I must compute the advantages (hence the policy step) using the *old* value function $V_{\phi_i}$, and update $V$ *after*. If I fit $V$ first and then computed advantages with the freshly-fitted $V_{\phi_{i+1}}$, I'd be using a $V$ that was just trained to predict the returns in this very batch — pushing $\delta_t\to0$ on exactly these samples — and the policy gradient would be biased toward zero (in the extreme, an overfit $V$ makes every residual zero and the gradient vanishes). So: advantages and policy step with the old $V$, then refit $V$.

Putting the loop together: simulate the current policy for a batch of timesteps; compute $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$ everywhere with the current $V$; compute $\hat A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$; take the TRPO policy step using those advantages; then refit $V$ with the trust-region value update; repeat.

Now the code. The advantage computation is the centerpiece, and there's a subtlety worth noting: I don't actually sum the infinite series naively per timestep — that'd be $O(T^2)$. The series $\hat A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$ obeys a one-line backward recursion, because $\hat A_t=\delta_t+\gamma\lambda\,\hat A_{t+1}$ (pull out the $l=0$ term and reindex). So I sweep backward through each episode accumulating $\delta_t+\gamma\lambda\cdot(\text{running})$, which is $O(T)$. At an episode boundary the recursion resets. Separately, the value function is fit to the Monte-Carlo discounted return $\sum_l\gamma^l r_{t+l}$, not to $\hat A_t+V(s_t)$ unless I deliberately choose the TD($\lambda$)-style target that the paper says was tried and did not improve on the $\lambda=1$ target.

```python
import numpy as np


def discount_cumsum(x, discount):
    """Backward discounted cumulative sum:
       out[t] = x[t] + discount*x[t+1] + discount^2*x[t+2] + ...
    Used twice: with discount=gamma*lambda on the TD residuals to get the
    advantage Sum_l (gamma*lambda)^l delta_{t+l}, and with discount=gamma on the
    rewards to get the Monte-Carlo value targets."""
    out = np.zeros_like(x, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(x))):
        running = x[t] + discount * running
        out[t] = running
    return out


def compute_gae(rewards, values, last_value, gamma, lam):
    """Generalized advantage estimation for one episode.

    rewards : r_0 .. r_{T-1}
    values  : V(s_0) .. V(s_{T-1})  (current value function)
    last_value : V(s_T), the bootstrap; 0 if s_T is terminal
    Returns the advantages A_t and Monte-Carlo discounted value targets.
    """
    # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)   (the TD residual / shaped reward)
    vals_next = np.append(values[1:], last_value)
    deltas = rewards + gamma * vals_next - values
    # A_t = sum_l (gamma*lambda)^l delta_{t+l}, via the backward recursion
    #       A_t = delta_t + gamma*lambda * A_{t+1}.
    advantages = discount_cumsum(deltas, gamma * lam)
    # Value-function target used in the paper/rllab: discounted return.
    returns = discount_cumsum(np.append(rewards, last_value), gamma)[:-1]
    return advantages, returns
```

The equivalent batched, in-place form people use when episodes are laid out in arrays with done-flags is the same recursion written as a single backward pass with a mask — `lastgaelam` carries $\hat A_{t+1}$ and the mask zeroes it at terminals:

```python
def compute_gae_batch(rewards, values, dones, last_value, gamma, lam):
    """rewards/values/dones are arrays over a rollout possibly spanning several
    episodes; dones[t]=1 if s_{t+1} is terminal (don't bootstrap across it)."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    running_return = last_value
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
        running_return = rewards[t] + gamma * nonterminal * running_return
        returns[t] = running_return
    return adv, returns
```

And the outer loop, tying each piece to the reasoning:

```python
def train(env, policy, value_fn, n_iters, batch_steps, gamma, lam, kl_policy, kl_vf):
    for _ in range(n_iters):
        # 1. roll out the CURRENT policy; record s, a, r and V(s) with the
        #    current value function (advantages must use the OLD V_phi_i).
        batch = collect_batch(env, policy, value_fn, batch_steps)

        # 2. per episode: TD residuals -> GAE advantages; rewards -> MC value targets
        advs, rets = [], []
        for ep in batch.episodes:
            last_v = 0.0 if ep.terminated else value_fn.predict(ep.last_state)
            a, r = compute_gae(ep.rewards, ep.values, last_v, gamma, lam)
            advs.append(a); rets.append(r)
        advs = normalize(np.concatenate(advs))   # standardize for step-size stability
        rets = np.concatenate(rets)

        # 3. policy step: KL-trust-region (TRPO) on the importance-weighted
        #    objective with these advantages; ascent direction ~ F^{-1} g
        #    (or descent on the negative surrogate), solved by CG/FVP + line search.
        trpo_step(policy, batch.states, batch.actions, advs, kl_policy)

        # 4. THEN refit the value function under its own trust region, so the
        #    next iteration's residuals don't get prematurely zeroed.
        value_fn.trust_region_fit(batch.states, rets, kl_vf)
```

So the causal chain, start to finish: I want to ascend expected reward, so I use the score-function gradient with a multiplier $\Psi_t$; the empirical return makes $\Psi_t$ unbearably noisy because each action is credited with everyone else's rewards, so I subtract a state baseline $V(s_t)$ — free, because state baselines don't bias the gradient — landing on the advantage as the ideal $\Psi_t$; not knowing the advantage, I estimate it from TD residuals $\delta_t$, which telescope into $k$-step returns-minus-baseline, where larger $k$ trades variance for less bias; rather than pick one $k$ I geometrically average them, which collapses to the single discounted sum $\hat A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$; $\lambda$ slides from one-step TD ($\lambda=0$, biased unless $V$ is exact, low variance) to Monte-Carlo ($\lambda=1$, unbiased for any $V$, high variance); reading $\delta=\tilde r$ as a $V$-shaped reward shows the estimator is the $\gamma\lambda$-discounted return of a shaped MDP — shaping with $V$ concentrates credit at $l=0$ and $\lambda$ then trims the long-delay noise; $\gamma$ and $\lambda$ are kept separate because $\gamma$ biases even with a perfect $V$ while $\lambda$ biases only through $V$'s error; the residual recursion $\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$ makes it an $O(T)$ backward pass; the policy is updated inside a KL trust region (yielding the natural-gradient direction) and the value function inside its own KL trust region to stop it overfitting and zeroing the residuals, with the policy stepped on the *old* $V$ to avoid that same bias.
