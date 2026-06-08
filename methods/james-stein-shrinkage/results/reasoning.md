Let me set the problem down as plainly as I can. I see one Gaussian vector $X=(X_1,\dots,X_p)$, the coordinates independent, $X_i\sim N(\theta_i,1)$, and I want to estimate the whole mean vector $\theta$ at once, graded by the total squared error $\|\hat\theta-\theta\|^2=\sum_i(\hat\theta_i-\theta_i)^2$. The coordinates are unrelated — coordinate one might be a physical constant, coordinate two a rate parameter, coordinate three the height of a mountain — I am not told they cluster, I just have to report all $p$ numbers and pay for the total miss.

The answer that everyone, including me, reaches for is $\hat\theta=X$: report each coordinate as observed. Its risk is the cleanest thing in statistics: $E\|X-\theta\|^2=\sum_i E(X_i-\theta_i)^2=p$, flat in $\theta$. And it is not just convenient, it is *certified*: it is the maximum-likelihood estimate, it is unbiased and among unbiased estimators it has minimum variance, it is minimax under squared error, it is the best translation-invariant rule, and Gauss already showed back in 1823 that for a normal mean the least-squares estimate beats every other linear unbiased one. So $X$ wears every medal the theory hands out. I have, in fact, been trying to *prove* that it is admissible — that nothing can uniformly beat it — and I expect to succeed, because what else could the answer possibly be? You have one observation per unknown; surely you cannot do better than reporting what you saw.

But let me actually stare at the thing I observe, the squared length $\|X\|^2$, instead of taking the certificates on faith. Write $X=\theta+(X-\theta)$ and expand:
$$\|X\|^2=\|\theta\|^2+\|X-\theta\|^2+2(X-\theta)\cdot\theta.$$
The middle piece has expectation $p$, one for each coordinate; the cross term has mean zero. So $E\|X\|^2=\|\theta\|^2+p$. And it is not just the mean — for fixed $\theta$ the relative fluctuation of $\|X\|^2$ is of order $\sqrt p/p\to0$, so for large $p$ I can write $\|X\|^2=\|\theta\|^2+p+O_p(\sqrt{p+\|\theta\|^2})$, concentrated. Read that again. The vector I observe is *systematically too long*. Its squared length overshoots the truth's squared length by about $p$. In high dimension, when I see $\|X\|^2$, I essentially *know* that $\|\theta\|^2\approx\|X\|^2-p$ — the true mean lies on a sphere of radius $\sqrt{\|X\|^2-p}$, well inside the sphere of radius $\|X\|$ where my observation sits.

And yet my estimator $\hat\theta=X$ keeps the full length $\|X\|$. So I am confidently placing my estimate out on a sphere I am practically certain is too big, outside the ball where I am nearly sure $\theta$ actually lives. That is a strange thing to be doing. If I know my point is too far out, why wouldn't I pull it in? The obvious correction is to multiply $X$ by the factor that restores the right length: shrink by $\sqrt{(\|X\|^2-p)/\|X\|^2}$, or to first order multiply by something like $1-p/\|X\|^2$. Let me hold onto the *form*: an estimator
$$\hat\theta(X)=\Big(1-h(\|X\|^2)\Big)\,X,$$
a spherically symmetric rule that slides $X$ inward along its own ray by a length that depends on how long $X$ is. Pulling toward the origin (toward whatever point I center on) — *shrinkage*.

This is already enough to make me nervous about the certificates. None of them — unbiasedness, minimax, invariance — is a statement that the *total* risk over $p$ coordinates can't be lowered. They are about single coordinates, or about a symmetry I might be wrong to demand. The length argument says: break the symmetry, accept a little bias, pull in.

Now, is this real, or am I fooling myself? The overshoot is $\approx p$ on top of $\|X\|^2$, and the noise in $\|X\|^2$ is of order $\sqrt p$. For one coordinate, $p=1$: the "overshoot" of $1$ is the same size as the fluctuation, I can't distinguish "$X$ is too long" from "$X$ jiggled" — and indeed Blyth in 1951, and Hodges and Lehmann, already proved $X$ admissible for $p=1$. There is nothing to exploit there. So whatever I am sensing only becomes visible when there are enough coordinates that the overshoot $p$ dominates the noise $\sqrt p$, i.e. when $p$ is genuinely large. The threshold is going to be a dimension thing.

Let me be careful, though, because the length heuristic is seductive and I don't want to over-trust it. It says "$X$ is too long *on average*, and concentrates as $p\to\infty$." That is a statement about a *fixed* $\theta$ as the dimension runs off to infinity. It is not, by itself, a proof that for a *fixed* finite $p$ the estimator $X$ is dominated *uniformly over all* $\theta$. Those are different claims, and the gap between them is exactly the orthogonal jiggle of $X$ around the true direction. If I tried to turn "$X$ overshoots" directly into "shrinking dominates," I would be quietly assuming the overshoot is reliable for every $\theta$, and for $\theta$ with small $\|\theta\|$ that is shakier. So the heuristic is a compass, not a proof. I will need an honest risk computation. But the compass is pointing somewhere, and it tells me the action is at $p\ge 3$.

Let me check the bottom of that threshold directly: is $X$ admissible at $p=2$? Here I can use the information inequality, the same Cramér–Rao machinery that bounds an estimator's risk from below by its squared bias plus a variance term. Take any competitor $\hat\theta$ with bias $b(\theta)=E_\theta\hat\theta-\theta$, and write $b_{ij}=\partial b_i/\partial\theta_j$. The information inequality gives, after choosing the optimal contrast direction,
$$R(\theta)\ \ge\ \|b(\theta)\|^2+p+2\sum_i b_{ii}(\theta)+\sum_{i,j}b_{ij}^2(\theta).$$
For a spherically symmetric competitor the bias points along $\theta$, $b(\theta)=-\varphi(\|\theta\|^2)\theta$, and dropping the last nonnegative sum,
$$R(\theta)\ \ge\ p+\|\theta\|^2\varphi^2-2p\varphi-4\|\theta\|^2\varphi'.$$
Suppose at $p=2$ some competitor strictly beats $X$, so $R(\theta)\le 2$ for all $\theta$ with strict inequality somewhere. Since $X$ is spherically symmetric, I can average the competitor over rotations without increasing risk, so a strict dominator would leave me with a spherically symmetric one and hence a $\varphi$ not identically zero. Combining $R\le2$ with the lower bound gives
$$0\ge \|\theta\|^2\varphi^2-4\varphi-4\|\theta\|^2\varphi'.$$
Put $t=\|\theta\|^2$ and $\psi(t)=t\varphi(t)$. Then $\varphi=\psi/t$ and $\varphi'=\psi'/t-\psi/t^2$, so the two linear terms cancel:
$$0\ge \frac{\psi^2(t)}{t}-4\psi'(t),\qquad\text{hence}\qquad \psi'(t)\ge \frac{\psi^2(t)}{4t}.$$
This makes $\psi$ nondecreasing. If $\psi(t_0)<0$ at some point, then $\psi$ stays negative to the left of $t_0$, and integrating from $t$ to $t_0$ gives
$$-\frac1{\psi(t_0)}+\frac1{\psi(t)}\ge \frac14\log\frac{t_0}{t}.$$
The left side is bounded as $t\downarrow0$, while the right side diverges. If $\psi(t_0)>0$, the same integration from $t_0$ to $t$ gives
$$\frac1{\psi(t_0)}-\frac1{\psi(t)}\ge \frac14\log\frac{t}{t_0},$$
whose right side diverges as $t\to\infty$ while the left side is bounded. Both signs are impossible; therefore $\psi\equiv0$, so $\varphi\equiv0$, and no strict dominator exists. At $p=2$ the usual estimator is admissible. Good. The threshold really is $p\ge3$, exactly where the recurrence-of-random-walks story also flips: dimension three is where independent fluctuations stop coming back and start being something I can average and lean on.

So the target is sharp: for $p\ge3$ I want to *prove* a shrinkage rule of the form $(1-h(\|X\|^2))X$ beats $X$ uniformly. Let me commit to a concrete one-parameter family and grind out the risk. Take
$$\hat\theta_b(X)=\Big(1-\frac{b}{a+\|X\|^2}\Big)X,\qquad a,b>0,$$
with $a$ a positive offset that keeps the factor from blowing up near the origin, $b$ controlling how hard I shrink. Substitute $X=Y+\theta$ with $Y\sim N(0,I)$ and expand the loss of this estimator. The leading term is $E\|Y\|^2=p$ — that's the usual risk — and the correction from shrinking is
$$E\Big\|\Big(1-\tfrac{b}{a+\|X\|^2}\Big)X-\theta\Big\|^2 = p-2b\,E\!\left[\frac{Y\cdot(Y+\theta)}{a+\|Y+\theta\|^2}\right]+b^2E\!\left[\frac{\|X\|^2}{(a+\|X\|^2)^2}\right].$$
The $b^2$ term is the price of shrinking — it is positive — and the $-2b$ term is the possible gain. I cannot certify the sign just by saying the $\theta\cdot Y$ term averages out, because the denominator also contains $\theta\cdot Y$. So I push $a\to\infty$, which makes the offset dominate and lets me expand $1/(a+\|X\|^2)$ in powers of $1/a$. The first-order cross term vanishes because $E[Y\mid\|Y\|^2]=0$, and the next quadratic correction is exactly where the loss of two dimensions appears. The risk comes out, uniformly in $\theta$ as $a\to\infty$, as
$$\le\ p-2b\,\frac{(p-2)-b/2}{a+\|\theta\|^2}+o\!\Big(\frac1{a+\|\theta\|^2}\Big).$$
There it is: as long as $0<b<2(p-2)$ the bracket $(p-2)-b/2$ is positive, the whole correction is negative, and for sufficiently large $a$ this estimator strictly beats $X$. So for $p\ge3$ the usual estimator is *inadmissible*. And the improvement, for the best $b$, is largest when the bracket $b\big[(p-2)-b/2\big]$ is maximized, which is at $b=p-2$, giving an improvement asymptotic to $(p-2)^2/\|\theta\|^2$. The number $p-2$ falls out of an optimization, not out of thin air; and it is exactly $\le0$ for $p\le2$, which is why those dimensions are safe.

I have my inadmissibility, but I am unhappy with this proof. It is an asymptotic-in-$a$ argument; the estimator carries this nuisance offset $a$ that I had to send to infinity; the closed form is not clean; and the algebra was a slog with several remainder terms I had to bound by hand. It proves the point, but it is too awkward to use as a practical rule. I want the *clean* estimator — the one with $a=0$, $\hat\theta=(1-b/\|X\|^2)X$ — and I want its exact risk, not an asymptotic one. So I ask the harder question directly: with $a=0$, what is the exact optimal $b$, and what is the exact risk?

There must be a cleaner way to handle the cross term. The object that keeps appearing is an expectation of a noise term times a function of the observation, and the normal density has exactly the derivative that should turn that product into a derivative. Start in one dimension. $Z\sim N(0,1)$ with density $\phi$, and $f$ a nice (absolutely continuous) function with $E|f'(Z)|<\infty$. Look at $E[f'(Z)]=\int f'(z)\phi(z)\,dz$ and integrate by parts. The boundary term $f(z)\phi(z)\big|_{-\infty}^{\infty}$ vanishes because $\phi$ kills it, and the key fact about the *normal* density is $\phi'(z)=-z\phi(z)$. So
$$E[f'(Z)]=-\int f(z)\phi'(z)\,dz=\int z f(z)\phi(z)\,dz=E[Z f(Z)].$$
That is the whole identity: $E[Z f(Z)]=E[f'(Z)]$. (If I distrust the boundary term I can do it another way — split the density as $\phi(z)=\int_z^\infty t\phi(t)\,dt$ for $z>0$ and the mirror for $z<0$, swap the order of integration by Fubini, and the same identity drops out without ever differentiating $\phi$ at the ends.) For a general $X\sim N(\mu,\sigma^2)$, set $Z=(X-\mu)/\sigma$ and I get
$$\tfrac{1}{\sigma^2}E\big[(X-\mu)f(X)\big]=E[f'(X)].$$
This little thing is already remarkable: it lets me compute $E[(X-\mu)f(X)]$ — a covariance that ought to depend on the unknown $\mu$ — purely as $E[f'(X)]$, a derivative I can evaluate at the data without knowing $\mu$ at all. And in $p$ dimensions, with $X\sim N(\mu,\sigma^2I)$ and $f:\mathbb R^p\to\mathbb R^p$ each coordinate nice, applying it coordinatewise and summing gives
$$\tfrac{1}{\sigma^2}\sum_i E\big[(X_i-\mu_i)f_i(X)\big]=E\Big[\sum_i\frac{\partial f_i}{\partial X_i}\Big]=E[\operatorname{div} f].$$

Now watch what this does to the risk of *any* estimator of the shrinkage shape. Write the estimator as $\hat\theta=X-g(X)$, so $g$ is the amount I pull $X$ in. Then
$$E\|\hat\theta-\theta\|^2=E\|(X-\theta)-g(X)\|^2=E\|X-\theta\|^2-2E\big[(X-\theta)\cdot g(X)\big]+E\|g(X)\|^2.$$
The first term is $p$ (taking $\sigma^2=1$). The middle term is exactly the covariance my identity handles: $E[(X-\theta)\cdot g(X)]=E[\operatorname{div}g(X)]$. So, with no asymptotics, no offset, no remainder,
$$R(\theta,\hat\theta)=p-2E[\operatorname{div}g(X)]+E\|g(X)\|^2.$$
This is an *unbiased risk estimate*: the random quantity $p-2\operatorname{div}g(X)+\|g(X)\|^2$ has expectation equal to the risk. The whole problem of comparing $\hat\theta$ to $X$ has reduced to computing one divergence and one norm. The bias has been integrated away by the normal's $\phi'=-z\phi$ structure.

So let me just plug in the clean shrinkage I wanted. Pull toward the origin by $g(X)=c\,X/\|X\|^2$, i.e. $\hat\theta=(1-c/\|X\|^2)X$, and leave the constant $c$ free for the moment. Two pieces. The norm:
$$\|g\|^2=\frac{c^2\|X\|^2}{\|X\|^4}=\frac{c^2}{\|X\|^2}.$$
Now the divergence. Differentiate the $i$-th coordinate $g_i=cX_i/\|X\|^2$:
$$\frac{\partial}{\partial X_i}\frac{cX_i}{\|X\|^2}=\frac{c}{\|X\|^2}+cX_i\cdot\Big(-\frac{2X_i}{\|X\|^4}\Big)=\frac{c}{\|X\|^2}-\frac{2cX_i^2}{\|X\|^4}.$$
Sum over $i=1,\dots,p$. The first piece sums to $cp/\|X\|^2$. The second piece sums to $2c\sum_iX_i^2/\|X\|^4=2c\|X\|^2/\|X\|^4=2c/\|X\|^2$. So
$$\operatorname{div}g=\frac{cp}{\|X\|^2}-\frac{2c}{\|X\|^2}=\frac{c(p-2)}{\|X\|^2}.$$
*There* is the $p-2$. It is not assumed and it is not optimized into existence — it is the dimension $p$ from differentiating $p$ coordinates, minus $2$ from the two derivatives that land on the $\|X\|^2$ in the denominator. The competition between "$p$ coordinates each contribute a shrink" and "the denominator fights back twice" leaves $p-2$. And it is manifestly $\le0$ for $p\le2$.

Assemble:
$$R(\theta,\hat\theta)=p-2E\Big[\frac{c(p-2)}{\|X\|^2}\Big]+E\Big[\frac{c^2}{\|X\|^2}\Big]=p-\big(2c(p-2)-c^2\big)\,E\Big[\frac{1}{\|X\|^2}\Big].$$
Now I choose $c$ to make the improvement $2c(p-2)-c^2$ as large as possible. Differentiate in $c$: $2(p-2)-2c=0$, so $c=p-2$, and the maximal improvement is $2(p-2)^2-(p-2)^2=(p-2)^2$. The clean optimal estimator is
$$\hat\theta^{\mathrm{JS}}(X)=\Big(1-\frac{p-2}{\|X\|^2}\Big)X,$$
and its exact risk is
$$\boxed{\,R(\theta,\hat\theta^{\mathrm{JS}})=p-(p-2)^2\,E_\theta\!\Big[\frac1{\|X\|^2}\Big]\,}.$$
Because $\|X\|^2>0$ almost surely and $E[1/\|X\|^2]$ is a strictly positive finite number for $p\ge3$, the second term is strictly positive for every $\theta$. So $R(\theta,\hat\theta^{\mathrm{JS}})<p=R(\theta,X)$ for all $\theta$. The usual estimator is dominated — uniformly, with strict inequality everywhere — and is therefore inadmissible whenever $p\ge3$. No asymptotics, no offset, no hand-waved remainder. The messy $a\to\infty$ computation was telling the truth; this is the same truth stated exactly, and it hands me the exact constant $p-2$ that the asymptotic version could only locate as the maximizer of a bracket.

If the coordinate variance is $\sigma^2$ instead of $1$, the same calculation just carries the scale through the cross term: with $g(X)=c\sigma^2X/\|X\|^2$,
$$R(\theta,X-g)=p\sigma^2-\big(2c(p-2)-c^2\big)\sigma^4E_\theta\!\Big[\frac1{\|X\|^2}\Big],$$
so the maximizing dimensionless constant is still $c=p-2$, and the shrink factor becomes $1-(p-2)\sigma^2/\|X\|^2$.

Let me sanity-check the constant against the two dimensions I already settled. For $p\le2$, $p-2\le0$: the shrinkage factor $1-(p-2)/\|X\|^2$ either does nothing useful or pushes outward, and indeed $E[1/\|X\|^2]=\infty$ there, so the formula degenerates — consistent with $X$ being admissible at $p=1,2$. For $p\ge3$ it bites.

And let me read the size of the win at the most favorable point, $\theta=0$. Then $\|X\|^2\sim\chi^2_p$ and $E[1/\chi^2_p]=1/(p-2)$, so
$$R(0,\hat\theta^{\mathrm{JS}})=p-(p-2)^2\cdot\frac1{p-2}=p-(p-2)=2.$$
The risk at the origin is $2$, in *every* dimension $p\ge3$. The usual estimator has risk $p$ there. So in $p=100$ I have cut the risk from $100$ to $2$ near the origin — a fiftyfold improvement — by doing nothing but pulling toward a point. More generally, off the origin, $\|X\|^2$ is a noncentral chi-square; writing it as a central $\chi^2_{p+2K}$ with $K\sim\mathrm{Poisson}(\|\theta\|^2/2)$, $E[1/\|X\|^2]=E[1/(p-2+2K)]$, so
$$R(\theta,\hat\theta^{\mathrm{JS}})=p-E\Big[\frac{(p-2)^2}{p-2+2K}\Big],$$
which interpolates from $2$ at $\theta=0$ up toward $p$ as $\|\theta\|\to\infty$ — the gain fades when the truth is far out, because then $X$ is barely too long relative to its own length and there is little to correct, but the gain is *always* strictly positive.

Now I have to face the thing that made me believe in admissibility in the first place, because the per-coordinate intuition still screams that this is impossible. Coordinate three is a mountain's height; coordinate one is a physical constant; they share nothing. How can using $X_1$ and $X_2$ improve my estimate of $\theta_3$? Let me look at where the improvement actually lives. The shrink factor $1-(p-2)/\|X\|^2$ couples the coordinates through the *single joint statistic* $\|X\|^2$, and that statistic is precisely the one that measures the common overshoot I found at the very start — it knows the typical scale of the whole vector even though it knows nothing about any individual mean. That is the "latent information between independent problems": not a causal link between the mountain and the constant, but the bare arithmetic fact that the squared length of $p$ independent overshoots concentrates, and that concentration is information about how much to pull in.

And it does cost something per coordinate. Shrinking introduces bias: for a coordinate whose true mean is large, multiplying by a factor below one pulls the estimate *away* from the truth, and that coordinate's own risk can go *up*. So there is no claim — and there had better not be — that shrinkage beats $X$ in every coordinate separately. The deterioration in some components is real and unavoidable; the theorem is entirely about the *sum*. What the risk formula proves is that the summed variance reduction from shrinking $p$ coordinates outweighs the summed squared bias, and it does so for every $\theta$ once $p\ge3$. The per-coordinate intuition resists because it is asking the wrong question: it asks whether each problem is helped, and the answer is no; the loss function asks whether the *total* is helped, and the answer is yes. The two answers only diverge once the loss adds up the components, which is exactly the regime I am in. If the loss did *not* add up — if I were graded on the single worst coordinate, say — the whole effect could disappear; the domination really does lean on the additivity of squared-error loss across coordinates.

Why the precise constant $p-2$, beyond "it maximizes the bracket"? Two readings make it feel inevitable. The first is the geometry. Picture the problem in two coordinates: one along the true direction $\theta$, one for the orthogonal residual. The observation lands, typically, at a point whose distance from the origin is about $\sqrt{\|\theta\|^2+p}$ rather than $\|\theta\|$ — it sits on a longer sphere, displaced both outward and sideways by the $p-1$ residual dimensions collapsed into one. If I knew exactly where it sat, the best spherically symmetric pull-in — the projection of the typical observation back onto the ray through the target — would shrink by a factor like $1-(p-1)/\|X\|^2$. But the observation is *not* exactly at its typical spot; it jiggles around that typical point by a unit of noise in each direction, and once I average the risk over that jiggle the optimal pull is slightly gentler. Accounting for the randomness shaves the geometric guess $p-1$ down to $p-2$: the lost unit is the price of $X$ being only approximately, not exactly, at its expected length. So $p-1$ is what you'd use if the overshoot were deterministic; $p-2$ is what survives the stochastic correction — which is also why the naïve geometric heuristic, run carelessly, would wrongly suggest improvement even at $p=2$, and you need the honest stochastic accounting to get the threshold right.

The second reading tells me *why this exact form and not some other shrinkage*. Pretend, just as a device, that the means really were drawn from a prior $\theta_i\sim N(0,A)$. The Bayes estimate under squared error is the posterior mean, which for this conjugate setup is the fixed shrink $\hat\theta=\big(1-\tfrac1{A+1}\big)X$. I don't know $A$, but I can estimate the quantity $1/(A+1)$ from the data, because under this prior the marginal of $X$ is $X_i\sim N(0,A+1)$, so $\|X\|^2\sim(A+1)\chi^2_p$ and therefore $E\big[(p-2)/\|X\|^2\big]=1/(A+1)$. That is to say: $(p-2)/\|X\|^2$ is an unbiased estimate of the very shrink factor $1/(A+1)$ that Bayes would use. Plug the data-driven estimate in place of the unknown and I get $\hat\theta=\big(1-\tfrac{p-2}{\|X\|^2}\big)X$ — exactly the estimator I derived. So the strange constant $p-2$ is the one that makes the shrink factor an honest plug-in for the Bayes shrink, learned from $\|X\|^2$ itself. The prior was only scaffolding to explain the form; the domination is a theorem about fixed, unrelated $\theta$ and needs no prior at all. This is also why I should shrink by a *data-driven* amount and not a fixed fraction: a fixed fraction would be right only for one value of $A$, whereas $(p-2)/\|X\|^2$ adapts — it shrinks hard when $\|X\|^2$ is small (the means look near the center) and barely at all when $\|X\|^2$ is huge (the means are far out and there's little to gain), which is exactly the behavior the risk formula rewards.

One more thing the formula warns me about. The factor $1-(p-2)/\|X\|^2$ goes *negative* when $\|X\|^2<p-2$, which flips the sign of $X$ — I'd be reporting an estimate pointing the wrong way, which is plainly silly. The repair writes itself: replace the factor by its positive part,
$$\hat\theta^{\mathrm{JS+}}(X)=\Big(1-\frac{p-2}{\|X\|^2}\Big)_{\!+}X,$$
clamping the shrink at zero so I never overshoot through the origin. This is no longer the exact rule whose risk I just computed, so I should not reuse the same formula for it; the separate risk comparison shows that the clamped rule dominates the plain $\hat\theta^{\mathrm{JS}}$.

A concrete way to picture the rule is a panel of unrelated rate estimates with comparable Gaussian errors and a center chosen before looking at the data. The usual estimate says to report each noisy rate by itself. Shrinkage says to pull every rate toward the common center by the data-driven factor. The total squared error over the panel is the object being controlled, so some genuinely high or low individual rates can be pulled the wrong way while the panel total still improves. When the score is a sum, unrelated problems can lend each other strength through their common scale.

Pulling the chain together: I started believing $X$ was unbeatable because it carries every classical optimality certificate. Staring at $\|X\|^2$ showed me $X$ is systematically too long by an amount that grows with $p$, so it confidently sits outside the ball where $\theta$ lives — a reason to pull in. That correction is invisible at $p=1,2$ (noise swamps the overshoot; $X$ is provably admissible there, including by an information-inequality argument at $p=2$) and real at $p\ge3$. Committing to a shrinkage family and computing the risk confirmed inadmissibility, first messily through an offset-and-limit argument that located the improvement $\propto(p-2)^2/\|\theta\|^2$, then cleanly through the normal's integration-by-parts identity $E[(X-\mu)f(X)]=E[f'(X)]$, which converts the risk of any $\hat\theta=X-g$ into $p-2E[\operatorname{div}g]+E\|g\|^2$. The divergence of $X/\|X\|^2$ is $(p-2)/\|X\|^2$ — the $p$ from $p$ coordinates and the $-2$ from the two derivatives on the denominator — so plugging $g=(p-2)X/\|X\|^2$ makes $\operatorname{div}g=(p-2)^2/\|X\|^2$ and gives the exact risk $p-(p-2)^2E[1/\|X\|^2]<p$ for all $\theta$. The constant $p-2$ is simultaneously the maximizer of the risk improvement, the stochastic correction of the geometric guess $p-1$, and the data-driven plug-in for the Bayes shrink $1/(A+1)$. The win lives in the sum, channeled through the one joint statistic $\|X\|^2$; per coordinate, some are hurt; that split is the whole content of the surprise.

```python
import numpy as np

def james_stein(X, sigma=1.0, center=0.0, positive_part=False):
    """Plain James-Stein estimate of theta from one draw X ~ N(theta, sigma^2 I).

    Shrinks X toward `center` by the data-driven factor 1 - (p-2) sigma^2 / ||X-center||^2.
    Set positive_part=True for the clamped refinement.
    """
    X = np.asarray(X, dtype=float)
    center = np.broadcast_to(np.asarray(center, dtype=float), X.shape)
    p = X.size
    if p < 3:
        raise ValueError("domination requires dimension p >= 3")
    Z = X - center
    norm2 = np.sum(Z**2)
    if norm2 == 0.0:
        if positive_part:
            return center.copy()
        raise ZeroDivisionError("plain James-Stein factor is undefined at the center")
    factor = 1.0 - (p - 2) * sigma**2 / norm2           # the (p-2) from div(X/||X||^2)
    if positive_part:
        factor = max(factor, 0.0)                       # clamp: never flip through the center
    return center + factor * Z

def plain_js_risk_from_identity(theta, sigma=1.0, n=200_000, rng=None):
    """Monte-Carlo evaluation of the exact plain-JS risk identity.

    R = p*sigma^2 - (p-2)^2*sigma^4*E[1/||X||^2].
    At theta=0 this equals 2*sigma^2 exactly.
    """
    theta = np.asarray(theta, dtype=float)
    p = theta.size
    if p < 3:
        raise ValueError("plain James-Stein domination requires dimension p >= 3")
    rng = rng or np.random.default_rng()
    X = theta + sigma * rng.standard_normal((n, p))
    inv = np.mean(1.0 / np.sum(X**2, axis=1))
    return p * sigma**2 - (p - 2)**2 * sigma**4 * inv
```
