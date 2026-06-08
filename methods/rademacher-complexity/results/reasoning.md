Let me start from the thing that actually goes wrong. I have a class $\mathcal{F}$ of functions, a sample $z_1,\dots,z_m$ drawn i.i.d. from some unknown $P$, and I pick the $f$ that looks best on the sample. For any *fixed* $f$, the empirical mean $\hat{E}_S f=\frac1m\sum_i f(z_i)$ hugs the true mean $Pf=E_P f$ — law of large numbers, done. But I don't fix $f$; I let the learner rummage through $\mathcal{F}$ and grab whichever $f$ fits the sample. So the quantity that decides whether my chosen predictor generalizes is not any single gap, it's the worst gap over the class,

$$\Phi(S)=\sup_{f\in\mathcal{F}}\bigl(Pf-\hat{E}_S f\bigr).$$

That's the enemy. If the class is rich enough that *some* member can shadow whatever the sample happens to look like, $\Phi(S)$ stays large no matter how big $m$ gets, and the empirical risk is a lie.

Now, the bounds I know all have the same skeleton: empirical risk plus a penalty, where the penalty is a number like the VC dimension. And here is what nags at me about that. The VC dimension of, say, halfspaces in $\mathbb{R}^d$ is $d+1$, full stop. It is the same number whether my data are jammed into a tiny ball, spread on a line, or scattered so that no two points share any structure. It is a supremum over *all* possible point configurations and *all* distributions. But the data in front of me are one particular configuration drawn from one particular $P$. The worst-case number is paying for adversarial inputs I will never see. For *model selection* — where I'm comparing a simple class against a rich one and the bound has to actually track which one will test better — a penalty that refuses to look at my sample is going to misrank them. And there's the experimental smell too: people have checked, and a penalty that doesn't depend on the data just doesn't reliably pick the right model. So I want a penalty I can *compute from this very sample*, that adapts to where the points actually fell.

What would that even look like? Let me think about what "capacity" really means operationally. A class is dangerous exactly when it can fit *anything* — when, handed an arbitrary target pattern over my $m$ points, some $f\in\mathcal{F}$ can match it. The cleanest "arbitrary pattern" is pure noise. So picture flipping a fair coin at each sample point: assign $\sigma_i\in\{+1,-1\}$, uniform and independent. These are meaningless labels — they have nothing to do with $P$. Ask: how well can the class line up with this noise on my sample? Measure the alignment by the correlation

$$\frac1m\sum_{i=1}^m \sigma_i f(z_i),$$

and let the class do its best:

$$\sup_{f\in\mathcal{F}}\frac1m\sum_i\sigma_i f(z_i).$$

If the class is poor, no $f$ can chase the random signs, this is small. If the class is so rich it can match almost any sign pattern, some $f$ achieves nearly $\frac1m\sum_i|f(z_i)|$ — large. And because $E\sigma_i=0$, *any* correlation a function achieves here is purely the class flexing its capacity to overfit noise; there's no real signal to find. Average over the coins:

$$\hat{R}_S(\mathcal{F})=E_\sigma\Bigl[\sup_{f\in\mathcal{F}}\frac1m\sum_i\sigma_i f(z_i)\,\Bigm|\,z_1,\dots,z_m\Bigr].$$

This is a number I can compute from the *one realized sample* — I never need $P$, I just resample the coins. It is, by construction, data-dependent: it sees exactly where my points landed. That feels like the right object. Call it the empirical capacity of $\mathcal{F}$ on $S$. (Its expectation over samples, $R_m(\mathcal{F})=E_S\hat{R}_S(\mathcal{F})$, I'll use when I want a distribution-level statement.)

So I have a candidate penalty. The question that decides everything: does it actually *control* $\Phi(S)$? Is there a clean inequality $E\,\Phi(S)\le (\text{const})\cdot R_m(\mathcal{F})$? If yes, I'm done in principle — the noise-correlation, which I can read off the sample, governs the generalization gap I care about.

Let me try to prove it. The trouble in $\Phi$ is the term $Pf=E_{z\sim P}f(z)$ — it's the true mean, I don't have it. But I can *manufacture* a copy of it. Imagine drawing a second independent sample $S'=(z_1',\dots,z_m')$ from the same $P$ — a ghost sample I never actually collect, purely a device. For each $f$, $E_{S'}\hat{E}_{S'}f = Pf$, because $S'$ is i.i.d. from $P$. So I can write the true mean as the expected empirical mean over the ghost:

$$E_S\,\Phi(S)=E_S\sup_f\bigl(Pf-\hat{E}_S f\bigr)=E_S\sup_f\Bigl(E_{S'}\bigl[\hat{E}_{S'}f\bigr]-\hat{E}_S f\Bigr).$$

Now the $E_{S'}$ is buried inside the $\sup_f$. I want to pull it out front. The supremum of an expectation is at most the expectation of the supremum — that's just Jensen, since $\sup$ is convex (a $\sup$ of linear-in-$S'$ things). Pulling $E_{S'}$ outside the $\sup$ only enlarges it:

$$E_S\,\Phi(S)\le E_{S,S'}\,\sup_f\bigl(\hat{E}_{S'}f-\hat{E}_S f\bigr)=E_{S,S'}\,\sup_f\frac1m\sum_i\bigl(f(z_i')-f(z_i)\bigr).$$

Good — now there's no $P$ anywhere, just two honest empirical sums. And look at the structure: each term in the sum is a *difference* $f(z_i')-f(z_i)$ between a ghost point and a real point, both drawn from the same $P$ and independent of each other. So the pair $(z_i,z_i')$ is exchangeable: swapping them, $f(z_i')-f(z_i)\mapsto f(z_i)-f(z_i')$, flips the sign of that term, but since $z_i$ and $z_i'$ have the same distribution, the joint expectation over $(S,S')$ is completely unchanged by the swap. I can swap any subset of the indices and nothing moves.

That exchangeability is exactly a hidden $\pm1$ symmetry. Multiply the $i$-th term by an independent random sign $\sigma_i\in\{\pm1\}$: when $\sigma_i=+1$ leave it, when $\sigma_i=-1$ flip $z_i\leftrightarrow z_i'$ — and I just argued the expectation is invariant under that flip. So I can insert the signs for free:

$$E_{S,S'}\,\sup_f\frac1m\sum_i\bigl(f(z_i')-f(z_i)\bigr)=E_{\sigma,S,S'}\,\sup_f\frac1m\sum_i\sigma_i\bigl(f(z_i')-f(z_i)\bigr).$$

There they are — Rademacher signs, conjured out of the symmetry of the ghost-sample construction, not assumed. Now split the supremum. $\sigma_i(f(z_i')-f(z_i))=\sigma_i f(z_i')+(-\sigma_i)f(z_i)$, and the sup of a sum is at most the sum of sups:

$$\le E_{\sigma,S'}\sup_f\frac1m\sum_i\sigma_i f(z_i')\;+\;E_{\sigma,S}\sup_f\frac1m\sum_i(-\sigma_i)f(z_i).$$

In the second term $-\sigma_i$ is distributed exactly like $\sigma_i$, so it equals the first; and $S,S'$ are identically distributed, so both terms equal $E_S\,\hat{R}_S(\mathcal{F})=R_m(\mathcal{F})$. Therefore

$$E_S\,\Phi(S)\;\le\;2\,R_m(\mathcal{F}).$$

The factor of $2$ isn't decoration — it's literally the two halves, ghost and real, each contributing one Rademacher term once I broke the difference apart. There's no obvious way to kill it with this argument, so I'll keep it and move on. The point is the inequality holds: the expected uniform gap is bounded by twice the noise-correlation capacity, and the right-hand side is computable from the sample. The thing I invented to measure overfitting-to-noise *is* the thing that governs the gap. That's the whole idea, and it dropped out of the symmetrization.

But an expectation bound isn't a guarantee on the realized run — I need "with high probability," and I need the gap for *every* $f$ simultaneously (that's what $\Phi$ being a sup already buys me). For that I want to show $\Phi(S)$ doesn't fluctuate much around its mean. So how sensitive is $\Phi(S)$ to a single data point? Suppose I swap one point, $z_m\to z_m'$, holding the rest. Then

$$\Phi(S')-\Phi(S)\le\sup_f\bigl(\hat{E}_S f-\hat{E}_{S'}f\bigr)=\sup_f\frac{f(z_m)-f(z_m')}{m},$$

because the difference of two suprema is at most the sup of the difference, and only the $m$-th term differs. If the functions live in $[0,1]$ — which I can always arrange for a bounded loss — then $|f(z_m)-f(z_m')|\le1$, so $|\Phi(S)-\Phi(S')|\le\frac1m$. One point moves the whole supremum by at most $1/m$. That bounded-difference property is exactly the hypothesis of McDiarmid's inequality: if changing coordinate $i$ moves $f$ by at most $c_i$, then $P\{f-Ef\ge t\}\le\exp\!\bigl(-2t^2/\sum_i c_i^2\bigr)$. Here every $c_i=1/m$, so $\sum_i c_i^2=m\cdot\frac1{m^2}=\frac1m$, and

$$P\{\Phi(S)-E\Phi(S)\ge t\}\le\exp(-2t^2 m).$$

Set the right side to $\delta$: $2t^2m=\log\frac1\delta$, i.e. $t=\sqrt{\frac{\log(1/\delta)}{2m}}$. So with probability at least $1-\delta$,

$$\Phi(S)\le E\,\Phi(S)+\sqrt{\frac{\log(1/\delta)}{2m}}\le 2R_m(\mathcal{F})+\sqrt{\frac{\log(1/\delta)}{2m}}.$$

Unfolding $\Phi(S)\ge Pf-\hat{E}_S f$ for every $f$ gives the bound I wanted: with probability $\ge1-\delta$, every $f\in\mathcal{F}$ obeys

$$Pf\;\le\;\hat{E}_S f\;+\;2R_m(\mathcal{F})\;+\;\sqrt{\frac{\log(1/\delta)}{2m}}.$$

There's a wrinkle though. $R_m(\mathcal{F})$ is an expectation over samples — it still references $P$ through $E_S$. The penalty I can *actually evaluate* is $\hat{R}_S(\mathcal{F})$, conditioned on this sample. So I need the direction $R_m(\mathcal F)\le\hat{R}_S(\mathcal F)+\text{slack}$ on the realized sample. Same trick: how much does $\hat{R}_S(\mathcal{F})$ change if I swap one point? Each summand $\sigma_i f(z_i)/m$ is bounded in magnitude by $1/m$ for $f\in[0,1]$, and the sup-then-expectation is $1$-Lipschitz in each coordinate's contribution, so swapping one $z_i$ moves $\hat{R}_S$ by at most $1/m$ again. McDiarmid once more gives the one-sided event

$$R_m(\mathcal F)\le\hat R_S(\mathcal F)+\sqrt{\frac{\log(2/\delta)}{2m}}$$

with probability $1-\delta/2$. (For $\{\pm1\}$-valued functions the per-point jump is $2/m$ rather than $1/m$, which changes the constant under the square root unless I first reduce to a $[0,1]$ loss class.) Now I spend the confidence budget on two events: McDiarmid on $\Phi$ at level $\delta/2$, and this one-sided $R_m\to\hat{R}_S$ substitution at level $\delta/2$. Each contributes a $\sqrt{\log(2/\delta)/2m}$ term — one directly, and the substitution doubles the second through the factor-$2$ on $R$ — so the two confidence terms add to a coefficient of $3$. With probability at least $1-\delta$, for all $f$:

$$Pf\;\le\;\hat{E}_S f\;+\;2\hat{R}_S(\mathcal{F})\;+\;3\sqrt{\frac{\log(2/\delta)}{2m}}.$$

This is the one I actually want, because every term on the right is computable from the single training sample. The penalty *is* the empirical Rademacher complexity, and it adapts to the data.

Now let me make sure this connects to the world I came from — pattern classification with $\{\pm1\}$ labels and zero-one loss — because the bound above is stated for a generic function class $\mathcal{F}\subseteq[0,1]$, and in classification the object is the loss class $\mathcal{G}=\{(x,y)\mapsto\mathbf{1}[h(x)\neq y]:h\in\mathcal H\}$. I want the penalty in terms of the *hypothesis* class $\mathcal H$, not the loss class. Write the indicator using the labels: for $h,y\in\{\pm1\}$,

$$\mathbf{1}[h(x)\neq y]=\frac{1-y\,h(x)}{2}.$$

Then

$$\hat{R}_S(\mathcal{G})=E_\sigma\sup_{h}\frac1m\sum_i\sigma_i\frac{1-y_i h(x_i)}{2}=\frac12 E_\sigma\sup_h\frac1m\sum_i(-\sigma_i y_i)\,h(x_i).$$

The constant $\frac1m\sum_i\sigma_i\cdot\frac12$ that came from the "$1$" drops out under the sup because it doesn't depend on $h$ and averages to zero over $\sigma$. And $-\sigma_i y_i$, for a fixed label $y_i\in\{\pm1\}$, is distributed exactly like $\sigma_i$. So

$$\hat{R}_S(\mathcal{G})=\frac12 E_\sigma\sup_h\frac1m\sum_i\sigma_i h(x_i)=\frac12\,\hat{R}_{S_X}(\mathcal H).$$

The loss-class capacity is exactly half the hypothesis-class capacity. Plug into the bound: with probability $\ge1-\delta$, every $h\in\mathcal H$ has

$$P(Y\neq h(X))\;\le\;\hat P_m(Y\neq h(X))\;+\;R_m(\mathcal H)\;+\;\sqrt{\frac{\log(1/\delta)}{2m}}.$$

If I use the fully empirical bound instead, the same half-factor gives

$$P(Y\neq h(X))\;\le\;\hat P_m(Y\neq h(X))\;+\;\hat R_{S_X}(\mathcal H)\;+\;3\sqrt{\frac{\log(2/\delta)}{2m}}.$$

The factor-$2$ on $R_m(\mathcal G)$ and the factor-$\tfrac12$ from the reduction cancel, leaving a clean $R_m(\mathcal H)$. Nice — the noise-correlation of the classifiers themselves is the penalty.

Now I have to convince myself this isn't just a different-looking bound that's secretly *worse* than VC. If $\hat{R}_S$ can be enormous for classes VC handles fine, the whole program is pointless. So let me bound $\hat{R}_S$ from above by something combinatorial and see what falls out. Fix the sample; the class restricted to the sample, $\mathcal H_{|S}=\{(h(x_1),\dots,h(x_m)):h\in\mathcal H\}$, is a *finite* set of $\pm1$ vectors — there are at most $\Pi_{\mathcal H}(m)$ of them, the growth function. Each such vector $u$ has Euclidean norm $\sqrt{m}$ (it's $\pm1$ in $m$ coordinates). So I've reduced the sup over a possibly-infinite class to a max over a finite set of bounded vectors, and I want $E_\sigma\max_{u}\frac1m\langle\sigma,u\rangle$.

For a finite set this is a standard maximal-inequality computation, and I should do it rather than wave at it. Let $A\subseteq\mathbb{R}^m$ be finite with $r=\max_{u\in A}\|u\|_2$. For any $t>0$, by Jensen (exp is convex) and then bounding the max by the sum,

$$\exp\!\Bigl(tE_\sigma\max_{u}\sum_i\sigma_i u_i\Bigr)\le E_\sigma\exp\!\Bigl(t\max_u\sum_i\sigma_i u_i\Bigr)=E_\sigma\max_u\exp\!\Bigl(t\sum_i\sigma_i u_i\Bigr)\le\sum_{u\in A}E_\sigma\prod_i e^{t\sigma_i u_i}.$$

The $\sigma_i$ are independent, so the expectation factorizes, and each factor is a centered $\pm1$ variable times $u_i$ — Hoeffding's lemma gives $E_{\sigma_i}e^{t\sigma_i u_i}\le e^{t^2 u_i^2/2}$. Hence

$$\sum_{u\in A}\prod_i e^{t^2 u_i^2/2}=\sum_{u\in A}e^{t^2\|u\|_2^2/2}\le|A|\,e^{t^2 r^2/2}.$$

Take logs and divide by $t$:

$$E_\sigma\max_u\sum_i\sigma_i u_i\le\frac{\log|A|}{t}+\frac{t r^2}{2}.$$

Minimize over $t$: the optimum is $t=\sqrt{2\log|A|}/r$, giving $r\sqrt{2\log|A|}$. So

$$E_\sigma\max_{u\in A}\frac1m\sum_i\sigma_i u_i\le\frac{r\sqrt{2\log|A|}}{m}.$$

That's Massart's finite-class lemma, derived. Apply it with $A=\mathcal H_{|S}$, $|A|\le\Pi_{\mathcal H}(m)$, $r=\sqrt m$:

$$R_m(\mathcal H)\le E_S\frac{\sqrt m\,\sqrt{2\log\Pi_{\mathcal H}(m)}}{m}=\sqrt{\frac{2\log\Pi_{\mathcal H}(m)}{m}}.$$

And Sauer's lemma bounds $\Pi_{\mathcal H}(m)\le(em/d)^d$ for VC dimension $d$, so $R_m(\mathcal H)\le\sqrt{2d\log(em/d)/m}$ — the VC rate falls right out as a *corollary*. So Rademacher is never essentially worse than VC. But it can be dramatically *better*: the chain went through $|\mathcal H_{|S}|$, the number of sign patterns the class actually realizes *on these points*, and through the realized $\hat{R}_S$ itself. On a benign sample that number — the empirical VC-entropy — can be far below the worst-case $\Pi_{\mathcal H}(m)$, and $\hat{R}_S$ can be far below its worst-case value. The bound *adapts*, exactly as I wanted, and degrades gracefully to VC in the worst case. That's the payoff for measuring capacity on the data instead of over all data.

Now the next pressure. Real classifiers are real-valued — boosted ensembles, neural nets, kernel machines output a score $f(x)\in\mathbb{R}$ and predict $\mathrm{sign}\,f(x)$. The zero-one loss $\mathbf{1}[yf(x)\le0]$ is a step; its Rademacher complexity over a rich real-valued class is unhelpful because the step throws away the margin. I'd rather work with the *margin* $yf(x)$ and a cost $\phi$ that softens the step — say $\phi$ Lipschitz with constant $L$, dominating $\mathbf{1}[\alpha\le0]$, ramping down over a margin width $\gamma$ (so $L=1/\gamma$). Then the empirical margin cost $\hat{E}_m\phi(Yf(X))$ replaces the empirical error, and I need the complexity of the class $\phi\circ\mathcal F=\{(x,y)\mapsto\phi(yf(x))\}$.

Here's the obstruction: I can compute or bound $\hat{R}_S(\mathcal F)$ — the raw scores — but I composed it with $\phi$, and $\hat{R}_S(\phi\circ\mathcal F)$ is a different, more complicated object. I need to peel the $\phi$ off and pay only a controlled price. Does composing with a Lipschitz map blow up the Rademacher average? Intuitively it shouldn't — a $1$-Lipschitz map can't *increase* how far apart function values are, and Rademacher complexity is, at bottom, about spread. I have to keep the normalization honest here: the structural comparison lemmas are usually stated for the absolute $2/m$ Rademacher norm

$$R_m^{\mathrm{str}}(\mathcal F)=E\,E_\sigma\sup_{f\in\mathcal F}\left|\frac2m\sum_i\sigma_i f(X_i)\right|.$$

In that convention, Ledoux–Talagrand's contraction principle says that if $\phi$ is $L$-Lipschitz and $\phi(0)=0$, then

$$R_m^{\mathrm{str}}(\phi\circ\mathcal F)\le 2L\,R_m^{\mathrm{str}}(\mathcal F).$$

The $\phi(0)=0$ normalization is what lets me subtract a reference value and make the composition pass through the origin; the important cost is $2L$, not a loose dimension-dependent term. With that lemma the margin bound assembles itself in the same structural convention. Domination gives $P(Yf(X)\le0)\le E\phi(Yf(X))$, and the centered cost class lets the symmetrized gap be controlled directly by the contracted raw-score class. Tracking the constants gives, with probability $\ge1-\delta$, every $f\in\mathcal F$ satisfies

$$P(Yf(X)\le0)\;\le\;\hat{E}_m\,\phi(Yf(X))\;+\;2L\,R_m^{\mathrm{str}}(\mathcal F)\;+\;\sqrt{\frac{\log(2/\delta)}{2m}}.$$

So a Lipschitz, margin-based loss costs me exactly a factor $2L=2/\gamma$ in front of the *raw* structural class complexity — and that raw complexity I can attack directly. This is the lever that turns the abstract bound into something usable for the classes people actually train.

And it composes. The whole reason to want contraction plus a couple of algebraic identities is that real classes are *built* from simpler ones, and I'd like a calculus: bound the complexity of the composite in terms of the pieces. Let me collect the moves, each of which I can check in a line. Monotonicity: $\mathcal F\subseteq\mathcal H\Rightarrow R_m^{\mathrm{str}}(\mathcal F)\le R_m^{\mathrm{str}}(\mathcal H)$, immediate from the sup. Scaling: $R_m^{\mathrm{str}}(c\mathcal F)=|c|R_m^{\mathrm{str}}(\mathcal F)$, pull $|c|$ out of the sup. Convex hull: $R_m^{\mathrm{str}}(\mathrm{conv}\,\mathcal F)=R_m^{\mathrm{str}}(\mathcal F)$ — because a linear functional over a convex hull is maximized at an extreme point, so taking the hull adds nothing; this is what makes *voting methods* tractable, since a boosted ensemble lives in the convex hull of base classifiers and I get its complexity from the base class for free. Sum: $R_m^{\mathrm{str}}(\sum_j\mathcal F_j)\le\sum_j R_m^{\mathrm{str}}(\mathcal F_j)$, triangle inequality on the sup (and it's tight when the classes coincide, so I can't hope for better in general). Translation by a fixed bounded $h$: $R_m^{\mathrm{str}}(\mathcal F+h)\le R_m^{\mathrm{str}}(\mathcal F)+\|h\|_\infty/\sqrt m$, since the fixed part is controlled by Jensen. With these plus contraction I can walk a decision tree (a fixed Boolean function of node tests), a two-layer network (Lipschitz squashing of linear combinations), or a kernel machine down to the complexity of its primitive pieces. For the kernel case specifically: the class $\{x\mapsto\langle w,\Phi(x)\rangle:\|w\|\le B\}$ has, by Cauchy–Schwarz inside the sup and Jensen,

$$\hat R_S(\mathcal F)\le\frac{B}{m}\Bigl(\sum_i k(X_i,X_i)\Bigr)^{1/2},$$

the trace of the Gram matrix — a quantity sitting right there in the kernel computation. If I write the same statement in the absolute $2/m$ structural convention, the displayed constant becomes $2B/m$; the one-sided $1/m$ oracle I will code below uses the $B/m$ normalization. Capacity becomes something I read off the kernel matrix.

Let me look back at the rate, because something is still unsatisfying. Every bound above carries $1/\sqrt m$ — the confidence term, and the typical size of $R_m$. But consider the *realizable* case: the target $f_0$ is in the class and a consistent estimate $\hat f_m$ fits the sample exactly, $\hat{E}_S|\hat f_m-f_0|=0$. Then the risk is the *whole* gap, with no empirical term to hide behind, and I know from classical theory it should decay like $\log m/m$ for a VC class, not $1/\sqrt m$. My global Rademacher bound can't see that — it measures the capacity of the *entire* class, including functions far from $f_0$ that are irrelevant once I've fit the data. The functions that actually matter are the ones *close* to the truth, with small variance $Pf^2\le r$. So I should localize: restrict the noise-correlation to the slice of the class inside a ball $B(r)=\{f:Pf^2\le r\}$ and use the *local* Rademacher norm $\|R_m\|_{\mathcal F\cap B(r)}$, which shrinks as $r\to0$.

The catch: a consistent estimate's own risk $r$ is what I'm trying to bound, so the ball that contains the relevant functions is defined by the quantity I don't yet know — circular. The way out is a fixed-point / iteration. Start with $r_0=1$ (everything), and note any consistent $\hat f_m$ has $P\hat f_m\le\|P_m-P\|_{\mathcal F\cap B(r_0)}=:r_1$; but then $\hat f_m\in B(r_1)$, so actually $P\hat f_m\le\|P_m-P\|_{\mathcal F\cap B(r_1)}=:r_2$, and I can keep tightening: $r_{k+1}=\|P_m-P\|_{\mathcal F\cap B(r_k)}$, a non-increasing sequence squeezing toward the true risk. To make it data-dependent I replace each $\|P_m-P\|$ by its Rademacher surrogate (symmetrization again, now localized: $E\|P_m-P\|_{\mathcal F\cap B(r)}\le2E\|R_m\|_{\mathcal F\cap B(r)}$) plus a concentration slack. But ordinary McDiarmid is too blunt here — I need the slack to *shrink with the variance* $r$, otherwise localization buys nothing. That's precisely what Talagrand's concentration inequality for the supremum of an empirical process gives: a deviation term governed by $\sigma^2=\sup_{\mathcal F\cap B(r)} m\,\mathrm{Var}(f)\le mr$, not by the crude range. With Massart's explicit constants the recursion becomes

$$\bar r_{k+1}=\bar\varphi(\bar r_k),\qquad\bar\varphi(r)=\bar K_1\|R_m\|_{\mathcal F\cap B(2r)}+\bar K_2\sqrt{r\varepsilon}+\bar K_3\varepsilon,$$

a non-increasing sequence whose fixed point $\delta$ solves $\delta=m^{-1/2}\psi(\sqrt\delta)$ with $\psi$ the random entropy integral of the class. The remarkable thing the iteration reveals: it converges to within a constant of $\delta$ after only $N\approx\log\log(1/\varepsilon)$ steps, because the gap to the fixed point contracts quadratically ($d_{k+1}\le\sqrt{\delta\,d_k}$). And $\delta$ is exactly the local rate — for a VC class it comes out $O(V(\mathcal H)\log m/m)$, the near-minimax fast rate. So localization recovers $1/m$ in the realizable case while staying fully data-dependent: the penalty is still just a Rademacher norm, only now evaluated on the relevant slice of the class.

Let me trace the causal chain once, to be sure it holds end to end. The thing that hurts is the uniform gap $\sup_f(Pf-\hat E_S f)$; worst-case combinatorial penalties bound it but ignore the sample. I measure capacity instead by how well the class correlates with random $\pm1$ noise on the *realized* points — the empirical Rademacher complexity $\hat R_S(\mathcal F)$ — which is data-dependent by construction. A ghost-sample symmetrization, where the $\pm1$ signs appear as the exchange symmetry between real and ghost points, proves $E\sup_f(Pf-\hat E_S f)\le2R_m(\mathcal F)$. McDiarmid, because moving one point shifts the supremum by $\le1/m$, upgrades the expectation to a high-probability bound; a second one-sided McDiarmid substitution turns $R_m$ into the realized $\hat R_S$, yielding $Pf\le\hat E_S f+2\hat R_S(\mathcal F)+3\sqrt{\log(2/\delta)/2m}$. Massart's finite-class lemma, through the growth function and Sauer, recovers the VC bound as a special case — so this is never much worse than VC and, because it reads the realized sign-patterns, often much tighter. The Ledoux–Talagrand contraction lets a Lipschitz margin loss be peeled at cost $2L$ in the structural norm, and a short calculus (hull, sum, scaling, translation) reduces composite classes to their primitives, including the kernel trace bound. Finally, localizing the Rademacher norm to a variance ball and iterating to its fixed point — using Talagrand's variance-sensitive concentration — sharpens the rate to the fast $1/m$ in the realizable case. The capacity-to-correlate-with-noise, measured on the data, governs generalization throughout.

So the bound I can carry forward is now explicit. For a class $\mathcal F$ of functions into $[0,1]$, with empirical Rademacher complexity

$$\hat R_S(\mathcal F)=E_\sigma\sup_{f\in\mathcal F}\frac1m\sum_{i=1}^m\sigma_i f(z_i),\qquad R_m(\mathcal F)=E_S\hat R_S(\mathcal F),$$

with probability at least $1-\delta$ over the sample, every $f\in\mathcal F$ satisfies

$$E_P f\;\le\;\frac1m\sum_{i=1}^m f(z_i)\;+\;2\,R_m(\mathcal F)\;+\;\sqrt{\frac{\log(1/\delta)}{2m}},$$

and the fully data-dependent form

$$E_P f\;\le\;\frac1m\sum_{i=1}^m f(z_i)\;+\;2\,\hat R_S(\mathcal F)\;+\;3\sqrt{\frac{\log(2/\delta)}{2m}}.$$

For $\{\pm1\}$ classification the loss-class reduction gives $P(Y\neq h(X))\le\hat P_m(Y\neq h(X))+R_m(\mathcal H)+\sqrt{\log(1/\delta)/2m}$, and its empirical version replaces $R_m(\mathcal H)$ by $\hat R_{S_X}(\mathcal H)$ and uses the coefficient-$3$ confidence term. For a margin loss $\phi$ that is $L$-Lipschitz and dominates the step, contraction in the structural norm gives $P(Yf(X)\le0)\le\hat E_m\phi(Yf(X))+2L\,R_m^{\mathrm{str}}(\mathcal F)+\sqrt{\log(2/\delta)/2m}$.

And because the penalty is an expectation over coin flips of a maximization I can run on the sample, I can estimate it directly. The proof gives a one-line algorithm:

```python
import numpy as np

def empirical_rademacher(corr_oracle, S, n_draws=1000, rng=None):
    """Estimate R_hat_S(F) = E_sigma [ sup_{f in F} (1/m) sum_i sigma_i f(z_i) ].

    corr_oracle(sigma, S) must return  sup_{f in F} (1/m) * sum_i sigma_i f(z_i)
    for a given sign vector sigma -- i.e. it maximizes the noise-correlation
    over the class on this fixed sample (an ERM-style call, the only hard part).
    """
    if rng is None:
        rng = np.random.default_rng()
    m = len(S)
    vals = []
    for _ in range(n_draws):
        sigma = rng.choice([-1.0, 1.0], size=m)   # fresh Rademacher noise
        vals.append(corr_oracle(sigma, S))        # best correlation with it
    return float(np.mean(vals))                   # average over the coins

def generalization_bound(emp_risk, F_corr_oracle, S, delta, n_draws=1000):
    m = len(S)
    R_hat = empirical_rademacher(F_corr_oracle, S, n_draws)   # the penalty
    conf  = 3.0 * np.sqrt(np.log(2.0 / delta) / (2.0 * m))
    return emp_risk + 2.0 * R_hat + conf                      # the bound

# For a finite/linear class the oracle is explicit. E.g. linear predictors
# {x -> <w,x> : ||w||_2 <= B}: sup_w (1/m) sum_i sigma_i <w,x_i>
#   = (B/m) || sum_i sigma_i x_i ||_2,   maximized by w aligned with the sum.
def linear_l2_corr_oracle(B):
    def oracle(sigma, X):                 # X: (m,d) array of features
        m = X.shape[0]
        return (B / m) * np.linalg.norm(sigma @ X)   # = (B/m)||sum_i sigma_i x_i||
    return oracle
```
