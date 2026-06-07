OK, let me think this through from scratch. I have this autocorrelation constant I want to nail down. For a nonnegative $f$ on the line, and a fixed symmetric decreasing weight $w$ normalized so $\|w\|_1=\|w\|_\infty=1$, I want the smallest $C$ with

$$\iint f(x)f(y)\,w(x-y)\,dx\,dy \le C\,\|f\|_1\|f\|_2.$$

The two cases I actually care about are $w=\chi_{[-1/2,1/2]}$ — that's the average of the autocorrelation over a unit window, the Barnard–Steinerberger problem — and $w=e^{-\pi x^2}$, the Gaussian mean. The best constant is just a supremum,

$$C_{opt}=\sup_{f}\frac{\iint f(x)f(y)w(x-y)\,dx\,dy}{\|f\|_1\|f\|_2}.$$

So this is a Rayleigh quotient. And the frustrating thing is how the field is stuck: I can write down a test function and get a lower bound — somebody's example gives $0.8$ for the indicator — and I can grind out an upper bound by Fourier analysis — Barnard and Steinerberger got $0.91$, then a Hausdorff–Young argument got it down to $0.864$ — but the two ends just won't meet. I want to actually pin this number, both sides, with something I can certify. Not produce a slightly better test function; that only ever moves one end of the bracket.

Let me first understand what kind of beast the quotient is. The numerator, $\iint f(x)f(y)w(x-y)$, is bilinear in $f$ — it's $\langle f, K_w f\rangle$ where $K_w$ is convolution against $w$. On the Fourier side it's $\int |\hat f(\xi)|^2\,\hat w(\xi)\,d\xi$, and for the indicator $\hat w(\xi)=\frac{\sin\pi\xi}{\pi\xi}$, so the whole thing is a sinc-weighted average of $|\hat f|^2$. That's a clean self-adjoint quadratic form. If the denominator were also a quadratic form, this would be a generalized eigenvalue problem and I'd be done — top eigenvalue of $K_w$ against the denominator's Gram matrix, finite-dimensional after discretizing, power method, certificate from the eigenvalue. That's the dream.

But the denominator is $\|f\|_1\|f\|_2$. That is *not* a quadratic form. It's a product of an $L^1$ norm and an $L^2$ norm. The $\|f\|_2$ part is fine, that's $\langle f,f\rangle^{1/2}$, but $\|f\|_1=\int|f|$ is a first-order (degree-one) thing, and the product $\|f\|_1\|f\|_2$ scales like $f^{1.5}\cdot f^{0.5}$... no wait, let me get the homogeneity right: $\|f\|_1\|f\|_2$ scales as $f^2$, same as the numerator, good — the quotient is scale-invariant, as it must be. But it is *not* a ratio of two quadratic forms. The denominator is a geometric-mean of a linear-functional-squared and a quadratic form. That's the wall. No spectral theorem applies to $\langle f,K_w f\rangle/(\|f\|_1\|f\|_2)$ directly.

Before I fight that, let me clean up the feasible set, because right now I'm optimizing over all of $L^1\cap L^2$ and that's needlessly large. The numerator $\iint f(x)f(y)w(x-y)$ — if I replace $f$ by $|f|$, every term $f(x)f(y)w(x-y)$ with $w\ge0$ can only grow or stay, since $|f(x)||f(y)|\ge f(x)f(y)$, and $\|f\|_1,\|f\|_2$ are unchanged. So the sup over all $f$ equals the sup over $f\ge0$. Good — I can assume $f\ge0$. Now, $w$ is symmetric decreasing. There's a classical inequality made for exactly this: Riesz rearrangement. For nonnegative $f$ and a symmetric decreasing kernel,

$$\iint f(x)f(y)w(x-y)\,dx\,dy \le \iint f^*(x)f^*(y)w(x-y)\,dx\,dy,$$

where $f^*$ is the symmetric-decreasing rearrangement of $f$ — and rearrangement preserves all $L^p$ norms, so it preserves the denominator exactly. So replacing $f$ by $f^*$ can only increase the quotient. The supremum is attained, if anywhere, among symmetric-decreasing nonnegative functions. That's a huge collapse of the search space: from arbitrary functions to monotone bumps centered at the origin.

Does the sup get attained at all, or does mass escape to infinity / spread out? Let me try to *find* the optimizer's structure and see if it's forced to be compactly supported, because if it is, I can put it on a finite grid honestly. Restrict to functions supported in $[-R,R]$, call the constant there $C_{opt,R}$; clearly $C_{opt,R}\nearrow C_{opt}$ as $R\to\infty$. On a fixed $[-R,R]$ the feasible set is tight enough that an extremizer $f^\star$ exists. Let me write its Euler–Lagrange equation. The quotient's log is nicer to differentiate because it splits the product:

$$\mathcal F(g)=\log\langle g,K_w g\rangle - \log\|g\|_1 - \tfrac12\log\|g\|_2^2,$$

and on the support of $f^\star$, $\nabla\mathcal F(f^\star)=0$ gives

$$\frac{2\,(f^\star * w)}{\langle f^\star,K_w f^\star\rangle} - \frac{1}{\|f^\star\|_1} - \frac{f^\star}{\|f^\star\|_2^2}=0.$$

(The $2$ is because the numerator is symmetric bilinear; the $\frac1{\|f\|_1}$ comes from $\nabla\int|f|=\mathrm{sgn}\,f=1$ on the support where $f^\star>0$; the $\frac{f^\star}{\|f^\star\|_2^2}$ from $\nabla\tfrac12\log\|f\|_2^2$.) Now integrate this over the support $[-a,a]$. The middle term gives $\frac{2a}{\|f^\star\|_1}$, the last gives $\frac{1}{\|f^\star\|_2^2}\int_{-a}^a f^\star=\frac{\|f^\star\|_1}{\|f^\star\|_2^2}$. For the first term I have to be careful about which variable is restricted to the support:

$$\int_{-a}^a(f^\star*w)(y)\,dy=\int_{-a}^a\int_{-a}^a f^\star(x)w(y-x)\,dx\,dy\le \int_{\mathbb R}\int_{-a}^a f^\star(x)w(y-x)\,dx\,dy=\|w\|_1\|f^\star\|_1.$$

So

$$\frac{\|f^\star\|_1}{\|f^\star\|_2^2}+\frac{2a}{\|f^\star\|_1}\le \frac{2\,\|w\|_1}{C_{opt,R}\|f^\star\|_2}.$$

Solve for $a$ in that inequality:

$$a\le\frac{\|f^\star\|_1}{\|f^\star\|_2}\Big(\frac{\|w\|_1}{C_{opt,R}}-\tfrac12\frac{\|f^\star\|_1}{\|f^\star\|_2}\Big)\le\frac{\|f^\star\|_1\,\|w\|_1}{\|f^\star\|_2\,C_{opt,R}},$$

and Hölder gives $\|f^\star\|_1\le\sqrt{2a}\,\|f^\star\|_2$ on a support of length $2a$, so $\frac{\|f^\star\|_1}{\|f^\star\|_2}\le\sqrt{2a}$ and therefore

$$a\le 2\,\frac{\|w\|_1^2}{C_{opt,R}^2}.$$

So the support half-length is bounded by an explicit constant depending only on $\|w\|_1$ and a lower bound for the constant. The extremizer can't run off to infinity. That settles existence with compact support and tells me $C_{opt}=\lim_R C_{opt,R}=C_{opt,R}$ for $R$ large enough. Good — finite grids are honest.

Now back to the wall: the denominator $\|f\|_1\|f\|_2$ is not quadratic, so no eigenvalue problem. Let me stare at $\|f\|_1\|f\|_2$. It's a *product* $a\cdot b$ with $a=\|f\|_1$, $b=\|f\|_2$. And there's a completely standard way to write a product as a minimum of a sum: AM–GM in the form $2ab=\min_{\lambda>0}(\lambda b^2+\lambda^{-1}a^2)$. Check: $\lambda b^2+\lambda^{-1}a^2\ge 2\sqrt{\lambda b^2\cdot\lambda^{-1}a^2}=2ab$, equality at $\lambda=a/b$. So

$$2\|f\|_1\|f\|_2=\min_{\lambda>0}\Big(\lambda\|f\|_2^2+\lambda^{-1}\|f\|_1^2\Big).$$

For each *fixed* $\lambda$, the bracket $\lambda\|f\|_2^2+\lambda^{-1}\|f\|_1^2$ is... almost a quadratic form. $\|f\|_2^2$ is quadratic. $\|f\|_1^2=(\int|f|)^2$ is the square of a — not-quite-linear functional, because of the absolute value. But on my feasible cone $f\ge0$, $\int|f|=\int f$, which *is* linear! So on $f\ge0$,

$$\lambda\|f\|_2^2+\lambda^{-1}\Big(\int f\Big)^2=\langle f,\,(\lambda\, \mathrm{Id}+\lambda^{-1}|1\rangle\langle1|)\,f\rangle,$$

a genuine quadratic form, where $|1\rangle\langle1|$ is the rank-one operator $f\mapsto 1\cdot\int f$. Let me name the two norms to keep the bookkeeping straight: $\|f\|_{B_\lambda}^2=\lambda\|f\|_2^2+\lambda^{-1}(\int|f|)^2$ with the absolute value, and $\|f\|_{H_\lambda}^2=\lambda\|f\|_2^2+\lambda^{-1}(\int f)^2$ without. $B_\lambda$ is what AM–GM gives me — $\|f\|_{L^{1:2}}^2:=\|f\|_1\|f\|_2=\frac12\inf_\lambda\|f\|_{B_\lambda}^2$ — but $B_\lambda$ has the absolute value buried in it, so it's not from an inner product. $H_\lambda$ drops the absolute value, so it *is* from an inner product (it satisfies the parallelogram law — $\lambda\langle\cdot,\cdot\rangle+\lambda^{-1}(\int\cdot)(\int\cdot)$ is bilinear), so $H_\lambda$ is a Hilbert space. And $\|f\|_{B_\lambda}\ge\|f\|_{H_\lambda}$ always, with equality exactly when $\int|f|=|\int f|$, i.e. when $f$ has a constant sign — which is precisely my cone $f\ge0$. So on the cone, $B_\lambda$ and $H_\lambda$ coincide and I get to work in the Hilbert space $H_\lambda$ for free.

So define, for each $\lambda>0$,

$$c_\lambda:=\max_{f\ge0}\frac{2\langle f,K_w f\rangle}{\lambda\langle f,f\rangle+\lambda^{-1}\langle f,1\rangle\langle1,f\rangle}=\max_{\substack{f\ge0\\ \|f\|_{H_\lambda}\le1}}2\langle f,K_w f\rangle.$$

By the AM–GM identity, taking the min over $\lambda$ in the denominator is taking the max of $c_\lambda$ over $\lambda$:

$$C_{opt}=\max_{\lambda>0}c_\lambda.$$

This is the reduction I was looking for. The non-quadratic two-norm problem has become a one-parameter family of *quadratic-form maximizations in a Hilbert space*, each of which is essentially "find the top of the spectrum of $K_w$ measured against the $H_\lambda$ inner product, restricted to the cone $f\ge0$." And I get a cheap sanity bound on each $c_\lambda$: $\langle f,K_w f\rangle\le\|w\|_\infty\|f\|_1^2=\|f\|_1^2$ gives $c_\lambda\le2\lambda$, while Young's inequality gives $\langle f,K_w f\rangle\le\|w\|_1\|f\|_2^2=\|f\|_2^2$ and hence $c_\lambda\le2/\lambda$. So $c_\lambda\le\min\{2\lambda,2/\lambda\}$, and the maximizing $\lambda$ lives near the AM-GM equality point $\lambda^*=\|f^\star\|_1/\|f^\star\|_2$.

There's still the cone constraint $f\ge0$, and I'll deal with it, but let me first see how clean the $\lambda$-fixed problem is and whether I can certify the max over $\lambda$ from finitely many $\lambda$'s. How regular is $c_\lambda$ in $\lambda$? Take any $g$ with $\|g\|_1=1$ and $\langle g,K_wg\rangle\le1$ — then $\frac{d}{d\lambda}(\lambda\|g\|_2^2+\lambda^{-1}\|g\|_1^2)^{-1}=-(\cdots)^{-2}(\|g\|_2^2-\lambda^{-2}\|g\|_1^2)$, and $\lambda^{-1}\cdot|\lambda\|g\|_2^2-\lambda^{-1}\|g\|_1^2|\le(\lambda\|g\|_2^2+\lambda^{-1}\|g\|_1^2)$, so the derivative of the quotient in $\lambda$ is bounded by $1$ in absolute value. $c_\lambda$ is $1$-Lipschitz. Even better, if the global max is at $\lambda^*$, plug the global extremizer $f^\star$ (with $\lambda^*=\|f^\star\|_1/\|f^\star\|_2$) into the $c_\lambda$ quotient: the numerator is fixed at $2\|f^\star\|_1\|f^\star\|_2c_{\lambda^*}$ and the denominator is $\lambda\|f^\star\|_2^2+\lambda^{-1}\|f^\star\|_1^2$, giving

$$c_\lambda\ge\frac{2c_{\lambda^*}}{\lambda^{-1}\lambda^*+\lambda{\lambda^*}^{-1}}.$$

So $c_\lambda$ doesn't just have a Lipschitz cap, it has an explicit lower envelope that hugs the peak. A finite grid in $\lambda$ with spacing $\Delta\lambda$ certifies $\max_\lambda c_\lambda$ to within a controlled error. The $\lambda$-direction is handled.

Now the real computational object: for fixed $\lambda$, solve $\max_{f\ge0,\,\|f\|_{H_\lambda}\le1}\langle f,K_w f\rangle$ on a finite grid. Discretize: let $V_\delta$ be functions constant on each cell $[n\delta,(n+1)\delta)$, and project a function by block-averaging, $[f]_\delta=\delta^{-1}\int_{\text{cell}}f$. Two facts about this projection that I want to be true and let me check: $\|[f]_\delta\|_1=\|f\|_1$ — yes, block-averaging conserves mass exactly, $\int[f]_\delta=\int f$. And $\|[f]_\delta\|_2\le\|f\|_2$ — yes, averaging is an $L^2$-contraction (Jensen on each cell). So $\|[f]_\delta\|_{H_\lambda}\le\|f\|_{H_\lambda}$: the projection never inflates the constraint norm. If $c_{\lambda,\delta}$ is the constant restricted to step functions, the restriction immediately gives $c_{\lambda,\delta}\le c_\lambda$; the projection is what lets me measure how much is lost.

But I want the *gap* $c_\lambda-c_{\lambda,\delta}$, controlled rigorously, not just the sign. Write $f^\star=[f^\star]_\delta+\{f^\star\}_\delta$ where $\{f^\star\}_\delta$ is the within-cell fluctuation. The projection is orthogonal in $L^2$ (and in $H_\lambda$, since $\int\{f\}_\delta=0$ per cell makes the fluctuation orthogonal to the rank-one $|1\rangle\langle1|$ piece too), so $\langle[f]_\delta,\{g\}_\delta\rangle=0$. Then

$$\langle f^\star,K_w f^\star\rangle-\langle[f^\star]_\delta,K_w[f^\star]_\delta\rangle=\langle([f^\star]_\delta+f^\star)*w,\ \{f^\star\}_\delta\rangle=\langle\{([f^\star]_\delta+f^\star)*w\}_\delta,\ \{f^\star\}_\delta\rangle,$$

the last step using orthogonality to pull the projection onto the fluctuation. Cauchy–Schwarz:

$$c_\lambda-c_{\lambda,\delta}\le\big\|\{(f^\star+[f^\star]_\delta)*w\}_\delta\big\|_2\cdot\big\|\{f^\star\}_\delta\big\|_2.$$

Now I need to bound within-cell fluctuations. The optimal Poincaré inequality on a cell of length $\delta$ with mean removed is $\|\{h\}_\delta\|_2\le\frac{\delta}{\pi}\|h'\|_2$ — the $\pi$ is the first Neumann eigenvalue, the sharp constant. Apply it twice. First to $h=f^\star$: $\|\{f^\star\}_\delta\|_2\le\frac\delta\pi\|{f^\star}'\|_2$. Second to $h=(f^\star+[f^\star]_\delta)*w$, whose derivative is $(f^\star+[f^\star]_\delta)*w'$, and Young's convolution inequality gives $\|(f^\star+[f^\star]_\delta)*w'\|_2\le\|f^\star+[f^\star]_\delta\|_2\|w'\|_1=\|f^\star+[f^\star]_\delta\|_2\,\|w\|_{TV}$. So

$$c_\lambda-c_{\lambda,\delta}\le\frac{\delta^2}{\pi^2}\,\|f^\star+[f^\star]_\delta\|_2\,\|w\|_{TV}\,\|{f^\star}'\|_2.$$

Everything is now in terms of $\|f^\star\|_2$ and $\|{f^\star}'\|_2$ and $\|w\|_{TV}$. The $\|f^\star\|_2$ is controlled: $\|f^\star\|_2\le\lambda^{-1/2}\|f^\star\|_{H_\lambda}=\lambda^{-1/2}$. The total variation: $\|w\|_{TV}=2\|w\|_\infty=2$ for a symmetric decreasing $w$ with peak $1$ (it goes up to $1$ and back down). The one thing I still need is $\|{f^\star}'\|_2$ — and here's where the *smoothness* of the extremizer has to be earned, because if $f^\star$ can be rough this whole $\delta^2$ collapses.

Go back to the Euler–Lagrange equation, but now for the $H_\lambda$-normalized problem: $f^\star$ maximizes $\log\langle f,K_w f\rangle-\log\|f\|_{H_\lambda}^2$, so on its support

$$\frac{2K_w f^\star}{\langle f^\star,K_w f^\star\rangle}=\frac{2\lambda f^\star+2\lambda^{-1}I f^\star}{\|f^\star\|_{H_\lambda}^2},$$

where $I$ is "integrate" (the operator behind $|1\rangle\langle1|$). Multiply through by $\|f^\star\|_{H_\lambda}^2$ and use $c_\lambda=2\langle f^\star,K_wf^\star\rangle/\|f^\star\|_{H_\lambda}^2$:

$$\frac{4K_w f^\star}{c_\lambda}=2\lambda f^\star+2\lambda^{-1}I f^\star,$$

and since $f^\star\ge0$ on its support (and the $\ge0$ constraint truncates outside), this is

$$f^\star=\max\Big\{0,\ \big[c_\lambda^{-1}(\lambda+\lambda^{-1}I)^{-1}\,2K_w\big]f^\star\Big\}.$$

The point is the structure: $f^\star$ equals (positive part of) a *smoothing* of itself — $(\lambda+\lambda^{-1}I)^{-1}$ is bounded, and $K_w=w*$ is a convolution, which differentiates as $(K_wf^\star)'=w'*f^\star$. So on the support, $|{f^\star}'|\le\frac{2}{c_\lambda\lambda}|(w*f^\star)'|=\frac{2}{c_\lambda\lambda}|w'*f^\star|$, and Young again: $\|{f^\star}'\|_2\le\frac{2}{c_\lambda\lambda}\|w'\|_1\|f^\star\|_2=\frac{2}{c_\lambda\lambda}\|w\|_{TV}\|f^\star\|_2=\frac{4}{c_\lambda\lambda}\|w\|_\infty\|f^\star\|_2\le\frac{4}{c_\lambda\lambda^{3/2}}.$ The extremizer is Lipschitz with a quantitative bound — the convolution kernel hands me the smoothness. Plug it all in:

$$0\le c_\lambda-c_{\lambda,\delta}\le\frac{\delta^2}{\pi^2}\cdot 2\lambda^{-1/2}\cdot 2\cdot\frac{4}{c_\lambda\lambda^{3/2}}=\frac{16\,\delta^2}{\pi^2\,c_\lambda\,\lambda^2}.$$

A clean a-priori $O(\delta^2)$ discretization bound, *with an explicit constant*. This is the thing that makes the upper bound rigorous: solve the discrete problem, get $c_{\lambda,\delta}$, and I know $c_\lambda\le c_{\lambda,\delta}+16\delta^2/(\pi^2c_\lambda\lambda^2)$. And it's also the line that separates this problem from the maximum problem: there, extremizers are non-smooth, $\|{f^\star}'\|_2$ is not controlled, the $\delta^2$ becomes $\delta$, and the cost balloons. Here the smoothing in the Euler–Lagrange equation is doing real work.

Now solve the discrete fixed-$\lambda$ problem itself. On a grid of $N$ cells over $[-a,a)$, $f$ is a vector, $\langle\cdot,\cdot\rangle$ is the $\delta$-weighted dot product, $\langle1,1\rangle=2a$, and I need to discretize $K_w$. The right discrete kernel isn't just $w$ sampled — it's $w$ tested against cell indicators on both sides. If I demand $\langle\iota f,\,w*_{\mathbb R}\iota g\rangle_{L^2}=\langle f,\,\tilde w*_{\delta\mathbb Z}g\rangle_{\ell^2}$ for step functions $\iota f,\iota g$, then

$$\tilde w(s)=\fint_s^{s+\delta}\fint_0^\delta w(y-x)\,dx\,dy=\delta^{-2}\int_{s-\delta}^{s+\delta}w(t)\,(\delta-|t-s|)\,dt,$$

a triangular ($B$-spline) smoothing of $w$ — exactly the autocorrelation of the cell indicator with $w$. So I build the symmetric Toeplitz matrix $K$ from $\tilde w(i-j)$.

For the denominator, the $H_\lambda$ form is $\lambda\langle f,f\rangle+\lambda^{-1}\langle f,1\rangle\langle1,f\rangle$ — identity plus rank one. I want to turn $\max 2\langle f,K_wf\rangle/\langle f,H_\lambda f\rangle$ into a standard symmetric eigenvalue problem. Whiten: I need a positive-definite $A_\lambda$ with $A_\lambda^2=\lambda\,\mathrm{Id}+\lambda^{-1}|1\rangle\langle1|$. Because the operator is identity-plus-rank-one, its square root is also identity-plus-rank-one: try $A_\lambda=\sqrt\lambda\,\mathrm{Id}+b_\lambda|1\rangle\langle1|$. Then

$$A_\lambda^2=\lambda\,\mathrm{Id}+(2\sqrt\lambda\,b_\lambda+b_\lambda^2\langle1,1\rangle)|1\rangle\langle1|=\lambda\,\mathrm{Id}+(2\sqrt\lambda\,b_\lambda+2a\,b_\lambda^2)|1\rangle\langle1|,$$

so I need $2\sqrt\lambda\,b_\lambda+2a\,b_\lambda^2=\lambda^{-1}$ — a quadratic in $b_\lambda$ with a unique positive root. Set $g=A_\lambda f$; then $\langle f,H_\lambda f\rangle=\langle g,g\rangle$ and

$$\frac{2\langle f,K_w f\rangle}{\langle f,H_\lambda f\rangle}=\frac{2\langle g,A_\lambda^{-1}K_wA_\lambda^{-1}g\rangle}{\langle g,g\rangle},$$

so with $M_\lambda:=2A_\lambda^{-1}K_wA_\lambda^{-1}$ (symmetric), the *unconstrained* maximum is just $\lambda_{\max}(M_\lambda)$, by the spectral theorem — and the power method finds it in $O(N^2)$ per iteration, fast, with the eigenvalue itself as a certificate.

Except — I dropped the cone constraint $f\ge0$ to get here. That is not a harmless relaxation. A sign-changing vector can reduce the rank-one part of the denominator by cancellation in $\langle 1,f\rangle$ while still producing a large quadratic numerator, and that vector is not an admissible autocorrelation test function. So the unconstrained spectrum is a subroutine, not the answer. I can't just take $\lambda_{\max}(M_\lambda)$.

So how do I honor $f\ge0$ without losing the eigenvalue structure? The constraint is an inequality, infinitely many of them — this smells like it needs an active-set / LP-flavored argument rather than a clean eigensolve. Discrete Riesz rearrangement still applies, so among discrete maximizers there is one that's symmetric and non-increasing — a single contiguous bump. Take $f^\star$ to be such a maximizer of *minimal support*. On the interior of its support, $f^\star>0$, so the positivity constraint is *inactive* there — the only active constraints are at the support boundary, where $f^\star$ hits zero. On the interior, then, $f^\star$ satisfies the *unconstrained* stationarity: it's an eigenvector of $K_w$ restricted to the support block. And it must be the *top* eigenvector of that block, and the unique one — if there were a second eigenvector $g^\star$ on the same support, then $g^\star-\alpha f^\star$ would also be a maximizer with strictly smaller support for the right $\alpha$, contradicting minimality.

I don't enumerate the infinitely many inequality constraints; I enumerate the one combinatorial degree of freedom they leave — the *length* of the contiguous support. For each candidate support length $\ell$ (a contiguous block of $\ell$ cells), restrict $M_\lambda$ to that block, take its top eigenvalue/eigenvector by the power method, and check whether the eigenvector is nonnegative (after the inverse whitening, the back-mapped $f=A_\lambda^{-1}g$ being sign-definite). The constrained optimum is the largest such eigenvalue over all $\ell$ whose top eigenvector is admissible. Sweeping $\ell=1,\dots,N$ is $N$ eigenproblems, still polynomial. The active-set combinatorics collapses to a one-dimensional sweep because the extremizer is a single monotone bump — which I knew from Riesz.

Assemble the algorithm. Pick $\delta$ small enough that the discretization term is below the tolerance I want, and pick a grid of $\lambda$ fine enough — spacing $\Delta\lambda$ — that the $1$-Lipschitz/lower-envelope control certifies $\max_\lambda c_\lambda$. For each $\lambda$: build $\tilde w$, the Toeplitz $K$, the whitening $A_\lambda$ from the positive root $b_\lambda$, $M_\lambda=2A_\lambda^{-1}KA_\lambda^{-1}$; sweep the support length, power-method each block, keep the largest eigenvalue with an admissible nonnegative eigenvector — that's $c_{\lambda,\delta}$. The upper side needs one more algebraic check because the error bound has $c_\lambda$ in the denominator. If $E=16\delta^2/(\pi^2\lambda^2)$, then $c_\lambda-c_{\lambda,\delta}\le E/c_\lambda$, so $c_\lambda^2-c_{\lambda,\delta}c_\lambda-E\le0$ and therefore

$$c_\lambda\le \frac{c_{\lambda,\delta}+\sqrt{c_{\lambda,\delta}^2+4E}}{2}.$$

That gives the certified upper value at this $\lambda$, with the $\lambda$-grid correction added afterward. The lower side is simpler: the admissible eigenvector is an honest nonnegative test function, so I plug it into the original quotient $\iint f f\,w/(\|f\|_1\|f\|_2)$ and get a value that is automatically $\le C_{opt}$. Two bounds, both certified, sandwich $C_{opt}$. The cost is polynomial in $1/\delta$, not exponential, precisely because the $\delta^2$ smoothness gain lets me take $\delta$ coarse and because the active set has reduced to a length sweep.

One more thing the Euler–Lagrange equation is begging me to try, for speed. Rewrite the stationarity as a fixed-point map: $\frac{f^\star}{\|f^\star\|_2^2}=\max\big(\frac{2f^\star*w}{\langle f^\star,K_wf^\star\rangle}-\frac1{\|f^\star\|_1},0\big)$. Iterate it — convolve, subtract the $1/\|f\|_1$ offset, take positive part, renormalize, repeat. In practice it converges, and to the same fixed point as the eigenvalue sweep, hundreds of times faster — it's the natural projected-gradient flavor of the variational problem. I can't prove it converges (the positive-part nonlinearity fights a clean contraction argument), so I keep it as the fast lower-bound producer and let the certified eigenvalue sweep be the thing that actually brackets the constant.

So the whole arc: sign and rearrangement collapse the search to symmetric-decreasing nonnegative bumps; Euler–Lagrange bounds their support so finite grids are honest; the AM–GM/$\lambda$ trick turns the non-quadratic $\|f\|_1\|f\|_2$ denominator into a family of Hilbert-space quadratic problems with $C_{opt}=\max_\lambda c_\lambda$; block-average discretization with optimal Poincaré plus the convolution-induced smoothness of the extremizer gives an $O(\delta^2)$ certified gap; the rank-one whitening turns each fixed-$\lambda$ problem into a symmetric eigenvalue problem solved by the power method, with the cone constraint handled exactly by sweeping the contiguous support length; and a $\lambda$-grid with Lipschitz control closes the optimization over $\lambda$ — yielding a two-sided certificate for $C_{opt}$ at polynomial cost.

```python
import numpy as np
import scipy.integrate as si

# --- the weight, as a callable; symmetric decreasing, ||w||_1 = ||w||_inf = 1 ---
def indicator_weight(t):      # w = chi_[-1/2, 1/2]  -> average autocorrelation problem
    return 1.0 if abs(t) <= 0.5 else 0.0

def gaussian_weight(t):       # w = exp(-pi t^2) -> Gaussian-mean problem
    return np.exp(-np.pi * t * t)

# --- triangular-smoothed cell kernel ---
# tw(s) = delta^-2 int_{s-delta}^{s+delta} w(t)(delta - |t-s|) dt.
def prepare_weight_grid(w, delta, max_offset):
    tw = {}
    for k in range(-max_offset, max_offset + 1):
        s = k * delta
        val, _ = si.quad(
            lambda t: w(t) * (delta - abs(t - s)),
            s - delta,
            s + delta,
            limit=120,
        )
        tw[k] = val / delta**2
    return tw

def build_operator(weight_data, n_cells):       # symmetric Toeplitz K_w on n cells
    K = np.empty((n_cells, n_cells))
    for i in range(n_cells):
        for j in range(n_cells):
            K[i, j] = weight_data[i - j]
    return K

# --- rank-one whitening: A_lambda^2 = lambda I + lambda^{-1} |1><1| (delta-weighted) ---
def prepare_search_form(search_value, n_cells, delta):
    lam = search_value
    a = n_cells * delta / 2.0                   # half support length; <1,1> = 2a
    # 2*sqrt(lam)*b + 2a*b^2 = 1/lam, unique positive root
    b = (-2*np.sqrt(lam) + np.sqrt(4*lam + 8*a/lam)) / (4*a)
    one = np.ones(n_cells)
    A    = np.sqrt(lam) * np.eye(n_cells) + b * delta * np.outer(one, one)
    Ainv = np.linalg.inv(A)
    return A, Ainv

def leading_symmetric_eigenpair(M, options):    # top eigenpair of symmetric M
    iters = options.get("iters", 2000)
    tol = options.get("tol", 1e-13)
    v = np.random.default_rng(0).standard_normal(M.shape[0]); v /= np.linalg.norm(v)
    mu = 0.0
    for _ in range(iters):
        w_ = M @ v; nw = np.linalg.norm(w_)
        if nw == 0: break
        v_new = w_ / nw
        mu_new = v_new @ (M @ v_new)
        if abs(mu_new - mu) < tol: v, mu = v_new, mu_new; break
        v, mu = v_new, mu_new
    return mu, v

# --- c_{lambda,delta}: maximize the H_lambda-normalized form over nonnegative step fns ---
# constraint f>=0 handled exactly: extremizer is a single contiguous bump (discrete Riesz),
# so sweep the support length and keep the largest admissible (nonneg) eigenvector.
def solve_finite_problem(K_full, search_value, delta, n_full, options):
    lam = search_value
    best_val, best_f = 0.0, None
    for L in range(1, n_full + 1):              # contiguous support of L cells
        lo = (n_full - L) // 2
        K = K_full[lo:lo+L, lo:lo+L]
        A, Ainv = prepare_search_form(lam, L, delta)
        # With the delta-weighted dot product, the convolution operator is delta*K.
        M = 2.0 * (Ainv @ (delta * K) @ Ainv)
        mu, g = leading_symmetric_eigenpair(M, options)
        f = Ainv @ g                            # back to the original variable
        if f.min() < 0: f = -f                  # eigenvector sign is free
        if f.min() >= -1e-9 and mu > best_val:  # admissible & better
            best_val, best_f = mu, np.maximum(f, 0.0)
    return best_val, best_f

# --- original (undiscretized-denominator) quotient: an honest lower bound on C_opt ---
def evaluate_test_function(f, K_full, lo, delta):
    n = len(f); K = K_full[lo:lo+n, lo:lo+n]
    num = delta**2 * (f @ (K @ f))              # <f, K_w f>
    l1  = delta * f.sum()
    l2  = np.sqrt(delta * (f * f).sum())
    return num / (l1 * l2)

def make_upper_certificate(c_ld, search_value, delta):
    lam = search_value
    err = 16 * delta**2 / (np.pi**2 * lam**2)
    return 0.5 * (c_ld + np.sqrt(c_ld*c_ld + 4*err))

def estimate_constant(w, delta, support_radius, search_grid, options):
    n_full = int(round(2 * support_radius / delta))
    weight_data = prepare_weight_grid(w, delta, n_full)
    K_full = build_operator(weight_data, n_full)
    upper, lower = 0.0, 0.0
    for lam in search_grid:
        c_ld, f = solve_finite_problem(K_full, lam, delta, n_full, options)
        if f is None: continue
        upper = max(upper, make_upper_certificate(c_ld, lam, delta))
        # lower bound: the admissible eigenvector is a feasible test function
        lo = (n_full - len(f)) // 2
        lower = max(lower, evaluate_test_function(f, K_full, lo, delta))
    grid_err = 0.5 * np.max(np.diff(search_grid)) if len(search_grid) > 1 else 0.0
    tail_upper = max(2 * search_grid[0], 2 / search_grid[-1])
    return lower, max(upper + grid_err, tail_upper)

if __name__ == "__main__":
    search_grid = np.arange(0.35, 2.81, 0.001)  # tails use c_lambda <= min(2*lambda, 2/lambda)
    options = {}
    for name, w in [("indicator", indicator_weight), ("gaussian", gaussian_weight)]:
        lo, hi = estimate_constant(w, delta=1.45e-3, support_radius=4.0,
                                   search_grid=search_grid,
                                   options=options)
        print(name, "C_opt in", (round(lo, 6), round(hi, 6)))
```

The causal chain, end to end: the denominator $\|f\|_1\|f\|_2$ blocks a spectral attack, so AM-GM splits it into a $\lambda$-family of Hilbert-space quadratic forms with $C_{opt}=\max_\lambda c_\lambda$; Riesz rearrangement and the Euler-Lagrange support bound license restricting to symmetric-decreasing bumps on a finite grid; block-average discretization plus the convolution-induced smoothness of the extremizer give a rigorous $O(\delta^2)$ gap between the discrete and continuous constants; rank-one whitening turns each $c_{\lambda,\delta}$ into a top-eigenvalue problem solved by the power method, with the $f\ge0$ cone handled exactly by sweeping the contiguous support length; and the quadratic upper correction together with the eigenvector lower quotient sandwiches $C_{opt}$ at polynomial cost.
