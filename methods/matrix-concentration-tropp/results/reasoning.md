Let me start from what actually keeps showing up. Over and over, in randomized low-rank factorization, in sparsification, in compressed sensing, in covariance estimation, the matrix I care about is a sum of independent random matrices, $\mtx Y=\sum_k \mtx X_k$, and the thing I need to control is how big its top eigenvalue can get: $\Prob{\lambda_{\max}(\sum_k \mtx X_k)\geq t}$. These matrices are finite-dimensional — I don't want an asymptotic $n\to\infty$ law, I want a number for a fixed $d\times d$ matrix. And I want explicit, reasonable constants, because the reason I'm doing this is to certify that some randomized algorithm works. The hypotheses I can actually check are local and simple: each summand is bounded, $\lambda_{\max}(\mtx X_k)\leq R$; or I know its mean and something about its variance. So the real question is: can I take that kind of cheap, per-summand information and turn it into a strong tail bound for the spectrum of the whole sum?

The scalar version of this question is completely solved, and the solution is so clean that I want to just imitate it. For a real random variable $Y$, Markov on the exponential does everything: for any $\theta>0$, $\Prob{Y\geq t}=\Prob{e^{\theta Y}\geq e^{\theta t}}\leq e^{-\theta t}\Expect e^{\theta Y}$, then optimize over $\theta$. And the reason this is *decisive* for a sum, rather than just a one-variable trick, is that the exponential turns sums into products: $\Expect e^{\theta\sum_k X_k}=\Expect\prod_k e^{\theta X_k}=\prod_k\Expect e^{\theta X_k}$ by independence. Take logs and the cumulant generating function is additive, $\log\Expect e^{\theta\sum_k X_k}=\sum_k\log\Expect e^{\theta X_k}$. That additivity is the whole game: I bound each summand's cgf using its structure (boundedness, variance) and just add them up. Chernoff, Bennett, Bernstein, Hoeffding — they're all this one move with a different per-summand cgf estimate plugged in.

So the dream is to run this verbatim with matrices. The first thing I need is a matrix version of "Markov on the exponential," and here I'm not starting from nothing — there's a way to set this up. Take $\mtx Y$ random self-adjoint. $\lambda_{\max}$ is homogeneous, so $\Prob{\lambda_{\max}(\mtx Y)\geq t}=\Prob{\lambda_{\max}(\theta\mtx Y)\geq \theta t}$ for $\theta>0$; the scalar exponential is monotone, so this is $\Prob{e^{\lambda_{\max}(\theta\mtx Y)}\geq e^{\theta t}}\leq e^{-\theta t}\Expect e^{\lambda_{\max}(\theta\mtx Y)}$ by Markov. Now I need to get a matrix back into the picture. The spectral mapping theorem says $e^{\lambda_{\max}(\theta\mtx Y)}=\lambda_{\max}(e^{\theta\mtx Y})$, because exponentiating a self-adjoint matrix just exponentiates its eigenvalues. And $e^{\theta\mtx Y}$ is positive definite, so its largest eigenvalue is dominated by its trace: $\lambda_{\max}(e^{\theta\mtx Y})\leq \trace e^{\theta\mtx Y}$. Putting it together,
$$
\Prob{\lambda_{\max}(\mtx Y)\geq t}\leq \inf_{\theta>0}\; e^{-\theta t}\,\Expect\trace e^{\theta\mtx Y}.
$$
Good — this is the exact analog of the scalar Laplace transform bound, with $\Expect\trace e^{\theta\mtx Y}$ playing the role of the mgf. I'll pause on that last step, though, because it looks crude: I threw away everything about $\lambda_{\max}$ and replaced it with the trace, a sum over *all* eigenvalues. That bound costs me something — it's where the dimension is going to sneak in. But I'll let it stand for now; it might be paying for itself later, and I don't yet see an alternative.

Now the hard part: I have to bound $\Expect\trace\exp(\sum_k\theta\mtx X_k)$ using only per-summand information. In the scalar world this was free — the exponential factorized. So let me just try to factorize. I want $e^{\sum_k\theta\mtx X_k}=\prod_k e^{\theta\mtx X_k}$. And it's false. The matrix exponential does *not* convert sums to products; $e^{\mtx A+\mtx H}=e^{\mtx A}e^{\mtx H}$ holds only when $\mtx A$ and $\mtx H$ commute, and my random summands certainly don't. Worse, even if I could split them, a product of self-adjoint matrices isn't self-adjoint, so I'd have lost the structure I'm relying on. The single fact that made the scalar method work just isn't available. That's the wall.

Is there *anything* of the multiplication rule that survives? The trace is more forgiving than the bare exponential. There's the Golden–Thompson inequality from quantum statistical mechanics: $\trace e^{\mtx A+\mtx H}\leq \trace(e^{\mtx A}e^{\mtx H})$. So for two independent summands I can write, using independence to pull the expectation through the product of the two now-separated factors,
$$
\trace\Expect e^{\theta\mtx X_1+\theta\mtx X_2}\leq \Expect\trace\big(e^{\theta\mtx X_1}e^{\theta\mtx X_2}\big)=\trace\big[(\Expect e^{\theta\mtx X_1})(\Expect e^{\theta\mtx X_2})\big].
$$
That's encouraging — a genuine two-matrix substitute for the product rule. So I try to iterate it to $n$ matrices. And it won't extend. The obvious three-matrix Golden–Thompson, $\trace e^{\mtx A+\mtx H+\mtx K}\leq \trace(e^{\mtx A}e^{\mtx H}e^{\mtx K})$, is simply false. So I can't keep peeling. What I *can* do — and this is what one does if one insists on Golden–Thompson — is peel off one factor at a time using a trace–norm bound: $\trace[\mtx M_1\mtx M_2]\leq \trace(\mtx M_1)\cdot\lambda_{\max}(\mtx M_2)$ when both are positive. So
$$
\trace\Expect e^{\theta\mtx Y}\leq \trace\big(\Expect e^{\sum_{k\leq n-1}\theta\mtx X_k}\big)\cdot\lambda_{\max}\big(\Expect e^{\theta\mtx X_n}\big),
$$
and iterating all the way down, since $\trace\Id=d$,
$$
\trace\Expect e^{\theta\mtx Y}\leq d\cdot\prod_k\lambda_{\max}\big(\Expect e^{\theta\mtx X_k}\big)=d\cdot\exp\Big(\sum_k\lambda_{\max}\big(\log\Expect e^{\theta\mtx X_k}\big)\Big).
$$
Let me look hard at what this gives, because it *is* a usable bound — it produces matrix Chernoff inequalities and so on — but something is off about the scale. Each summand enters through $\lambda_{\max}(\log\Expect e^{\theta\mtx X_k})$, a *maximum eigenvalue of one summand at a time*, and these get summed. So my scale parameter is a "sum of eigenvalues." Compare what I'd want. Take a concrete case to see the damage: a Gaussian series $\sum_k\gamma_k\mtx A_k$. I'll work out in a moment that $\log\Expect e^{\theta\gamma_k\mtx A_k}=\tfrac{\theta^2}{2}\mtx A_k^2$, so the peeling bound's scale is $\sum_k\lambda_{\max}(\mtx A_k^2)=\sum_k\|\mtx A_k\|^2$. But the honest scale of the fluctuation ought to be $\lambda_{\max}(\sum_k\mtx A_k^2)=\|\sum_k\mtx A_k^2\|$. The difference between "sum of the eigenvalues" and "the eigenvalue of the sum" can be a factor of $d$, the ambient dimension. And it's not sitting out front as a harmless prefactor — it's *inside the exponent*. A factor of $d$ in the exponent of a large-deviation bound is catastrophic; it moves the whole transition. The peeling is bleeding the matrices apart into separate exponentials where they can never recombine, and that separation is exactly what loses the cross-information that would let "sum of eigenvalues" tighten to "eigenvalue of a sum."

So Golden–Thompson is the wrong hammer. It's a two-matrix inequality and the problem has $n$ matrices that genuinely interact; forcing pairwise separations throws the interaction away. I need to stop trying to imitate the *multiplication* rule. Let me go back to the scalar method and ask what I'm really imitating. The mgf multiplied; but the thing I actually used was that the *cgf added*: $\log\Expect e^{\theta\sum_k X_k}=\sum_k\log\Expect e^{\theta X_k}$. Maybe additivity, not multiplicativity, is the property with a matrix life. I don't want $e^{\sum}=\prod e$; I want something like $\Expect\trace\exp(\sum_k\theta\mtx X_k)\leq \trace\exp(\sum_k\log\Expect e^{\theta\mtx X_k})$ — the summands' cumulant generating functions, $\log\Expect e^{\theta\mtx X_k}$, all *added up under a single exponential*. If I could get that, the matrices would stay together inside one $\exp$, the scale would be an eigenvalue-of-a-sum, and the factor $d$ would never appear in the exponent. That's the target. The question is whether such an inequality is even true, and if so, why.

Stare at the shape of what I want. I want to take a random matrix sitting inside a trace-exponential and pull the expectation *inside a logarithm*: replace $\Expect\trace\exp(\mtx H+\mtx X)$ by something with $\log\Expect e^{\mtx X}$ in it. Pulling an expectation inside a concave function and getting an inequality is exactly Jensen. So really I'm asking: is some trace-exponential-of-a-log a *concave* function of its matrix argument? Phrase it cleanly. Let $\mtx H$ be fixed self-adjoint and consider the map
$$
\mtx A\;\longmapsto\;\trace\exp(\mtx H+\log\mtx A)
$$
on positive-definite $\mtx A$. If *this* is concave, Jensen will do everything I want. In the scalar case it's not even interesting: $\exp(h+\log a)=e^h a$ is linear in $a$, so "concave" is vacuously true and Jensen is an equality. So whatever I'm asking for is a genuinely matrix statement, an extra rigidity that only noncommuting matrices can violate or respect — there's no scalar shadow to guess from.

A theorem from matrix analysis has exactly this shape: for fixed self-adjoint $\mtx H$, the map $\mtx A\mapsto \trace\exp(\mtx H+\log\mtx A)$ is concave on the positive-definite cone. So the rigidity holds. Read as probability rather than as trace-function geometry, it says a random positive-definite matrix can be averaged before the logarithm, provided the whole expression remains inside a trace exponential. Once I have that, the rest is Jensen.

Let me make the probabilistic corollary explicit, because this is the hinge. Let $\mtx H$ be fixed self-adjoint and $\mtx X$ random self-adjoint. Set the random *positive-definite* matrix $\mtx Y:=e^{\mtx X}$. Since $\log$ is the functional inverse of $\exp$ on the pd cone, $\mtx X=\log\mtx Y$, so
$$
\Expect\trace\exp(\mtx H+\mtx X)=\Expect\trace\exp(\mtx H+\log\mtx Y).
$$
By Lieb's theorem the function $\mtx Y\mapsto\trace\exp(\mtx H+\log\mtx Y)$ is concave, so Jensen lets me move the expectation inside:
$$
\Expect\trace\exp(\mtx H+\log\mtx Y)\leq \trace\exp(\mtx H+\log\Expect\mtx Y)=\trace\exp\big(\mtx H+\log\Expect e^{\mtx X}\big).
$$
There it is:
$$
\Expect\trace\exp(\mtx H+\mtx X)\leq \trace\exp\big(\mtx H+\log\Expect e^{\mtx X}\big).
$$
A single random matrix $\mtx X$, sitting next to anything fixed, can be replaced by its cgf $\log\Expect e^{\mtx X}$ at the cost of an inequality — and crucially it stays *inside the same exponential* as $\mtx H$. Nothing got peeled off. This is the analytic substitute for the algebraic factorization I couldn't have.

Now I get the full subadditivity by feeding this in one summand at a time, with the conditioning arranged so that everything except the summand I'm currently absorbing is "fixed." Take $\theta=1$ without loss of generality (absorb it into the $\mtx X_k$). Let $\Expect_k$ be expectation conditioned on $\mtx X_1,\dots,\mtx X_k$, and abbreviate $\mtx\Xi_k:=\log(\Expect e^{\mtx X_k})$ — by independence the conditional and unconditional cgfs coincide. Peel from the top using the tower property:
$$
\Expect\trace\exp\Big(\sum_{k=1}^n\mtx X_k\Big)=\Expect_0\cdots\Expect_{n-1}\trace\exp\Big(\sum_{k=1}^{n-1}\mtx X_k+\mtx X_n\Big).
$$
Apply the corollary to the innermost expectation $\Expect_{n-1}$ with the fixed matrix $\mtx H=\sum_{k\leq n-1}\mtx X_k$ (which doesn't depend on $\mtx X_n$): the term $\mtx X_n$ becomes $\mtx\Xi_n=\log\Expect e^{\mtx X_n}$,
$$
\leq \Expect_0\cdots\Expect_{n-2}\trace\exp\Big(\sum_{k=1}^{n-2}\mtx X_k+\mtx X_{n-1}+\mtx\Xi_n\Big).
$$
Now $\mtx X_{n-1}$ is the live random matrix and everything else — $\sum_{k\leq n-2}\mtx X_k$ and $\mtx\Xi_n$ — is fixed relative to it, so I apply the corollary again with $\mtx H=\sum_{k\leq n-2}\mtx X_k+\mtx\Xi_n$, turning $\mtx X_{n-1}$ into $\mtx\Xi_{n-1}$. At the general step $m$ I use $\mtx H_m=\sum_{k<m}\mtx X_k+\sum_{k>m}\mtx\Xi_k$, which is legal precisely because $\mtx H_m$ doesn't involve $\mtx X_m$. Repeating down to the bottom,
$$
\Expect\trace\exp\Big(\sum_{k=1}^n\theta\mtx X_k\Big)\leq \trace\exp\Big(\sum_{k=1}^n\log\Expect e^{\theta\mtx X_k}\Big).
$$
This is the matrix subadditivity of cumulants — the exact statement I wanted, and it is the honest replacement for the additivity rule $\sum_k\Xi_{X_k}$ from the scalar world. The many-matrix Golden–Thompson path that failed under peeling is now irrelevant; the trace concavity did the work that pairwise product inequalities could not do.

Combine it with the Laplace transform bound and I have a master inequality. Substituting the subadditivity into $\Prob{\lambda_{\max}(\mtx Y)\geq t}\leq \inf_{\theta>0}e^{-\theta t}\Expect\trace e^{\theta\mtx Y}$:
$$
\Prob{\lambda_{\max}\Big(\sum_k\mtx X_k\Big)\geq t}\leq \inf_{\theta>0}\; e^{-\theta t}\,\trace\exp\Big(\sum_k\log\Expect e^{\theta\mtx X_k}\Big).
$$
Now I want a deployable form, because in practice I won't have the cgf exactly — I'll have a *semidefinite upper bound* on each summand's mgf coming from its structure. Suppose I can produce a scalar function $g$ and fixed matrices $\mtx A_k$ with
$$
\Expect e^{\theta\mtx X_k}\preceq e^{g(\theta)\mtx A_k}\qquad(\theta>0).
$$
The matrix logarithm is operator monotone, so this implies $\log\Expect e^{\theta\mtx X_k}\preceq g(\theta)\mtx A_k$ — and here is where it matters that the comparison is passing through $\log$ on the positive-definite cone, not through a false matrix-exponential monotonicity rule. The trace exponential is monotone with respect to the semidefinite order, so I can slot all these bounds into the master inequality at once:
$$
\Prob{\lambda_{\max}\Big(\sum_k\mtx X_k\Big)\geq t}\leq e^{-\theta t}\trace\exp\Big(g(\theta)\sum_k\mtx A_k\Big).
$$
Finally I cash out the trace. The exponential of a self-adjoint matrix is positive definite, and the trace of a positive-definite matrix is at most $d$ times its largest eigenvalue: $\trace\exp(g(\theta)\sum_k\mtx A_k)\leq d\cdot\lambda_{\max}(\exp(g(\theta)\sum_k\mtx A_k))=d\cdot\exp(g(\theta)\lambda_{\max}(\sum_k\mtx A_k))$, the last step by spectral mapping and $g\geq 0$. Writing $\rho:=\lambda_{\max}(\sum_k\mtx A_k)$,
$$
\Prob{\lambda_{\max}\Big(\sum_k\mtx X_k\Big)\geq t}\leq d\cdot\inf_{\theta>0}e^{-\theta t+g(\theta)\rho}.
$$
And *now* the dimension shows up — exactly once, as a multiplicative prefactor $d$ out front, born from that crude "$\lambda_{\max}\leq\trace$" step at the very beginning. It is *not* in the exponent. The whole payoff of going through Lieb instead of Golden–Thompson is the difference between this $d\cdot e^{(\dots)}$ and the peeling bound's $e^{d\cdot(\dots)}$. The exponent here optimizes to the perspective of a Fenchel conjugate, $d\cdot\exp(-\rho\, g^*(t/\rho))$, which is the matrix mirror of Cramér's classical large-deviation rate. The scale $\rho$ is an eigenvalue-of-a-sum. That's the structurally correct object.

I should also record a second deployment form, because for some structures I'd rather combine the mgfs under one logarithm than bound each cgf separately. The matrix logarithm is operator *concave*, so $\sum_k\log\Expect e^{\theta\mtx X_k}=n\cdot\tfrac1n\sum_k\log\Expect e^{\theta\mtx X_k}\preceq n\log(\tfrac1n\sum_k\Expect e^{\theta\mtx X_k})$, and feeding this into the master inequality and cashing out the trace the same way gives
$$
\Prob{\lambda_{\max}\Big(\sum_{k=1}^n\mtx X_k\Big)\geq t}\leq d\cdot\inf_{\theta>0}\exp\Big(-\theta t+n\log\lambda_{\max}\big(\tfrac1n\textstyle\sum_k\Expect e^{\theta\mtx X_k}\big)\Big).
$$
I'll want this one for Chernoff.

So the engine is built. Everything left is supplying the per-summand mgf bound $\Expect e^{\theta\mtx X_k}\preceq e^{g(\theta)\mtx A_k}$ that encodes the structure I have, and reading off $g$ and $\rho$. And there's a discipline here that I keep relearning: scalar inequalities do *not* generally transfer to the semidefinite order. The one bridge that always holds is the transfer rule — if $f(a)\leq h(a)$ for every $a$ in an interval containing the eigenvalues of $\mtx X$, then $f(\mtx X)\preceq h(\mtx X)$. So the way to get a matrix mgf bound is to find a *scalar* function inequality on the right interval and lift it, not to manipulate matrices directly.

Start with the cleanest case, a Gaussian or Rademacher series $\sum_k\xi_k\mtx A_k$, fixed self-adjoint $\mtx A_k$. Absorb $\theta$ into $\mtx A$ and take $\theta=1$. For a Rademacher sign $\eps$, $\Expect e^{\eps\mtx A}=\tfrac12(e^{\mtx A}+e^{-\mtx A})=\cosh(\mtx A)$. I need to dominate $\cosh$ by a Gaussian-looking exponential. The scalar inequality is clean: $q(a)=a^2/2-\log\cosh a$ is even, $q(0)=0$, and for $a\geq0$ its derivative is $q'(a)=a-\tanh a\geq0$, so $\cosh(a)\leq e^{a^2/2}$. The transfer rule lifts this to $\cosh(\mtx A)\preceq e^{\mtx A^2/2}$. So $\Expect e^{\eps\mtx A}\preceq e^{\mtx A^2/2}$. For a standard Gaussian $\gamma$ it's exact: the odd moments vanish and $\Expect\gamma^{2p}=\tfrac{(2p)!}{p!\,2^p}$, so $\Expect e^{\gamma\mtx A}=\Id+\sum_{p\geq1}\tfrac{\Expect\gamma^{2p}}{(2p)!}\mtx A^{2p}=\Id+\sum_{p\geq1}\tfrac{(\mtx A^2/2)^p}{p!}=e^{\mtx A^2/2}$. Either way, restoring $\theta$,
$$
\Expect e^{\xi_k\theta\mtx A_k}\preceq e^{(\theta^2/2)\mtx A_k^2}.
$$
So $g(\theta)=\theta^2/2$ and the role of "$\mtx A_k$" in the deployment corollary is played by $\mtx A_k^2$, giving $\rho=\lambda_{\max}(\sum_k\mtx A_k^2)=\|\sum_k\mtx A_k^2\|=:\sigma^2$. Plug in:
$$
\Prob{\lambda_{\max}\Big(\sum_k\xi_k\mtx A_k\Big)\geq t}\leq d\cdot\inf_{\theta>0}e^{-\theta t+\theta^2\sigma^2/2}=d\cdot e^{-t^2/(2\sigma^2)},
$$
the infimum at $\theta=t/\sigma^2$. Subgaussian, with the *correct* variance $\sigma^2=\|\sum_k\mtx A_k^2\|$ — the eigenvalue of a sum, just as I wanted, not the Golden–Thompson $\sum_k\|\mtx A_k\|^2$. For the spectral norm rather than the top eigenvalue, $\|\mtx Y\|=\max\{\lambda_{\max}(\mtx Y),-\lambda_{\min}(\mtx Y)\}$; since $\xi_k$ is symmetric, $-\lambda_{\min}(\sum_k\xi_k\mtx A_k)$ obeys the same bound (replace $\xi_k$ by $-\xi_k$), and a union bound gives $\Prob{\|\sum_k\xi_k\mtx A_k\|\geq t}\leq 2d\cdot e^{-t^2/(2\sigma^2)}$.

That $\sigma^2$ deserves a name and an interpretation, because it's the quantity that governs everything. For a centered sum $\mtx Y=\sum_k\mtx X_k$ it is $\|\sum_k\Expect\mtx X_k^2\|=\|\Expect\mtx Y^2\|$ — the spectral norm of the expected *square*, i.e. the magnitude of the expected squared deviation of $\mtx Y$ from its mean. It is one matrix that simultaneously records the variance in every direction at once, and its operator norm is the scale of the normal concentration. The rectangular case makes the structure vivid. A rectangular $\mtx B$ has no square, but its self-adjoint dilation $\coll S(\mtx B)=\left[\begin{smallmatrix}\mtx 0&\mtx B\\\mtx B^*&\mtx 0\end{smallmatrix}\right]$ does, and $\coll S(\mtx B)^2=\left[\begin{smallmatrix}\mtx B\mtx B^*&\mtx 0\\\mtx 0&\mtx B^*\mtx B\end{smallmatrix}\right]$, while $\lambda_{\max}(\coll S(\mtx B))=\|\mtx B\|$. So running the self-adjoint result on $\sum_k\xi_k\coll S(\mtx B_k)$ in dimension $d_1+d_2$ converts the norm of the rectangular series into a top eigenvalue, and the variance becomes $\sigma^2=\max\{\|\sum_k\mtx B_k\mtx B_k^*\|,\|\sum_k\mtx B_k^*\mtx B_k\|\}$ — the row space and the column space each get their own "sum of squares," and these are independent of each other; the matrix variance is genuinely a noncommutative two-sided sum of squares.

I want to understand that prefactor $d$ honestly, not just tolerate it — is it an artifact of my crude trace step, or is it real? Two thought experiments settle it. Take a diagonal Gaussian $\sum_{k=1}^d\gamma_k\mtx E_{kk}$. Its norm is $\max_k|\gamma_k|$, which for $d$ independent standard Gaussians exceeds $\sqrt{2\log d}$ with high probability. Here $\sigma^2=1$, so my bound is $2d\cdot e^{-t^2/2}$, which only becomes informative once $t\gtrsim\sqrt{2\log(2d)}$ — and that's exactly the right threshold, because the maximum of $d$ Gaussians really does sit near $\sqrt{2\log d}$. So the prefactor is *necessary*; the dimension genuinely costs a $\sqrt{\log d}$ in the typical norm, and the $d$ out front is encoding it. But it can also be loose: the unnormalized GOE $\mtx W=\sum_{j\leq k}\gamma_{jk}(\mtx E_{jk}+\mtx E_{kj})$ has the sharp $\Expect\|\mtx W\|\leq 2\sqrt d$, whereas integrating my tail gives $\Expect\|\mtx W\|\leq\sqrt{(d+3)\log(2ed)}$ — too large by about $\sqrt{\log d}$. So the picture is: my bound controls deviations of $\mtx Y$ *as a random matrix* in operator norm, $\Prob{\|\mtx Y-\Expect\mtx Y\|\geq t}\leq 2d\,e^{-t^2/(2\sigma^2)}$, and pays a dimensional toll for that completeness; the classical Gaussian concentration inequality instead controls the fluctuation of the *scalar* $\|\mtx Y\|$ about its own mean, with no $d$ but on the scale of the weak variance $\sigma_*^2=\sup_{\|u\|=\|v\|=1}\sum_k|u^*\mtx A_k v|^2$. These measure different things; $\sigma_*^2\leq\sigma^2\leq d\,\sigma_*^2$, and either extreme is attained. And the dimension that matters is *effective*, not nominal: if the ranges of the $\mtx A_k$ lie in an $r$-dimensional subspace I can replace $d$ by $r$ throughout, since the trace-vs-$\lambda_{\max}$ step only ever sees that subspace.

Now the bounded case — sums of positive matrices, the Chernoff regime. Suppose each $\mtx X_k$ is psd with $\lambda_{\max}(\mtx X_k)\leq 1$. On $[0,1]$ the convex function $e^{\theta x}$ lies below its chord: $e^{\theta x}\leq 1+(e^\theta-1)x$. The eigenvalues of $\mtx X$ live in $[0,1]$, so the transfer rule gives $e^{\theta\mtx X}\preceq\Id+(e^\theta-1)\mtx X$, and taking expectations (which respects $\preceq$) yields the mgf bound $\Expect e^{\theta\mtx X}\preceq\Id+(e^\theta-1)\Expect\mtx X$. This is *affine* in $\Expect\mtx X$, so I want the second deployment form, the one that combines mgfs under a single logarithm. With $g(\theta)=e^\theta-1$,
$$
\Prob{\lambda_{\max}\Big(\sum_k\mtx X_k\Big)\geq t}\leq d\cdot\exp\Big(-\theta t+n\log\lambda_{\max}\big(\Id+g(\theta)\tfrac1n\textstyle\sum_k\Expect\mtx X_k\big)\Big)=d\cdot\exp\big(-\theta t+n\log(1+g(\theta)\bar\mu_{\max})\big),
$$
where $\bar\mu_{\max}=\tfrac1n\lambda_{\max}(\sum_k\Expect\mtx X_k)$. Writing $t=n\alpha$ and minimizing over $\theta$ — for $\bar\mu_{\max}<\alpha<1$ the optimum is $\theta=\log\tfrac{\alpha}{1-\alpha}-\log\tfrac{\bar\mu_{\max}}{1-\bar\mu_{\max}}$, while $\alpha\geq1$ is already outside the possible range — produces the information-divergence Chernoff bound; and using the cruder $\log(1+x)\leq x$ collapses it to the familiar binomial form. Setting $t=(1+\delta)\mu_{\max}$ with $\mu_{\max}=\lambda_{\max}(\sum_k\Expect\mtx X_k)$ and, in the normalized $R=1$ case, $\theta=\log(1+\delta)$,
$$
\Prob{\lambda_{\max}\Big(\sum_k\mtx X_k\Big)\geq(1+\delta)\mu_{\max}}\leq d\cdot\Big[\frac{e^{\delta}}{(1+\delta)^{1+\delta}}\Big]^{\mu_{\max}/R}
$$
after restoring the scale $R$ by homogeneity, equivalently using $\theta=R^{-1}\log(1+\delta)$ before rescaling. The lower tail comes from running the same argument on $\{-\mtx X_k\}$ — $\lambda_{\min}(\sum_k\mtx X_k)=-\lambda_{\max}(\sum_k(-\mtx X_k))$ — with $g(\theta)=1-e^{-\theta}$ in the normalized case and optimizer $\theta=-\log(1-\delta)$, giving $\Prob{\lambda_{\min}(\sum_k\mtx X_k)\leq(1-\delta)\mu_{\min}}\leq d\,[\,e^{-\delta}/(1-\delta)^{1-\delta}\,]^{\mu_{\min}/R}$, or $\theta=-R^{-1}\log(1-\delta)$ on the original scale. So the extreme eigenvalues of a sum of bounded positive matrices have exactly the binomial-tail behavior of a scalar Chernoff sum, with $\mu_{\max},\mu_{\min}$ the extreme eigenvalues of the mean.

Last, the inequality I reach for most, Bernstein: centered summands, a uniform bound, and the variance as the scale. Suppose $\Expect\mtx X=\mtx 0$ and $\lambda_{\max}(\mtx X)\leq 1$. I need an mgf bound that surfaces $\Expect\mtx X^2$ rather than $\Expect\mtx X$. Define the scalar $f(x)=(e^{\theta x}-\theta x-1)/x^2$ for $x\neq0$ and $f(0)=\theta^2/2$ — this is exactly the second-order remainder of the exponential divided by $x^2$. The integral identity $f(x)=\int_0^\theta(\theta-s)e^{sx}\,ds$ makes monotonicity transparent, since $f'(x)=\int_0^\theta s(\theta-s)e^{sx}\,ds\geq0$. Thus $f(x)\leq f(1)$ for $x\leq1$; by the transfer rule $f(\mtx X)\preceq f(1)\Id$. Now expand the exponential keeping the first two terms exactly: $e^{\theta\mtx X}=\Id+\theta\mtx X+\mtx X f(\mtx X)\mtx X$ (the third term is the remainder, and $f(\mtx X)$ commutes with $\mtx X$, so sandwiching is legitimate), hence
$$
e^{\theta\mtx X}\preceq\Id+\theta\mtx X+f(1)\mtx X^2.
$$
Take the expectation; the linear term *dies* because $\Expect\mtx X=\mtx 0$ — this is why centering is the hypothesis that buys me the variance — leaving $\Expect e^{\theta\mtx X}\preceq\Id+f(1)\Expect\mtx X^2$. Since $\Id+\mtx M\preceq e^{\mtx M}$ for any self-adjoint $\mtx M$, and $f(1)=e^\theta-\theta-1$,
$$
\Expect e^{\theta\mtx X}\preceq\exp\big((e^\theta-\theta-1)\Expect\mtx X^2\big).
$$
So $g(\theta)=e^\theta-\theta-1$, the structural matrix is $\Expect\mtx X_k^2$, and $\rho=\lambda_{\max}(\sum_k\Expect\mtx X_k^2)=\|\sum_k\Expect\mtx X_k^2\|=\sigma^2$ — the matrix variance again. The first deployment corollary gives
$$
\Prob{\lambda_{\max}\Big(\sum_k\mtx X_k\Big)\geq t}\leq d\cdot\inf_{\theta>0}e^{-\theta t+(e^\theta-\theta-1)\sigma^2},
$$
and the infimum is at $\theta=\log(1+t/\sigma^2)$ in the normalized $R=1$ case. Restoring $R$ gives the original-scale optimizer $\theta=R^{-1}\log(1+Rt/\sigma^2)$ and the Bennett form $d\cdot\exp(-\tfrac{\sigma^2}{R^2}h(\tfrac{Rt}{\sigma^2}))$ with $h(u)=(1+u)\log(1+u)-u$. The scalar bound $h(u)\geq\tfrac{u^2/2}{1+u/3}$ smooths this into the Bernstein inequality I'll actually quote; if $q(u)=h(u)-\tfrac{u^2}{2(1+u/3)}$, then $q(0)=q'(0)=0$ and $q''(u)=u^2(u+9)/[(u+1)(u+3)^3]\geq0$, so no sign is hiding in the smoothing:
$$
\Prob{\lambda_{\max}\Big(\sum_k\mtx X_k\Big)\geq t}\leq d\cdot\exp\Big(\frac{-t^2/2}{\sigma^2+Rt/3}\Big),\qquad \sigma^2=\Big\|\sum_k\Expect\mtx X_k^2\Big\|.
$$
And I can read the two regimes straight off the denominator: when $t\leq\sigma^2/R$ the variance term dominates and the bound is subgaussian, $\sim e^{-3t^2/(8\sigma^2)}$; when $t\geq\sigma^2/R$ the $Rt/3$ term dominates and it's subexponential, $\sim e^{-3t/(8R)}$ — normal concentration near the mean on the variance scale, crossing to slower exponential decay in the tail on the uniform-bound scale, exactly the shape of scalar Bernstein, now with the matrix variance $\|\sum_k\Expect\mtx X_k^2\|$ in the role of the variance and $R$ in the role of the bound. If instead of a hard bound I only control moment growth, $\Expect\mtx X^p\preceq\tfrac{p!}{2}R^{p-2}\mtx A^2$ (the subexponential profile), I bound the mgf term-by-term in its Taylor series: $\Expect e^{\theta\mtx X}\preceq\Id+\sum_{p\geq2}\tfrac{\theta^p R^{p-2}}{2}\mtx A^2=\Id+\tfrac{\theta^2}{2(1-R\theta)}\mtx A^2\preceq\exp(\tfrac{\theta^2}{2(1-R\theta)}\mtx A^2)$ for $0<\theta<1/R$, so $g(\theta)=\tfrac{\theta^2}{2(1-R\theta)}$ and the choice $\theta=t/(\sigma^2+Rt)$ gives the subexponential Bernstein form $d\cdot\exp(-(t^2/2)/(\sigma^2+Rt))$. The rectangular Bernstein follows, as always, by running the self-adjoint result on the dilation, with $\sigma^2=\max\{\|\sum_k\Expect\mtx Z_k\mtx Z_k^*\|,\|\sum_k\Expect\mtx Z_k^*\mtx Z_k\|\}$ and dimension $d_1+d_2$.

Let me say back the causal chain, because the whole thing rests on one substitution. I want a tail bound on $\lambda_{\max}$ of an independent matrix sum from cheap per-summand data; the scalar Laplace-transform method does this by factorizing the mgf, but the matrix exponential refuses to factorize and Golden–Thompson only patches two matrices at a time, so peeling $n$ of them apart bleeds the scale parameter from "the eigenvalue of a sum" up to "the sum of eigenvalues," a factor of $d$ in the exponent. The escape is to stop imitating mgf *multiplicativity* and instead generalize cgf *additivity*: I bound $\lambda_{\max}$ by the trace, which costs a prefactor $d$ but unlocks the concavity of the trace exponential; the trace-concavity theorem, read as a fact about probability, lets Jensen pull an expectation inside the cgf without separating the summands; iterating that keeps all the cgfs under one exponential, so the scale stays an eigenvalue-of-a-sum and the dimension survives only as a benign multiplicative prefactor. With the master bound $\Prob{\lambda_{\max}(\sum_k\mtx X_k)\geq t}\leq d\inf_\theta e^{-\theta t+g(\theta)\rho}$ in hand, every classical inequality drops out by supplying one scalar function inequality and lifting it with the transfer rule: $\cosh(a)\leq e^{a^2/2}$ gives Gaussian/Rademacher with $\sigma^2=\|\sum\mtx A_k^2\|$, the chord of $e^{\theta x}$ on $[0,1]$ gives Chernoff, and the second-order remainder bound gives Bernstein with the matrix variance $\sigma^2=\|\sum_k\Expect\mtx X_k^2\|$. The matrix variance — the norm of the expected square — measures the spread of the centered sum on the normal-concentration scale.

```python
import numpy as np

# Matrix concentration via the matrix Laplace-transform method.
# Each bound is: supply a scalar function inequality, lift it by the transfer rule
# to a semidefinite mgf bound  E e^{theta X_k} <= e^{g(theta) A_k},  then optimize.

def lambda_max(M):
    return np.linalg.eigvalsh((M + M.conj().T) / 2).max()

# --- Master deployment bound:  P{ lambda_max(sum X_k) >= t } <= d * inf_theta e^{-theta t + g(theta) rho}
#     g, rho come from the per-summand semidefinite mgf bound; d is the prefactor
#     from bounding lambda_max by the trace.  (eigenvalue-of-a-sum scale, dimension only out front)
def master_bound(d, g, rho, t, thetas):
    return d * min(np.exp(-th * t + g(th) * rho) for th in thetas)

# --- Matrix Gaussian / Rademacher series  sum_k xi_k A_k :
#     cosh(A) <= e^{A^2/2}  =>  g(theta)=theta^2/2,  sigma^2 = || sum_k A_k^2 ||,  opt theta = t/sigma^2
def gaussian_series_bound(A_list, t, two_sided=False):
    d = A_list[0].shape[0]
    sigma2 = lambda_max(sum(A @ A for A in A_list))          # eigenvalue OF A SUM, not sum of eigenvalues
    pref = 2 * d if two_sided else d
    return pref * np.exp(-t**2 / (2 * sigma2))

# --- Matrix Bernstein (bounded case):  E X_k = 0,  lambda_max(X_k) <= R
#     remainder bound  =>  g(theta)=e^theta-theta-1,  sigma^2 = || sum_k E X_k^2 ||  (the matrix variance)
def bernstein_bound(EX2_list, R, t):
    d = EX2_list[0].shape[0]
    sigma2 = lambda_max(sum(EX2_list))                       # matrix variance = norm of the expected square
    return d * np.exp(-(t**2 / 2) / (sigma2 + R * t / 3))    # subgaussian then subexponential

# --- Matrix Chernoff (psd summands, lambda_max(X_k) <= R):
#     chord of e^{theta x} on [0,1]  =>  binomial-type bound on the extreme eigenvalues of the sum
def chernoff_upper(mu_max, R, delta, d):
    return d * (np.exp(delta) / (1 + delta)**(1 + delta))**(mu_max / R)

def chernoff_lower(mu_min, R, delta, d):
    return d * (np.exp(-delta) / (1 - delta)**(1 - delta))**(mu_min / R)
```
