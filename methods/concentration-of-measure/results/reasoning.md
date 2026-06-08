I keep coming back to the same gap between what I can say and what I want to say. I have $n$ independent random variables $X_1,\dots,X_n$, I form the average $\bar X = (X_1+\dots+X_n)/n$, and I know it concentrates around $\mu = E\bar X$. The law of large numbers tells me $\bar X\to\mu$, the central limit theorem tells me the fluctuation is of size $1/\sqrt n$ and shaped like a Gaussian. But both of those are statements about $n\to\infty$. If someone hands me a concrete $n$ — a sample size — and a concrete tolerance $t$, and asks "what is the chance the average is off by more than $t$?", neither theorem gives me a number I can stand behind. I want a finite-sample bound on

$$\Pr\{\bar X - \mu \ge t\} = \Pr\{S - ES \ge nt\},\qquad S = X_1+\dots+X_n,$$

and I want it to be small — exponentially small in $n$ if I can manage it, because that is what the Gaussian limit hints is the truth.

The only honest tool I have for bounding a probability by an expectation is Markov: for a nonnegative $Y$ and $t>0$, since $Y\ge t$ exactly where the indicator $\mathbf 1\{Y\ge t\}$ fires, I have $Y\ge t\,\mathbf 1\{Y\ge t\}$ pointwise, and taking expectations,

$$\Pr\{Y\ge t\}\le \frac{EY}{t}.$$

That is the whole content. Everything I do will be a clever choice of what to feed into Markov. The obvious thing is to feed it the second moment of the deviation: take $Y=(S-ES)^2$, or work with $|\bar X-\mu|$ and square. That gives Chebyshev,

$$\Pr\{|\bar X-\mu|\ge t\}\le \frac{\operatorname{Var}(\bar X)}{t^2}=\frac{\sigma^2}{nt^2},$$

using independence so the variance of the sum is the sum of the variances. And this is genuinely something — it decays in $n$, it needs no boundedness, just a finite variance. But look at the decay in $t$: it is $1/t^2$. Polynomial. The Gaussian tail decays like $e^{-nt^2/(2\sigma^2)}$, which for any fixed $t$ crushes $1/(nt^2)$ as $n$ grows. So Chebyshev is leaving almost all of the concentration on the table. I am bounding an exponentially small probability by a polynomially small one.

Why is it so weak? Because the second moment is a blunt instrument — it only knows the *spread* of $S$, not the fact that $S$ has light tails. Higher moments would see more: $\Pr\{|S-ES|\ge nt\}\le E|S-ES|^q/(nt)^q$, and I could optimize over $q$. That is sharper, but the moments $E|S-ES|^q$ of a sum are a combinatorial mess to control. I want something that packages *all* the moments at once and still factorizes over the independent summands.

The exponential does exactly that. The indicator $\mathbf 1\{S-ES\ge nt\}$ is bounded above by $\exp\{h(S-ES-nt)\}$ for any constant $h>0$ — because where the indicator is $1$ the exponent is nonnegative so the exponential is $\ge 1$, and where the indicator is $0$ the exponential is still positive. So

$$\Pr\{S-ES\ge nt\}=E\,\mathbf 1\{S-ES\ge nt\}\le E\,e^{h(S-ES-nt)}=e^{-hnt}\,E\,e^{h(S-ES)}.$$

Now independence pays off in a way it never could for raw moments: the expectation of a product of independent things is the product of the expectations, so

$$E\,e^{h(S-ES)}=\prod_{i=1}^n E\,e^{h(X_i-EX_i)}.$$

The sum inside the exponent became a product outside. I have converted the joint problem into $n$ one-dimensional problems, one moment generating function per summand, glued by a single free parameter $h$ that I get to tune. So the plan is: bound each MGF $E\,e^{h(X_i-EX_i)}$, multiply, and then minimize $e^{-hnt}\prod_i(\dots)$ over $h>0$. Let me hold onto the structure of that minimization, because it is going to recur. If I write $\psi(h)=\log E e^{hZ}$ for the log-MGF of a centered $Z$, the bound is $\exp(-ht + \psi(h))$ and I want $\inf_{h>0}(-ht+\psi(h))$, i.e. I want $\sup_{h>0}(ht-\psi(h))$ in the exponent with a minus sign. That supremum is the Legendre transform of $\psi$. Just to feel the target: if $Z$ were genuinely Gaussian with variance $v$, then $\psi(h)=h^2v/2$ exactly, and $\sup_h(ht-h^2v/2)$ is attained at $h=t/v$ with value $t^2/(2v)$, so I'd get $\Pr\{Z\ge t\}\le e^{-t^2/(2v)}$. That is the shape I am chasing. If I can show each summand's log-MGF is bounded by a quadratic $h^2 v_i/2$, the whole thing will be sub-Gaussian and I'll get a clean $e^{-t^2/(2\sum_i v_i)}$.

So the entire game reduces to: for a centered random variable $X$ confined to a bounded interval, how big can $E\,e^{hX}$ be? I need an upper bound on the MGF that is no worse than the Gaussian one $e^{h^2(\text{something})/2}$.

Let me just stare at a bounded variable. Say $a\le X\le b$. What do I actually know? I know $X$ lives in $[a,b]$ and (let me center it) $EX=0$. I want $E e^{hX}$. The function $x\mapsto e^{hx}$ is convex. And a convex function on $[a,b]$ lies below the straight line — the chord — connecting its two endpoints. Concretely, for $a\le x\le b$, write $x$ as the convex combination $x=\frac{b-x}{b-a}\,a+\frac{x-a}{b-a}\,b$; then by convexity

$$e^{hx}\le \frac{b-x}{b-a}\,e^{ha}+\frac{x-a}{b-a}\,e^{hb},\qquad a\le x\le b.$$

This is the lever. The left side is the thing I can't average; the right side is *linear* in $x$, so I can take its expectation trivially. With $EX=0$:

$$E\,e^{hX}\le \frac{b-EX}{b-a}\,e^{ha}+\frac{EX-a}{b-a}\,e^{hb}=\frac{b}{b-a}\,e^{ha}+\frac{-a}{b-a}\,e^{hb}.$$

Good — I've reduced the MGF of an arbitrary centered bounded variable to an explicit function of $h$, with the only inputs being the endpoints $a,b$. Let me name $p=\frac{-a}{b-a}$, which is in $[0,1]$ since $a\le 0\le b$ (the interval has to contain $0$ because the mean is $0$). If $p$ is $0$ or $1$, the centered bounded variable is forced to be $0$ almost surely on one side and the bound is trivial; the same formulas still behave correctly. Then $\frac{b}{b-a}=1-p$, and the bound is $(1-p)e^{ha}+p\,e^{hb}$. Pull out $e^{ha}$ and use $a=-p(b-a)$: write $u=h(b-a)$, so $e^{ha}=e^{-pu}$ and $e^{hb}=e^{ha}e^{h(b-a)}=e^{-pu}e^{u}$. Then

$$E\,e^{hX}\le e^{-pu}\big(1-p+p\,e^{u}\big)=e^{L},\qquad L(u)=-pu+\log(1-p+p\,e^{u}).$$

Now I need to bound $L(u)$. The cleanest possible bound would be $L(u)\le u^2/8$, because then $E e^{hX}\le e^{h^2(b-a)^2/8}$, exactly the quadratic-in-$h$ sub-Gaussian form I wanted, with variance proxy $(b-a)^2/4$. Let me see if $u^2/8$ is right and where the $8$ comes from. Compute derivatives. At $u=0$: $L(0)=\log 1=0$. Good, the bound must pass through the origin. First derivative:

$$L'(u)=-p+\frac{p\,e^{u}}{1-p+p\,e^{u}}.$$

At $u=0$, $L'(0)=-p+\frac{p}{1}=0$. So $L$ has a critical point at the origin too — the constant and linear Taylor terms both vanish, and the whole bound will come from the second-order term. Second derivative — differentiate $\frac{p e^u}{1-p+pe^u}$. Let me set $q(u)=\frac{p e^u}{1-p+p e^u}$, which is a number in $[0,1]$, in fact $q$ is itself a tilted probability. Then $L'(u)=-p+q(u)$ and

$$L''(u)=q(u)\big(1-q(u)\big).$$

(That is just the derivative of a logistic: if $q=\frac{pe^u}{1-p+pe^u}$ then $q'=q(1-q)$.) Now $q(1-q)$ for $q\in[0,1]$ is maximized at $q=1/2$, where it equals $1/4$. So

$$L''(u)\le \tfrac14\quad\text{for all }u.$$

By Taylor's theorem with the integral/Lagrange remainder, using $L(0)=L'(0)=0$,

$$L(u)=\tfrac12 L''(\xi)\,u^2\le \tfrac12\cdot\tfrac14\cdot u^2=\frac{u^2}{8}.$$

There is the $8$: it is $\frac12\cdot\frac14$, the $\frac14$ from $\max q(1-q)$ and the $\frac12$ from Taylor. So, undoing $u=h(b-a)$,

$$\boxed{\,E\,e^{hX}\le \exp\!\Big(\frac{h^2(b-a)^2}{8}\Big)\,}\qquad\text{for }EX=0,\ a\le X\le b.$$

This is the lemma the whole edifice stands on. Notice what it says in MGF language: a bounded centered variable behaves, as far as its exponential moments go, *no worse than a Gaussian* of variance $(b-a)^2/4$ — even if its actual variance is much smaller. That's a slight loss (it uses only the range, not the variance), but it's the cleanest possible statement and it's exactly what I needed for the Chernoff machine.

Now assemble. Let $S=\sum_i(X_i-EX_i)$ with each $X_i\in[a_i,b_i]$. From the exponential-Markov bound and factorization,

$$\Pr\{S\ge t\}\le e^{-ht}\prod_{i=1}^n E\,e^{h(X_i-EX_i)}\le e^{-ht}\prod_{i=1}^n \exp\!\Big(\frac{h^2(b_i-a_i)^2}{8}\Big)=\exp\!\Big(-ht+\frac{h^2}{8}\sum_{i=1}^n (b_i-a_i)^2\Big).$$

The exponent is a quadratic in $h$ opening upward; minimize it. Let $D=\sum_i(b_i-a_i)^2$. The derivative $-t+\frac{h}{4}D=0$ gives $h^\star=4t/D$. Plug back:

$$-h^\star t+\frac{(h^\star)^2}{8}D=-\frac{4t^2}{D}+\frac{16t^2/D^2}{8}D=-\frac{4t^2}{D}+\frac{2t^2}{D}=-\frac{2t^2}{D}.$$

So

$$\boxed{\,\Pr\Big\{\sum_i(X_i-EX_i)\ge t\Big\}\le \exp\!\Big(-\frac{2t^2}{\sum_{i=1}^n (b_i-a_i)^2}\Big).\,}$$

That is the inequality I was after, and it is everything Chebyshev was not: exponential in $t^2$, exponential in $n$ when the ranges are comparable (then $D\asymp n$ and the exponent is $\asymp t^2/n$... wait, let me keep units straight). Let me sanity-check the scaling against the Gaussian. Take all $X_i\in[0,1]$ and look at the *average* deviation: $\Pr\{\bar X-\mu\ge t\}=\Pr\{S-ES\ge nt\}$, so I put $nt$ in place of $t$ and $D=\sum(1-0)^2=n$:

$$\Pr\{\bar X-\mu\ge t\}\le \exp\!\Big(-\frac{2(nt)^2}{n}\Big)=e^{-2nt^2}.$$

Exponential in $n$, just as the CLT promised, and with an honest explicit constant $2$. Let me double-check that constant isn't too good to be true by comparing to the actual Gaussian limit. For $X_i\in[0,1]$ the worst-case variance is $1/4$ (a fair coin), so the CLT tail is $\approx e^{-nt^2/(2\cdot 1/4)}=e^{-2nt^2}$. The constants *match exactly* in the symmetric Bernoulli case. So the bound is not merely exponential — its constant is tight against the Gaussian for the extremal two-point distribution. The $(b-a)^2/4$ variance proxy in the lemma was the fair-coin variance, and it propagated through to the right place. That is reassuring; I haven't thrown away more than I had to.

I should pin down whether I can do *better* when I actually know the variance is small. Right now the bound only sees the range $b-a$, and pays the fair-coin price $1/4$ even for a nearly-deterministic variable. If a centered summand is bounded above by $b$ and I also know its second moment, the chord trick is wasteful — I'd want an MGF bound in terms of the variance and $b$. There is such a bound (the extremal distribution is the two-point one putting mass at the negative point determined by the variance and at $b$), and pushing it through Chernoff for a sum with total variance $v$ gives a tail $\exp\!\{-\frac{v}{b^2}h(\frac{bt}{v})\}$, $h(u)=(1+u)\log(1+u)-u$ — sharper than the range-only bound whenever $v$ is much less than $\sum(b_i-a_i)^2$. That's the Bennett/Bernstein refinement, and it matters in practice, but it is a side road: it needs the variances, and the clean, assumption-light, *just-the-range* statement is the one I want as the headline. I'll keep the range-only inequality as the main object and note that the variance-aware version exists when one can afford the extra hypothesis.

Now the thing that has been nagging me through the whole derivation. Where, exactly, did I use independence? I used it in *one* place: to factor $E e^{h\sum_i Z_i}=\prod_i E e^{hZ_i}$. Everything else — Markov, the chord bound, the Taylor estimate of $L$, the minimization over $h$ — used nothing about how the $X_i$ relate to each other. And factorization is a stronger assumption than I need. To peel the product apart I don't need the $Z_i$ to be independent; I only need, at each stage, that conditioning on the past does not change the exponential moment of the next increment beyond the bound I have for it. Let me make that precise, because it is the doorway out of "sums" into "general functions".

Suppose instead of independent increments I have a *martingale*: a sequence $Z_0,Z_1,\dots,Z_n$ with differences $d_k=Z_k-Z_{k-1}$ satisfying $E[d_k\mid \mathcal F_{k-1}]=0$, where $\mathcal F_{k-1}$ is the history up to step $k-1$. "Conditional mean zero given the past" — that is exactly what independence-and-centering gave me before, but now it's the *only* thing I assume. Watch the factorization survive. By the tower property, conditioning on $\mathcal F_{n-1}$ before taking the outer expectation,

$$E\,e^{h(Z_n-Z_0)}=E\Big[e^{h(Z_{n-1}-Z_0)}\,E\big[e^{h d_n}\mid \mathcal F_{n-1}\big]\Big],$$

because $Z_{n-1}-Z_0$ is determined by $\mathcal F_{n-1}$ and pulls out of the inner conditional expectation. Now suppose I can bound the *conditional* MGF of the increment: $E[e^{h d_n}\mid \mathcal F_{n-1}]\le e^{h^2 c_n^2/2}$ almost surely, a deterministic bound. Then the inner factor comes out as a constant and

$$E\,e^{h(Z_n-Z_0)}\le e^{h^2 c_n^2/2}\,E\,e^{h(Z_{n-1}-Z_0)}.$$

Peel again, and again — each step strips one increment and contributes one factor $e^{h^2 c_k^2/2}$:

$$E\,e^{h(Z_n-Z_0)}\le \exp\!\Big(\frac{h^2}{2}\sum_{k=1}^n c_k^2\Big).$$

That is the same sub-Gaussian MGF bound as before, but it never assumed independence — only the martingale-difference property and a conditional MGF bound per step. And the conditional MGF bound is supplied by the very lemma I already proved: if, given the past, $d_k$ is centered (it is, that's the martingale property) and confined to an interval, the chord bound applies *conditionally* and gives the quadratic. Run Chernoff on this:

$$\Pr\{Z_n-Z_0\ge t\}\le e^{-ht}\exp\!\Big(\frac{h^2}{2}\sum_k c_k^2\Big),\quad\text{minimized at }h=\frac{t}{\sum_k c_k^2},$$

$$\boxed{\,\Pr\{Z_n-Z_0\ge t\}\le \exp\!\Big(-\frac{t^2}{2\sum_{k=1}^n c_k^2}\Big),\,}$$

and by applying the same to $-Z$, a two-sided $2\exp(-t^2/(2\sum c_k^2))$. This is the Azuma–Hoeffding inequality: a martingale whose increments are conditionally controlled is as concentrated as a sum of independent bounded variables. I should fix the bookkeeping on $c_k$, because there are two natural conventions and they differ by a factor of $2$ that I must not fumble. If I assume $|d_k|\le c_k$, then conditionally $d_k$ lives in an interval of *length* $2c_k$, so the chord lemma gives $E[e^{hd_k}\mid\mathcal F_{k-1}]\le e^{h^2(2c_k)^2/8}=e^{h^2 c_k^2/2}$ — that's where the $/2$ in the per-step factor comes from, and it's consistent with what I wrote. (If instead I phrase the hypothesis as "$d_k$ lies in an interval of *length* $c_k$", the per-step factor is $e^{h^2 c_k^2/8}$ and the final exponent is $-2t^2/\sum c_k^2$. Same inequality, different name for $c_k$. I'll keep $|d_k|\le c_k$ for Azuma and "range $\le c_k$" for the function version below, and stay careful.) Let me also notice the cosh route, since it's the most transparent way to see the conditional bound when the increment is symmetric-ish: for a centered variable supported in $[-c,c]$ with $c>0$, convexity gives $e^{hd}\le \cosh(hc)+\frac{d}{c}\sinh(hc)$, and taking the conditional expectation kills the $\sinh$ term (conditional mean zero), leaving $E[e^{hd}\mid\mathcal F_{k-1}]\le \cosh(hc)$. And $\cosh(z)=\sum_{m\ge0}\frac{z^{2m}}{(2m)!}\le \sum_{m\ge0}\frac{z^{2m}}{2^m m!}=e^{z^2/2}$, using $(2m)!\ge 2^m m!$. So $\cosh(hc)\le e^{h^2c^2/2}$ — the same factor, derived purely from the series. Either way I land on $e^{h^2 c_k^2/2}$ per step.

I started wanting to control a *sum*, but the martingale argument used only conditional-mean-zero increments, and *any* function of independent variables can be written as such a sum. Let $f(X_1,\dots,X_n)$ be the quantity I care about and set $Z=f(X_1,\dots,X_n)$. Reveal the inputs one at a time and look at my best forecast of $Z$ given what I've seen:

$$Z_i=E[\,Z\mid X_1,\dots,X_i\,],\qquad i=0,1,\dots,n,$$

so $Z_0=E[Z]$ (nothing revealed) and $Z_n=Z$ (everything revealed). This is the Doob martingale, the filtration generated by revealing coordinates. Its differences

$$\Delta_i=E[Z\mid X_1,\dots,X_i]-E[Z\mid X_1,\dots,X_{i-1}]$$

automatically satisfy $E[\Delta_i\mid X_1,\dots,X_{i-1}]=0$ — that's the tower property again, the conditional mean of a one-step-finer forecast equals the coarser forecast. So $\{Z_i\}$ is a martingale and $Z-EZ=\sum_{i=1}^n \Delta_i$ telescopes. I get concentration of $f$ for free *if* I can bound each $\Delta_i$.

What controls $\Delta_i$? This is where I need a hypothesis on $f$, and it should be the weakest natural one: that no single coordinate can move $f$ much. Say $f$ has the **bounded differences** property — there are constants $c_1,\dots,c_n$ with

$$\big|f(x_1,\dots,x_i,\dots,x_n)-f(x_1,\dots,x_i',\dots,x_n)\big|\le c_i$$

for all coordinates, whenever you change only the $i$-th input. (Equivalently: $f$ is $1$-Lipschitz in the weighted Hamming metric $\sum_i c_i\mathbf 1\{x_i\ne y_i\}$.) Now I claim that, *conditionally on $X_1,\dots,X_{i-1}$*, the increment $\Delta_i$ lives in an interval of length at most $c_i$. Why: $\Delta_i$ is $E[Z\mid X_1,\dots,X_i]-E[Z\mid X_1,\dots,X_{i-1}]$, and the second term is the first averaged over $X_i$. As $X_i$ ranges over its values (with $X_{i+1},\dots,X_n$ still integrated out independently), the conditional expectation $E[Z\mid X_1,\dots,X_{i-1},X_i=x_i]$ can change by at most $c_i$, because changing only the $i$-th coordinate changes $f$ — hence its average over the remaining coordinates — by at most $c_i$. The supremum-over-$x_i$ minus the infimum-over-$x_i$ of that conditional expectation is $\le c_i$. Subtracting its mean over $X_i$, the increment $\Delta_i$ is a centered random variable whose conditional support has width $\le c_i$. That's precisely the hypothesis the conditional chord lemma needs.

So apply the lemma conditionally: $E[e^{h\Delta_i}\mid X_1,\dots,X_{i-1}]\le e^{h^2 c_i^2/8}$ (range $c_i$, so the factor is $c_i^2/8$, not $c_i^2/2$ — here I'm using the "interval of length $c_i$" convention, $\Delta_i$ is centered in a window of width $c_i$). Peel the Doob martingale exactly as before:

$$E\,e^{h(Z-EZ)}=E\Big[e^{h\sum_{i<n}\Delta_i}\,E[e^{h\Delta_n}\mid X_1,\dots,X_{n-1}]\Big]\le e^{h^2c_n^2/8}\,E\,e^{h\sum_{i<n}\Delta_i}\le\dots\le \exp\!\Big(\frac{h^2}{8}\sum_{i=1}^n c_i^2\Big).$$

Chernoff and minimize over $h$ ($h^\star=4t/\sum c_i^2$, the same arithmetic as the sum case):

$$\boxed{\,\Pr\{f(X_1,\dots,X_n)-Ef\ge t\}\le \exp\!\Big(-\frac{2t^2}{\sum_{i=1}^n c_i^2}\Big),\,}$$

and two-sided with the factor $2$. This is the bounded-differences inequality. It is *distribution-free* — beyond independence, nothing about the laws of the $X_i$ enters, only the sensitivities $c_i$ of $f$. And it contains the sum case as the trivial special case: if $f(x)=\sum_i x_i$ with $x_i\in[a_i,b_i]$, then changing the $i$-th coordinate changes $f$ by at most $b_i-a_i$, so $c_i=b_i-a_i$ and I recover $\exp(-2t^2/\sum(b_i-a_i)^2)$ exactly. The general inequality didn't cost me anything in the special case — it just freed me from needing $f$ to be additive at all.

Let me re-walk the constant for $f$ once more, because I flipped conventions mid-stream and I want it airtight. For the *sum* and for the *function* version I bound a centered increment confined to a window of width $w$ (=$b_i-a_i$ for sums, =$c_i$ for $f$); the chord lemma gives MGF factor $e^{h^2 w^2/8}$; the product over $n$ steps gives $\exp(\frac{h^2}{8}\sum w^2)$; minimizing $-ht+\frac{h^2}{8}\sum w^2$ at $h=4t/\sum w^2$ gives exponent $-2t^2/\sum w^2$. For *Azuma* I phrased the hypothesis as $|d_k|\le c_k$, i.e. width $2c_k$, giving factor $e^{h^2(2c_k)^2/8}=e^{h^2c_k^2/2}$, product $\exp(\frac{h^2}{2}\sum c_k^2)$, minimizer $h=t/\sum c_k^2$, exponent $-t^2/(2\sum c_k^2)$. Both are the *same* statement; they only differ in whether $c$ names the half-width or the full width. Everything is consistent: $-2t^2/\sum(\text{width})^2 = -t^2/(2\sum(\text{half-width})^2)$.

One more thing I want to make sure I didn't leave behind. The exponential-moment argument, even in the pure-sum case, never used the *full* joint distribution — and in fact the running maximum is controlled too, not just the endpoint. Since $e^{h(S_m-ES_m)}$ is a submartingale in $m$ (the exponential of a martingale, $e^{hx}$ convex), Doob's maximal inequality gives $\Pr\{\max_{m\le n}(S_m-ES_m)\ge t\}$ the *same* bound $e^{-ht}E e^{h(S_n-ES_n)}$, so all my tail bounds upgrade for free to bounds on the maximal partial-sum deviation — the analogue of Kolmogorov's maximal inequality. And the same proof goes through verbatim if the centered partial sums $S_m-ES_m$ form a martingale rather than a sum of independent variables; I already saw that, it's the Azuma case in disguise.

The whole thing is really one idea applied three times. Markov on the exponential turns a tail probability into a moment generating function and a free parameter $h$ to optimize — that's Chernoff, and the optimizer is the Legendre transform of the log-MGF, with the Gaussian giving the target rate $e^{-t^2/(2v)}$. A bounded centered variable has its MGF pinned below the Gaussian one by convexity of the exponential: the chord bound reduces it to $L(u)=-pu+\log(1-p+pe^u)$, whose second derivative is $q(1-q)\le 1/4$, so $L(u)\le u^2/8$ and $Ee^{hX}\le e^{h^2(b-a)^2/8}$ — Hoeffding's lemma. Feed that into Chernoff over an independent sum and optimize: $\exp(-2t^2/\sum(b_i-a_i)^2)$ — Hoeffding's inequality. Now notice the only use of independence was to factor the MGF, and the martingale-difference property (conditional mean zero) lets the factorization survive one increment at a time via the tower property, with the chord lemma applied conditionally: $\exp(-t^2/(2\sum c_k^2))$ — Azuma–Hoeffding. Finally, *any* function of independent inputs becomes a martingale by revealing the coordinates one at a time (the Doob martingale), and the bounded-differences property forces each conditional increment into a window of width $c_i$, so the same bound applies: $\exp(-2t^2/\sum c_i^2)$ — the bounded-differences inequality. A function of many independent variables that is not too sensitive to any one of them is sharply, sub-Gaussianly concentrated around its mean, and the only thing I ever had to control was a one-dimensional MGF of a bounded centered variable.

I can now write the final statements in the same notation, with the constants exposed.

```text
LEMMA (bounded-variable MGF, "Hoeffding's lemma").
  Let X be a random variable with EX = 0 and a <= X <= b a.s. Then for every real h,
      E e^{hX} <= exp( h^2 (b-a)^2 / 8 ).
  Proof. Convexity of x -> e^{hx} gives, for a <= x <= b,
      e^{hx} <= (b-x)/(b-a) e^{ha} + (x-a)/(b-a) e^{hb}.
  Take E (EX=0):  E e^{hX} <= (b/(b-a)) e^{ha} + (-a/(b-a)) e^{hb} = e^{L(u)},
  with p = -a/(b-a) in [0,1], u = h(b-a), L(u) = -p u + log(1 - p + p e^u).
  L(0)=0, L'(0)=0, and L''(u) = q(1-q) with q = p e^u/(1-p+p e^u) in [0,1],
  so L''(u) <= 1/4.  Taylor: L(u) = (1/2) L''(xi) u^2 <= u^2/8.  QED.

THEOREM (Hoeffding's inequality).
  X_1,...,X_n independent, X_i in [a_i,b_i].  S = sum_i (X_i - E X_i).  For t > 0,
      Pr{ S >= t } <= exp( - 2 t^2 / sum_i (b_i - a_i)^2 ).
  Proof. Markov on the exponential and independence:
      Pr{S>=t} <= e^{-h t} prod_i E e^{h(X_i - EX_i)} <= exp( -h t + (h^2/8) sum_i (b_i-a_i)^2 ),
  the last by the Lemma.  Minimize over h>0 at h = 4 t / sum_i (b_i-a_i)^2;
  the exponent becomes -2 t^2 / sum_i (b_i-a_i)^2.  QED.
  (For X_i in [0,1], the average obeys  Pr{ Xbar - mu >= t } <= e^{ -2 n t^2 }.)

THEOREM (Azuma-Hoeffding, martingale version).
  Z_0,...,Z_n a martingale, differences d_k = Z_k - Z_{k-1} with E[d_k | F_{k-1}] = 0
  and |d_k| <= c_k a.s.  For t > 0,
      Pr{ Z_n - Z_0 >= t } <= exp( - t^2 / ( 2 sum_k c_k^2 ) ),
  and Pr{ |Z_n - Z_0| >= t } <= 2 exp( - t^2 / (2 sum_k c_k^2) ).
  Proof. Conditionally, d_k is centered in an interval of length 2 c_k, so by the
  Lemma  E[ e^{h d_k} | F_{k-1} ] <= e^{ h^2 (2 c_k)^2 / 8 } = e^{ h^2 c_k^2 / 2 }.
  Tower property, peeling one increment at a time:
      E e^{h(Z_n - Z_0)} = E[ e^{h(Z_{n-1}-Z_0)} E[e^{h d_n}|F_{n-1}] ]
                        <= e^{ h^2 c_n^2 / 2 } E e^{h(Z_{n-1}-Z_0)}
                        <= ... <= exp( (h^2/2) sum_k c_k^2 ).
  Chernoff: Pr{Z_n-Z_0>=t} <= exp( -h t + (h^2/2) sum c_k^2 ), minimized at
  h = t / sum c_k^2, giving exponent - t^2/(2 sum c_k^2).  QED.

THEOREM (bounded-differences / McDiarmid's inequality).
  X_1,...,X_n independent.  f satisfies the bounded-differences condition:
  changing only coordinate i changes f by at most c_i.  Z = f(X_1,...,X_n).  For t>0,
      Pr{ Z - E Z >= t } <= exp( - 2 t^2 / sum_i c_i^2 ),
  and the two-sided bound carries a factor 2.
  Proof. Doob martingale: Z_i = E[ Z | X_1,...,X_i ], so Z_0 = EZ, Z_n = Z, and
  Delta_i = Z_i - Z_{i-1} has E[Delta_i | X_1,...,X_{i-1}] = 0 (tower property).
  Bounded differences => conditionally on X_1,...,X_{i-1}, Delta_i lies in an interval
  of width <= c_i (sup minus inf over X_i of E[Z | X_1..X_{i-1}, X_i], averaged over the rest).
  Apply the Lemma conditionally: E[ e^{h Delta_i} | X_1..X_{i-1} ] <= e^{ h^2 c_i^2 / 8 }.
  Peel the Doob martingale: E e^{h(Z - EZ)} <= exp( (h^2/8) sum_i c_i^2 ).
  Chernoff, minimize at h = 4 t / sum_i c_i^2: exponent - 2 t^2 / sum_i c_i^2.  QED.
  (f = sum_i X_i with X_i in [a_i,b_i] has c_i = b_i - a_i, recovering Hoeffding.)
```
