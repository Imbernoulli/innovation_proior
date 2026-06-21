# Context

## Research question

Let $X_1,\dots,X_n$ be independent random variables and let $S = X_1 + \dots + X_n$. The law of large numbers says the average $\bar X = S/n$ converges to its mean $\mu = ES/n$, and the central limit theorem says the fluctuations are of order $1/\sqrt n$ and asymptotically Gaussian. Both are *asymptotic* statements. The quantitative question is: for a **fixed** $n$ and a **fixed** deviation $t>0$, how large can

$$\Pr\{\bar X - \mu \ge t\} = \Pr\{S - ES \ge nt\}$$

be? One wants a numerical upper bound that holds at finite sample size — a guarantee into which one can plug a confidence level and read off how many samples are enough.

The setting extends beyond sums. One may ask the same question for a general **function** $f(X_1,\dots,X_n)$ of many independent variables. Sums, maxima, the length of the longest common subsequence, the chromatic number of a random graph, the error of an empirical estimate — all are such functions. The task is to produce a theorem with explicit constants that quantifies how such quantities deviate from their expectation.

## Background

**Markov's inequality.** The atom from which everything is built. For a nonnegative random variable $Y$ and any $t>0$, the pointwise bound $Y\,\mathbf 1\{Y\ge t\}\ge t\,\mathbf 1\{Y\ge t\}$ gives, on taking expectations,
$$\Pr\{Y\ge t\}\le \frac{EY}{t}.$$
Applied not to $Y$ directly but to $\phi(Y)$ for a nondecreasing nonnegative $\phi$, it yields $\Pr\{Y\ge t\}\le E\phi(Y)/\phi(t)$ for any $t$ with $\phi(t)>0$.

**Chebyshev / Bienaymé.** Taking $\phi(t)=t^2$ and $Y=|Z-EZ|$ gives $\Pr\{|Z-EZ|\ge t\}\le \operatorname{Var}(Z)/t^2$. For a sum of independent variables $\operatorname{Var}(S)=\sum_i \operatorname{Var}(X_i)$, so $\Pr\{|\bar X-\mu|\ge t\}\le \sigma^2/(nt^2)$ with $\sigma^2=n^{-1}\sum\operatorname{Var}(X_i)$. This is the standard known bound; it requires only a finite variance and no boundedness, and decays like $1/t^2$. More generally $\Pr\{|Z-EZ|\ge t\}\le E|Z-EZ|^q/t^q$, and one may optimize over $q$.

**The exponential-moment idea (Bernstein, Cramér).** Used apparently first by S. N. Bernstein: the indicator $\mathbf 1\{S-ES\ge nt\}$ never exceeds $\exp\{h(S-ES-nt)\}$ for any constant $h>0$. Hence $\Pr\{S-ES\ge nt\}\le e^{-hnt}\,E e^{h(S-ES)}$, and by independence $E e^{h(S-ES)} = \prod_i E e^{h(X_i-EX_i)}$: the exponential turns a *sum* into a *product* of moment generating functions. The asymptotic face of this device is Cramér's large-deviation theory, in which the exponential rate is governed by the Legendre transform of the log-MGF. The programme rests on controlling the MGF.

**Convexity / Jensen.** A continuous function $f$ is convex on an interval if it lies below its chords: $f(px+(1-p)y)\le pf(x)+(1-p)f(y)$ for $0<p<1$; equivalently it has a nonnegative second derivative. Jensen's inequality $f(\sum p_i x_i)\le \sum p_i f(x_i)$ for a probability weighting $p_i$ is the discrete form. The exponential is convex, so on any bounded interval its graph lies below the chord joining its two endpoints.

**Martingales and Doob's filtration.** A sequence $S'_1,\dots,S'_n$ is a martingale if $E[S'_m\mid S'_1,\dots,S'_j]=S'_j$ for $j\le m$: the best forecast of the future given the past is the present. Doob's maximal inequality controls $\Pr\{\max_{m\le n}(S_m-ES_m)\ge nt\}$ by the same exponential-moment bound as the endpoint, because $\exp\{h(S_m-ES_m)\}$ is a submartingale (Doob, *Stochastic Processes*, 1953).

**Variance-aware refinements (Bennett, Bernstein, Prohorov).** When the variance of a bounded summand is much smaller than the square of its range, bounds that use the variance — Bennett's Poisson-type rate, Bernstein's Gaussian-to-exponential simplification, and Prohorov's arcsinh-type refinement — give sharper rates than bounds depending on the range alone. These are the natural comparison points for a range-based bound.

## Baselines

- **Chebyshev's inequality** (Bienaymé 1853 / Chebyshev). $\Pr\{|\bar X-\mu|\ge t\}\le \sigma^2/(nt^2)$. Core idea: second moment + Markov. Distribution-free given a finite variance; needs no boundedness. Decays polynomially in $t$.

- **Bernstein's inequality** (S. N. Bernstein). Core idea: the exponential-moment method with a variance-sensitive quadratic simplification; for mean-zero summands with $|X_i|\le b$ and $v=\sum_i\operatorname{Var}(X_i)$, a standard form is $\Pr\{S\ge t\}\le \exp\{-t^2/(2(v+bt/3))\}$. Gives a tail interpolating between Gaussian (small $t$) and exponential (large $t$). Stated for sums of independent variables, using control of the variances.

- **Bennett's inequality** (Bennett 1962). Core idea: same exponential-moment route; for mean-zero $X_i\le b$ with total variance $v=\sum_i\operatorname{Var}(X_i)$, $\log E e^{\lambda S}\le \frac{v}{b^2}\phi(b\lambda)$ with $\phi(u)=e^u-u-1$, giving $\Pr\{S\ge t\}\le \exp\!\big(-\frac{v}{b^2}h(bt/v)\big)$, $h(u)=(1+u)\log(1+u)-u$. The tightest of the classical range/variance bounds; reduces to a Poisson tail. A statement about sums, using the variance.

- **Prohorov's inequality** (Prohorov). Core idea: another variance-sensitive exponential bound for bounded summands, with an arcsinh-type rate expression. A variance-dependent bound for sums.

- **Doob's martingale maximal inequality** (Doob 1953). Core idea: for a martingale, the running maximum obeys the same exponential-moment tail as the endpoint, given an MGF bound for the increments.

## Evaluation settings

This is a theoretical result; the natural yardsticks are mathematical, not benchmark datasets.

- **Comparison targets.** The Chebyshev bound $\sigma^2/(nt^2)$, the Gaussian CLT tail $\bar\Phi(\sqrt n\, t/\sigma)\le e^{-nt^2/(2\sigma^2)}$, and the variance-aware bounds of Bennett, Bernstein and Prohorov. A candidate bound is judged by whether it decays exponentially in $n$ for fixed $t$, by the constant in the exponent, and by whether the constant is *attained* by an extremal distribution (e.g. a two-point or binomial distribution).

- **Canonical test functions for a function-of-independent-variables bound.** Sums and weighted sums; the empirical mean of a bounded loss; sampling without replacement from a finite population; $U$-statistics; combinatorial quantities on random structures (e.g. a low-sensitivity function on a product space, where no single coordinate can move the output far). These are the objects whose concentration one would want any general inequality to deliver as special cases.

- **Diagnostics.** Whether the bound recovers known special cases (the binomial bounded-sum case, the Rademacher/sub-Gaussian case where $\operatorname{Var}(S)=\sum\alpha_i^2$), and whether the limiting behaviour as $t\to0$ or $n\to\infty$ matches the Gaussian/large-deviation rate.
