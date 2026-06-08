Let me start from the thing that actually bothers me. Shannon's theorem tells me that below capacity I can communicate as reliably as I please, and above capacity I can't. Good — but that is a statement about a limit, $P_e \to 0$ as the block length $N \to \infty$. It tells me nothing an engineer can hold onto. If I pick a rate $R < C$ and I want $P_e = 10^{-6}$, how long does my block have to be? And if I have two modulation schemes with the same capacity, which one should I code over? The capacity number is silent on both. What I really want is a *speed*: a positive function $E(R)$, computable from the channel's transition probabilities, such that the best codes drive the error down like $e^{-N E(R)}$. Then the asymptotic limit becomes a law I can plan around, and I can compare whole channels by comparing their $E(R)$ curves instead of a single scalar.

So the question is not "is reliable communication possible" — Shannon answered that — but "how fast, as a function of rate." And I want the answer for *any* discrete memoryless channel, in one clean argument, not a separate epic for the binary symmetric channel and another for Gaussian noise. Elias managed exponentially decaying upper and lower bounds for the BSC, and they even coincide over a band of rates up to capacity, which is gorgeous — but it is the BSC, and the construction does not obviously carry over. Fano has general bounds for an arbitrary DMC, and he sandwiches $P_e$ between $e^{-N[E_L(R)+o(1)]}$ and $2e^{-NE(R)}$ with both exponents positive below $C$ and equal just under capacity. But his derivation is long and heavy; the exponent that comes out is hard to compute and hard to read. I want something I can derive on a page and evaluate almost as easily as capacity itself.

Let me set up the object honestly. I have a code of $M$ words $x_1,\dots,x_M$, each a length-$N$ input string, and I decode by maximum likelihood: given the received $y$, I pick the $m$ maximizing $P(y\mid x_m)$. That is the right decoder — for equiprobable messages it minimizes error — so it is the one I should bound. When does it fail when $x_m$ is sent? When some other word looks at least as good: there's an $m' \ne m$ with $P(y\mid x_{m'}) \ge P(y\mid x_m)$. Let me write the error probability for message $m$ exactly as

$$P_{e,m} = \sum_y P(y\mid x_m)\, \phi_m(y), \qquad \phi_m(y) = \begin{cases}1 & \text{if } P(y\mid x_{m'}) \ge P(y\mid x_m)\ \text{for some } m'\ne m\\ 0 & \text{otherwise.}\end{cases}$$

The trouble is right there in $\phi_m$: it depends on *all* the other codewords at once, jointly. I cannot evaluate it for a specific clever code without knowing the whole code, and there is no specific clever code I can write down and analyze for general $N$.

The way out is the move Shannon and Elias already taught me: stop trying to analyze a particular code. Build an *ensemble* — draw each codeword at random, and not just each word but each *letter* independently, from some input distribution $p(x)$ — and average. The average error over the ensemble is something I can compute because the randomness factorizes; and since at least one code in the ensemble is at least as good as the average, whatever bound I prove for the average is achieved by *some* actual code. That is the whole trick of existence-by-averaging. Why draw the letters independently rather than whole words? Because a length-$N$ block is really $N$ uses of the same channel side by side, and if I treat the block as one big product channel the natural ensemble that respects the product structure is i.i.d. letters; later I'll see this is exactly what makes everything factorize over $n$. And Elias already showed the average over such an ensemble is essentially as good as the best code, so I am not giving anything up by averaging.

Now, how do I bound $\phi_m$? The reflex is the union bound: the event "some $m'$ beats me" is contained in the union over $m'$, so

$$\phi_m(y) \le \sum_{m'\ne m} \mathbf 1\!\left[P(y\mid x_{m'}) \ge P(y\mid x_m)\right].$$

And each indicator I can soften: if $P(y\mid x_{m'}) \ge P(y\mid x_m)$ then the ratio $\big(P(y\mid x_{m'})/P(y\mid x_m)\big)^{1/2} \ge 1$, and if the inequality fails the ratio is still nonnegative, so the indicator is $\le \sqrt{P(y\mid x_{m'})/P(y\mid x_m)}$. Sum over $y$ against $P(y\mid x_m)$ and the pairwise term becomes the Bhattacharyya sum $\sum_y \sqrt{P(y\mid x_m)P(y\mid x_{m'})}$. Average that over the independent random words and I get a clean exponent. Let me see what it is: with $M-1 < e^{NR}$ competitors, the exponent works out to $E_0(1,p) - R$ where $E_0(1,p) = -\ln \sum_j \big(\sum_k p_k \sqrt{P_{jk}}\big)^2$.

But this is one line, of slope $-1$. Its exponent hits zero at the rate $R_0$ where $E_0(1,p) = R$ — the cutoff rate — and $R_0$ is strictly below $C$. So this argument certifies a positive exponent only up to $R_0$, and says nothing for $R_0 < R < C$, exactly the high-rate regime I care about most. I hit the wall here, and the reason is concrete: the union bound counts every competitor as if it were the sole threat, and at high rate, where there are exponentially many plausible competitors, that overcounting is catastrophic — I am bounding a probability (which can't exceed $1$) by a huge sum that has long since blown past $1$. The square root, the $s = 1/2$ tilt, was an arbitrary choice; it gave me one slope. I need a knob.

So let me not commit to the union bound's "sum of indicators" shape. Let me keep a free parameter. I want to overbound the indicator $\phi_m(y)$, which is either $0$ or $1$, by something built from the competitors that (a) is always nonnegative, so it automatically covers the $\phi_m = 0$ case, and (b) is $\ge 1$ whenever $\phi_m = 1$. The union-bound sum does this, but it does it *linearly*. What if I raise the whole competitor-sum to a power $\rho$? Try

$$\phi_m(y) \le \left[\frac{\sum_{m'\ne m} P(y\mid x_{m'})^{1/(1+\rho)}}{P(y\mid x_m)^{1/(1+\rho)}}\right]^{\rho}, \qquad \rho > 0.$$

Let me check it actually bounds the indicator. The right side is a ratio of nonnegative things raised to a positive power, so it is always $\ge 0$ — the $\phi_m = 0$ case is handled for free. Now suppose $\phi_m(y) = 1$, i.e. some particular $m'$ has $P(y\mid x_{m'}) \ge P(y\mid x_m)$. Then that one term in the numerator, $P(y\mid x_{m'})^{1/(1+\rho)}$, is already $\ge P(y\mid x_m)^{1/(1+\rho)}$, the denominator; so the bracket (a sum that includes that term, over nonnegative things) is $\ge 1$, and raising something $\ge 1$ to the power $\rho > 0$ keeps it $\ge 1$. So the bound holds in both cases, for *every* $\rho > 0$. And notice the union bound is the special case $\rho = 1$ with exponent $1/(1+\rho) = 1/2$ — the Bhattacharyya tilt was just one point on a family I now control.

Why the exponent $1/(1+\rho)$ specifically, and not just any tilt? Because I want the sent word and the competing words to collapse to the same averaged object. Substitute this bound for $\phi_m$ into $P_{e,m} = \sum_y P(y\mid x_m)\phi_m(y)$. The denominator is also raised to the outer power $\rho$, so it contributes $P(y\mid x_m)^{\rho/(1+\rho)}$ downstairs; the likelihood $P(y\mid x_m)$ out front leaves $P(y\mid x_m)^{1-\rho/(1+\rho)} = P(y\mid x_m)^{1/(1+\rho)}$. The competitors appear as $\big(\sum_{m'} P(y\mid x_{m'})^{1/(1+\rho)}\big)^\rho$. Now the sent word and each competing word carry the same inner exponent $1/(1+\rho)$ before averaging, so the two averages will be copies of the same sum. Any other split would leave two different single-letter objects and the clean collapse would be gone. Concretely:

$$P_{e,m} \le \sum_y P(y\mid x_m)^{1/(1+\rho)}\left[\sum_{m'\ne m} P(y\mid x_{m'})^{1/(1+\rho)}\right]^{\rho}, \qquad \rho > 0.$$

This is still a bound for a fixed code. Now average over the ensemble — put a bar over everything. The factor $P(y\mid x_m)^{1/(1+\rho)}$ depends only on word $m$; the bracket depends only on the *other* words; and all words are drawn independently. The average of a sum is the sum of averages, and the average of a product of independent random variables is the product of the averages — so the bar splits across the $m$-factor and the bracket-factor:

$$\overline{P_{e,m}} \le \sum_y \overline{P(y\mid x_m)^{1/(1+\rho)}} \cdot \overline{\left[\sum_{m'\ne m} P(y\mid x_{m'})^{1/(1+\rho)}\right]^{\rho}}.$$

The first bar I can evaluate directly: $\overline{P(y\mid x_m)^{1/(1+\rho)}} = \sum_x P(x)\, P(y\mid x)^{1/(1+\rho)}$, since word $m$ is drawn from $P(x)$. The obstruction is the second bar — the average of a *sum raised to the $\rho$ power*. I can't split a power across a sum of dependent-looking terms directly. But here is where the free parameter has to earn its keep. The function $\xi \mapsto \xi^{\rho}$ — when is it concave? Its second derivative is $\rho(\rho-1)\xi^{\rho-2}$, which is $\le 0$ exactly when $0 < \rho \le 1$. For those $\rho$, $\xi^{\rho}$ is concave, and Jensen's inequality runs the friendly way: the average of $\xi^{\rho}$ is *at most* the $\rho$-power of the average,

$$\overline{\xi^{\rho}} \le \big(\overline{\xi}\big)^{\rho}, \qquad 0 < \rho \le 1.$$

So I pay for the freedom by restricting $\rho$ to $[0,1]$ — and in exchange I get to pull the bar *inside* the bracket. Now $\overline{\sum_{m'\ne m} P(y\mid x_{m'})^{1/(1+\rho)}}$ is a sum of $M-1$ identical averages, each equal to $\sum_x P(x)P(y\mid x)^{1/(1+\rho)}$, so the bracket becomes $\big[(M-1)\sum_x P(x)P(y\mid x)^{1/(1+\rho)}\big]^{\rho}$. Combining with the first factor:

$$\overline{P_{e,m}} \le (M-1)^{\rho} \sum_y \left[\sum_x P(x)\, P(y\mid x)^{1/(1+\rho)}\right]^{1+\rho}, \qquad 0 < \rho \le 1.$$

The same averaged quantity appears once from the sent-word factor and $\rho$ more times from the Jensen-bounded competitor sum, so it becomes the $(1+\rho)$-power of a single inner sum, $\sum_x P(x)P(y\mid x)^{1/(1+\rho)}$. *That* is what the symmetric choice bought me: one clean expression, valid for any channel — memoryless or not — and any input distribution $P(x)$, with $\rho$ free in $(0,1]$.

Now let the channel be memoryless, $P(y\mid x) = \prod_{n=1}^N P(y_n\mid x_n)$, and let the ensemble draw letters i.i.d., $P(x) = \prod_n p(x_n)$. Watch the inner object: $\sum_x P(x)P(y\mid x)^{1/(1+\rho)} = \sum_{x_1,\dots,x_N}\prod_n p(x_n) P(y_n\mid x_n)^{1/(1+\rho)} = \prod_n \big[\sum_{x_n} p(x_n)P(y_n\mid x_n)^{1/(1+\rho)}\big]$ — a product over the $N$ letters. Raise it to $1+\rho$ and sum over $y = (y_1,\dots,y_N)$, and because the sum over $y$ also factorizes, the whole thing becomes the $N$-th power of the single-letter quantity. Writing the input letters $a_k$ with $p(a_k) = p_k$ and the transition probabilities $P_{jk} = P(b_j\mid a_k)$, the single-letter sum is $\sum_j \big(\sum_k p_k P_{jk}^{1/(1+\rho)}\big)^{1+\rho}$, and

$$\overline{P_{e,m}} \le (M-1)^{\rho}\left[\sum_{j=1}^{J}\Big(\sum_{k=1}^{K} p_k P_{jk}^{1/(1+\rho)}\Big)^{1+\rho}\right]^{N}.$$

Set $M = e^{NR}$, bound $M - 1 \le M$, and take logs. Define

$$E_0(\rho,p) = -\ln \sum_{j=1}^{J}\left(\sum_{k=1}^{K} p_k\, P_{jk}^{1/(1+\rho)}\right)^{1+\rho},$$

so that the bracketed $N$-th power is $e^{-N E_0(\rho,p)}$ and $(M-1)^{\rho} \le e^{N\rho R}$. Then

$$\overline{P_{e,m}} \le \exp\!\big[-N\big(E_0(\rho,p) - \rho R\big)\big], \qquad 0 \le \rho \le 1.$$

Since the right side no longer depends on $m$, the *ensemble-average* error is bounded by it, and at least one code in the ensemble does at least this well. So there exists a code of block length $N$ and rate $R$ whose ML decoding error obeys $P_e \le \exp[-N(E_0(\rho,p) - \rho R)]$ — for every $\rho \in [0,1]$ and every input distribution $p$. I am free to optimize, and I should, because each $\rho$ and $p$ gives a different bound and I want the best:

$$E(R) = \max_{0 \le \rho \le 1,\; p}\big[E_0(\rho,p) - \rho R\big], \qquad P_e \le e^{-N E(R)}.$$

Now the question that decides whether any of this was worth it: is $E(R) > 0$ for every $R < C$? If the exponent collapses to zero before capacity — like the cutoff-rate line did — I've gained nothing in the regime I cared about. Everything hinges on how $E_0(\rho,p)$ behaves as $\rho$ leaves $0$. So let me look at small $\rho$.

At $\rho = 0$: the inner exponent $1+\rho = 1$ and $1/(1+\rho) = 1$, so the inner sum is $\sum_k p_k P_{jk}$, the output marginal $q_j$, and $\sum_j q_j = 1$, so $E_0(0,p) = -\ln 1 = 0$. So the exponent $E_0 - \rho R$ starts at $0$ when $\rho = 0$ regardless of $R$ — no help yet. The decisive quantity is the *slope* in $\rho$ at the origin. Let me differentiate $E_0(\rho,p) = -\ln F(\rho)$ with $F(\rho) = \sum_j \big(\sum_k p_k P_{jk}^{1/(1+\rho)}\big)^{1+\rho}$. Write $s = 1/(1+\rho)$ and $g_j(\rho) = \sum_k p_k P_{jk}^{s}$, so $F = \sum_j g_j^{1+\rho}$. Differentiating the $j$-th term, $\frac{d}{d\rho} g_j^{1+\rho} = g_j^{1+\rho}\big[\ln g_j + (1+\rho)\,g_j^{-1}\, g_j'\big]$, and $g_j' = \frac{ds}{d\rho}\sum_k p_k P_{jk}^{s}\ln P_{jk}$ with $\frac{ds}{d\rho} = -(1+\rho)^{-2}$. At $\rho = 0$: $s = 1$, $g_j(0) = \sum_k p_k P_{jk} = q_j$, $\frac{ds}{d\rho} = -1$, so $g_j'(0) = -\sum_k p_k P_{jk}\ln P_{jk}$, and the bracket is $\ln q_j + q_j^{-1}\big(-\sum_k p_k P_{jk}\ln P_{jk}\big)$. Thus

$$F'(0) = \sum_j q_j\Big[\ln q_j - \tfrac{1}{q_j}\sum_k p_k P_{jk}\ln P_{jk}\Big] = \sum_j\Big[q_j\ln q_j - \sum_k p_k P_{jk}\ln P_{jk}\Big] = -\sum_{j,k} p_k P_{jk}\ln\frac{P_{jk}}{q_j}.$$

And since $F(0) = 1$, $E_0'(0) = -F'(0)/F(0) = \sum_{j,k} p_k P_{jk}\ln\frac{P_{jk}}{q_j} = I(p)$ — the average mutual information. So the random-coding exponent leaves the origin with slope exactly the mutual information of the input distribution I chose.

That is the moment everything turns. Near $\rho = 0$,

$$E_0(\rho,p) - \rho R \approx \rho\, I(p) - \rho R = \rho\,\big(I(p) - R\big),$$

which is *strictly positive* for any small $\rho > 0$ as long as $R < I(p)$. So pick $p$ to be the capacity-achieving distribution, $I(p) = C$; then for every $R < C$ there is some $\rho \in (0,1]$ making $E_0(\rho,p) - \rho R > 0$, hence $E(R) > 0$. The exponent is positive all the way up to capacity — not just to the cutoff rate. The union bound died at $R_0$ precisely because it was frozen at $\rho = 1$; letting $\rho$ slide down toward $0$ gives gentler supporting lines whose intercepts shrink to $0$ in exactly the way the mutual-information slope requires, and that keeps the exponent alive right up to $C$.

I should make sure this small-$\rho$ picture is the whole story and not a local accident, so let me pin down the shape of $E_0(\rho,p)$ in $\rho$. I claim it is increasing and concave in $\rho$. Concavity is what I need both to trust the optimization and to know the bound doesn't do something pathological. The cleanest route is the weighted-power-mean fact behind Hölder: for nonnegative $a_\ell$ and probabilities $q_\ell$, the quantity $\big(\sum_\ell q_\ell a_\ell^{1/x}\big)^x$ moves monotonically with $x$, and the logarithmic interpolation between two such exponents is controlled by Hölder. Applying that fact to the inner sums and then applying Hölder once more across the output letters gives exactly $\partial^2 E_0(\rho,p)/\partial\rho^2 \le 0$. Combined with $E_0(0,p) = 0$ and $E_0'(0) = I(p) > 0$, this gives $E_0(\rho,p) > 0$ and $\partial E_0/\partial\rho > 0$ for $\rho > 0$, with the slope $\partial E_0/\partial\rho$ nonincreasing from $I(p)$ down. Good — the curve rises and bends over, exactly the shape that makes the rate–exponent trade well behaved.

Now the optimization over $\rho$, for a fixed $p$. I'm maximizing $-\rho R + E_0(\rho,p)$ over $\rho \in [0,1]$. Setting the derivative to zero,

$$R = \frac{\partial E_0(\rho,p)}{\partial\rho}.$$

Because $\partial E_0/\partial\rho$ is nonincreasing in $\rho$, running from $I(p)$ at $\rho=0$ down to its value at $\rho = 1$, this stationarity equation has an interior solution $\rho^\star \in (0,1)$ exactly when $R$ lies in the band $\frac{\partial E_0}{\partial\rho}\big|_{\rho=1} \le R \le I(p)$. In that band I get the curve parametrically: $R = \partial E_0/\partial\rho$ and $E(R,p) = E_0(\rho,p) - \rho\,\partial E_0/\partial\rho$. And taking the ratio of the derivatives of these two with respect to $\rho$ gives $\partial E(R,p)/\partial R = -\rho$ — so the parameter $\rho$ is nothing but the *negative slope* of the exponent–rate curve. That's a clean reading: high rate $\leftrightarrow$ small $\rho$ $\leftrightarrow$ gentle slope; low rate $\leftrightarrow$ $\rho$ pushing toward $1$ $\leftrightarrow$ slope toward $-1$.

What happens below the band, when $R < \partial E_0/\partial\rho\big|_{\rho=1}$? Then $-\rho R + E_0(\rho,p)$ is still increasing at $\rho = 1$, so the maximum over $[0,1]$ sits at the boundary $\rho = 1$, and

$$E(R,p) = E_0(1,p) - R,$$

a straight line of slope $-1$. So the whole curve, for fixed $p$, is: a curved piece for $R_{\text{crit}} < R < I(p)$ where $\rho$ slides in $(0,1)$, reaching $\rho=1$ at the *critical rate* $R_{\text{crit}} = \partial E_0/\partial\rho\big|_{\rho=1}$ and $\rho=0$ at $R=I(p)$; below $R_{\text{crit}}$ it continues as a straight slope-$-1$ line. There is a tidy geometric way to see the same thing: for each fixed $\rho$, $-\rho R + E_0(\rho,p)$ is a line in $R$ with slope $-\rho$ and intercept $E_0(\rho,p)$ on the $E$-axis; the maximum over $\rho$ is the *upper envelope* of this family of lines, and $E_0(\rho,p)$ is exactly the $E$-axis intercept of the tangent to the $E(R)$ curve at slope $-\rho$. The envelope of lines is automatically convex, so $E(R,p)$ is convex downward in $R$, which I already knew from $\partial E/\partial R = -\rho$ increasing with $R$.

Finally maximize over the input distribution too: $E(R) = \max_p E(R,p)$, the upper envelope of all these per-$p$ curves. For the optimization over $p$ at fixed $\rho$ I want to minimize $F(p,p) = \sum_j\big(\sum_k p_k P_{jk}^{1/(1+\rho)}\big)^{1+\rho}$, and the relevant fact is that $F$ is convex in $p$ — which again falls out of Hölder applied to the inner sums — so a stationary point is the global minimum, and Kuhn–Tucker on $\sum_k p_k = 1$, $p_k \ge 0$ gives the condition $\sum_j P_{jk}^{1/(1+\rho)}\alpha_j^{\rho} \ge \sum_j \alpha_j^{1+\rho}$ for all $k$ with equality where $p_k \ne 0$, writing $\alpha_j = \sum_k p_k P_{jk}^{1/(1+\rho)}$. Putting the pieces together: $E(R)$ is positive, continuous, and convex downward on $0 < R < C$, with slope between $0$ and $-1$ — so $P_e \le e^{-NE(R)}$ is genuinely exponentially decreasing for every rate below capacity. That is the reliability curve I wanted.

Let me sanity-check on the binary symmetric channel, crossover $q$. By symmetry $p_1 = p_2 = \tfrac12$ maximizes everything, and the inner sum $\sum_k \tfrac12 P_{jk}^{1/(1+\rho)}$ is the same for both outputs: $\tfrac12\big(q^{1/(1+\rho)} + (1-q)^{1/(1+\rho)}\big)$. Then $E_0(\rho,p) = \rho\ln 2 - (1+\rho)\ln\big(q^{1/(1+\rho)} + (1-q)^{1/(1+\rho)}\big)$. Differentiate and run the parametric equations and the high-rate piece comes out as $E(R,\rho) = q_\rho\ln\frac{q_\rho}{q} + (1-q_\rho)\ln\frac{1-q_\rho}{1-q}$ with $q_\rho = q^{1/(1+\rho)}/\big(q^{1/(1+\rho)} + (1-q)^{1/(1+\rho)}\big)$ and $R = \ln 2 - H(q_\rho)$ — exactly the random-coding exponent Elias found for the BSC. And the low-rate line $\rho = 1$ gives $E(R) = \ln 2 - 2\ln(\sqrt q + \sqrt{1-q}) - R$, whose intercept is the cutoff rate. So the general machine reproduces the known BSC curve as a special case — reassuring.

Now I have to be honest about the low-rate end, because the union bound I started from secretly cheated there. At small $R$ the exponent is large, and an effect I had been neglecting — that the random ensemble occasionally assigns essentially the *same* word to two messages — stops being negligible. The bound $\overline{P_{e,m}}$ averages over all codes including these self-colliding ones, and at low rate those rare bad codes dominate the average and drag the exponent down. The remedy is surgical: don't average over the bad codes — *throw them out*. Go back to the fixed-code bound at $\rho = 1$ (the $s = 1/2$ shape), regard $P_{e,m}$ as a random variable over the ensemble, and bound $\Pr(P_{e,m} \ge B)$ by tilting: $\phi(\text{code}) \le \sum_{m'\ne m} \big(q(x_m,x_{m'})/B\big)^s$ where $q(x_m,x_{m'}) = \sum_y \sqrt{P(y\mid x_m)P(y\mid x_{m'})}$ is the pairwise "discrepancy" (the generalization of Hamming distance), valid for $0 < s \le 1$. Averaging the right side over independently chosen words and choosing $B$ so that $\Pr(P_{e,m} \ge B) \le \tfrac12$ means at most half the words in a typical code are bad; expurgate them. Removing words only lowers everyone else's error, and halving $M$ costs $(\ln 2)/N$ in rate — vanishing. Writing $\rho = 1/s \ge 1$, the surviving words obey $P_{e,m} \le \exp[-N(E_x(\rho,p) - \rho R)]$ with

$$E_x(\rho,p) = -\rho\,\ln\sum_{k,i} p_k\, p_i\left[\sum_j \sqrt{P_{jk}P_{ji}}\right]^{1/\rho}, \qquad \rho \ge 1.$$

So expurgation is exactly the continuation of the pairwise-distance family *past* $\rho = 1$ — the regime where the Jensen step would have flipped, which is precisely why I had to cap $\rho$ at $1$ in the averaging argument and switch to throwing words away here. At $\rho = 1$ the intercepts meet, $E_x(1,p) = E_0(1,p)$, so the $\rho=1$ expurgated line is the same straight line $E_0(1,p)-R$ that random coding uses at low rates. The optimized expurgated envelope agrees with that line until its own slope condition pushes the optimizer to $\rho>1$; once that happens, it lies above the random-coding line and tightens the genuinely low-rate exponent.

And the upper edge of the curve closes off beautifully against the converse. Fano's lower bound on error has the form $P_e \ge e^{-N[E_L(R) + o(1)]}$ with

$$E_L(R) = \sup_{0 < \rho < \infty}\,\max_p\,[E_0(\rho,p) - \rho R],$$

the *same* $E_0$, the *same* $E_0(\rho,p) - \rho R$, the only difference being that the lower-bound envelope lets $\rho$ run over all of $(0,\infty)$ while my achievability capped it at $[0,1]$. Reading both as upper envelopes of lines of slope $-\rho$: wherever the optimal $\rho$ for the converse already lies in $[0,1]$, the two envelopes coincide. That happens for $R_{\text{crit}} < R < C$. So in that whole high-rate band my upper bound and Fano's lower bound *agree*, and $E(R)$ is not merely a bound but the exact reliability function of the channel. Only below $R_{\text{crit}}$ do they part, and there the expurgated bound is what pushes the achievable side up.

One more thing the same argument hands me almost for free: input constraints and continuous channels. Suppose each codeword must satisfy a constraint $\sum_n f(x_n) \le 0$ — for the Gaussian channel $f(x) = x^2 - A$, i.e. average power $\le A$. The difficulty is that a random word won't exactly satisfy it. The fix is to tilt the ensemble toward the constraint: include an indicator that the word's $\sum_n f(x_n) \in [-\delta, 0]$ and normalize, then overbound that indicator by $\exp\big[r(\sum_n f(x_n) + \delta)\big]$ for any $r \ge 0$ — true because when the indicator is $1$ the exponent's argument is $\ge 0$. Carrying this through the identical steps from the general bound replaces $E_0$ by $E_0(\rho,p,r) = -\ln \sum_j\big(\sum_k p_k P_{jk}^{1/(1+\rho)} e^{rf_k}\big)^{1+\rho}$ — the same expression with each input weighted by $e^{rf_k}$ — and a sub-exponential coefficient $B$ that the central limit theorem controls. For the additive Gaussian-noise channel, the stationary input is the Gaussian $p(x) = \mathcal N(0, A)$, and the integrals close to give a curved $E(R)$ matching Shannon's exact Gaussian exponent in the band where it is known. So the one tilted-union-plus-Jensen argument covers the discrete channel, its low-rate expurgated refinement, and the power-constrained Gaussian channel in a single sweep — which was the whole point of wanting a *simple* derivation.

Let me collect the landing result. Starting from the exact ML error $P_{e,m} = \sum_y P(y\mid x_m)\phi_m(y)$, I overbounded the indicator $\phi_m$ by the competitor-sum raised to a free power $\rho$ with the symmetric tilt $1/(1+\rho)$; averaging over an i.i.d. random codebook and using Jensen's inequality with $\rho \in [0,1]$ to pull the bar inside the bracket collapsed everything, for a memoryless channel, into the single-letter function $E_0(\rho,p) = -\ln\sum_j\big(\sum_k p_k P_{jk}^{1/(1+\rho)}\big)^{1+\rho}$ and the bound $P_e \le \exp[-N(E_0(\rho,p) - \rho R)]$. Optimizing the free parameter and the input distribution gives the random-coding exponent $E_r(R) = \max_{0\le\rho\le1,\,p}[E_0(\rho,p) - \rho R]$; and because $E_0(0,p) = 0$ with slope $E_0'(0) = I(p)$, the exponent stays strictly positive for every rate below capacity, which is exactly the quantitative reliability law I set out to find — with the expurgated bound ($\rho \ge 1$) sharpening it at low rates and Fano's converse matching it from $R_{\text{crit}}$ up to $C$.

```python
import numpy as np

# Random-coding error exponent E_r(R) for a discrete memoryless channel.
# P[j, k] = Pr(output b_j | input a_k);  p[k] = input letter probabilities.

def E0(rho, p, P):
    # the single-letter tilted log-sum the averaging+Jensen step produced:
    # E0 = -ln sum_j ( sum_k p_k P_jk^{1/(1+rho)} )^{1+rho}
    inner = (p[None, :] * P ** (1.0 / (1.0 + rho))).sum(axis=1)   # over k, per output j
    return -np.log((inner ** (1.0 + rho)).sum())

def E_r(R, P, p_grid, rho_grid):
    # E_r(R) = max over rho in [0,1] and input dist p of E0(rho, p) - rho*R
    best = 0.0
    for p in p_grid:
        for rho in rho_grid:            # rho_grid subset of [0, 1]
            best = max(best, E0(rho, p, P) - rho * R)
    return best

def E_x(rho, p, P):
    # Expurgated single-letter function, rho >= 1.
    pair = np.sqrt(P[:, :, None] * P[:, None, :]).sum(axis=0)   # pair[k, i]
    return -rho * np.log((p[:, None] * p[None, :] * pair ** (1.0 / rho)).sum())

def E_ex(R, P, p_grid, rho_grid):
    # Expurgated exponent: maximize E_x(rho, p) - rho*R over rho >= 1.
    return max(E_x(rho, p, P) - rho * R
               for p in p_grid for rho in rho_grid)

def E_L(R, P, p_grid, rho_grid):
    # Fano/sphere-packing exponent uses the same E0 with rho over (0, infinity).
    return max(E0(rho, p, P) - rho * R
               for p in p_grid for rho in rho_grid)

def error_probability_bound(N, R, P, p_grid, rho_grid):
    # there exists a length-N rate-R code with ML error <= exp(-N E_r(R))
    return np.exp(-N * E_r(R, P, p_grid, rho_grid))

# Mutual information = E0'(0), the slope at the origin that keeps E_r(R) > 0 for R < C.
def mutual_information(p, P):
    q = P @ p
    return sum(p[k] * P[j, k] * np.log(P[j, k] / q[j])
               for k in range(len(p)) for j in range(P.shape[0])
               if p[k] > 0 and P[j, k] > 0)
```
