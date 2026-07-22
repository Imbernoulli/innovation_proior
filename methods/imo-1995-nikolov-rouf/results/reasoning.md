I have the set $\{1,2,\dots,2p\}$, $2p$ numbers, $p$ an odd prime. I'm choosing exactly $p$ of them — half — and I care only about whether their sum is a multiple of $p$. So among all $\binom{2p}{p}$ ways to pick half the numbers, how many land their sum in the single residue class $0$ mod $p$? Two things are tangled together: a *size* constraint (pick exactly $p$) and a *congruence* on an additive quantity (the sum is $\equiv 0$). That combination is what I have to deal with.

My instinct on any "how many objects with property X mod $p$" question is that the property should be roughly equidistributed — if subset sums were spread evenly across the $p$ residue classes, the answer would be about $\binom{2p}{p}/p$. But $\binom{2p}{p}$ isn't a multiple of $p$ in general, so it can't be exactly that, and I don't yet know the size of the deviation. Let me get my hands dirty with the smallest case to calibrate, $p=3$, set $\{1,2,3,4,5,6\}$, choosing $3$ of them. There are $\binom{6}{3}=20$ triples. Which have sum divisible by $3$? Sum ranges from $1+2+3=6$ to $4+5+6=15$. Let me just hunt: $\{1,2,3\}=6$, $\{1,2,6\}=9$, $\{1,3,5\}=9$, $\{2,3,4\}=9$, $\{1,5,6\}=12$, $\{2,4,6\}=12$, $\{3,4,5\}=12$, $\{4,5,6\}=15$. That's eight. So for $p=3$ the answer is $8$.

Now $\binom{6}{3}=20$, and $20/3=6.67$, while the true count is $8$. The gap is $8-6.67=1.33$, i.e. $8 = 20/3 + 4/3 = (20+4)/3$. Hmm, $(20+4)/3$. Or write it as $8 = 6 + 2 = (20-2)/3 + 2$. That second form is suggestive: $(20-2)/3$ is $18/3=6$, an integer, plus a clean integer correction. Two subsets stick out as obviously special: $\{1,2,3\}$ (the bottom half) and $\{4,5,6\}$ (the top half). Their sums are $6$ and $15$ -- both divisible by $3$ -- and they respect the natural split of $\{1,\dots,6\}$ into a low block and a high block. So my working guess is that the answer may have the form $\frac{\binom{2p}{p}-2}{p}+2$. I still need to derive it rather than let one example and two visible subsets fool me.

The honest difficulty is the congruence on the sum. I can't case-split on "what is the sum mod $p$" because subset sums scatter all over the place and there's no clean recursion staring at me. What I want is a machine that takes a family of objects, each carrying an integer weight, and counts only the ones whose weight sits in a prescribed residue class. The natural machine for "objects carrying an additive weight" is a generating function: encode each weight as an exponent, and the bookkeeping of which subset has which sum becomes polynomial multiplication.

Let me build it carefully, because I need to track *two* things at once — the size of the subset and its sum — so one variable won't do. Give each element $k\in\{1,\dots,2p\}$ a factor
$$1 + x\,y^{k}.$$
The idea: when I expand the product over all $k$, each factor offers a binary choice. Take the "$1$" and I've left element $k$ out; take the "$x\,y^{k}$" and I've put it in, paying one unit of $x$ (to count *one* more element) and $y^{k}$ (to record its contribution $k$ to the sum). So
$$F(x,y)=\prod_{k=1}^{2p}\bigl(1+x\,y^{k}\bigr),$$
and when I multiply everything out, the monomial $x^{m}y^{s}$ appears with coefficient equal to the number of $m$-element subsets whose elements sum to $s$. The size lives in the $x$-exponent, the sum lives in the $y$-exponent. Exactly the two statistics I need, separated.

So the quantity I'm after is: sum the coefficient of $x^{p}y^{s}$ over all $s$ divisible by $p$. The $x^p$ part pins the size to $p$; that part I'm not worried about — I just look at the coefficient of $x^p$. The trouble is the other constraint. The coefficient of $x^p$ in $F$ is itself a polynomial in $y$ — call it $G(y)=\sum_s c_s\,y^s$, where $c_s$ counts $p$-subsets with sum $s$ — and I want $\sum_{p\mid s}c_s$, the sum of the coefficients whose exponent is a multiple of $p$. But even after fixing the size, $G(y)$ can run up to the sum of the largest $p$ elements,
$$
(p+1)+(p+2)+\cdots+2p=\frac{p(3p+1)}2,
$$
so it is still far too large to expand by hand, and the exponents I want — $0,p,2p,3p,\dots$ — are sprinkled through it. I need to *project* $G$ onto its "exponent $\equiv 0 \pmod p$" part without ever writing $G$ out. That's the real obstacle. Plain generating functions hand me the whole distribution of sums; I want one slice of it.

Let me think about what kind of operation reads off only the multiples-of-$p$ exponents of a polynomial. I want a linear gadget $L$ acting on $G(y)=\sum_s c_s y^s$ that returns $\sum_{p\mid s}c_s$ — i.e. a gadget that multiplies $c_s$ by $1$ when $p\mid s$ and by $0$ otherwise. So I need numbers I can plug in for $y$ such that averaging $y^{s}$ over them gives $1$ when $p\mid s$ and $0$ when $p\nmid s$. A single real substitution can't do that; one number $y$ gives $y^s$, which can't be both an indicator of divisibility and stay bounded as $s$ grows. I need several substitutions averaged together, and the average of $y^s$ over them has to detect $s$ mod $p$.

The objects whose powers cycle with period $p$ and average to an indicator of divisibility are the $p$-th roots of unity. Let $\omega=e^{2\pi i/p}$, so $1,\omega,\omega^2,\dots,\omega^{p-1}$ are the $p$ solutions of $z^p=1$. The fact I want is the orthogonality of these roots: for any integer $s$,
$$\sum_{j=0}^{p-1}\omega^{js}=\begin{cases}p,& p\mid s,\\[2pt]0,& p\nmid s.\end{cases}$$
If $p\mid s$ then $\omega^{s}=1$ and each term is $1$, so the sum is $p$. If $p\nmid s$ then $\omega^{s}\ne1$, and $\sum_{j=0}^{p-1}(\omega^{s})^{j}$ is a finite geometric series equal to $\frac{(\omega^{s})^{p}-1}{\omega^{s}-1}=\frac{(\omega^{p})^{s}-1}{\omega^{s}-1}=\frac{1-1}{\omega^{s}-1}=0$. Clean. So averaging $\omega^{js}$ over $j$ is *exactly* the indicator of $p\mid s$, scaled by $p$.

Now feed this into $G$. Substitute $y=\omega^{j}$ and sum over $j$:
$$\sum_{j=0}^{p-1}G(\omega^{j})=\sum_{j=0}^{p-1}\sum_{s}c_s\,\omega^{js}=\sum_{s}c_s\sum_{j=0}^{p-1}\omega^{js}=\sum_{s}c_s\cdot p\,[\,p\mid s\,]=p\sum_{p\mid s}c_s.$$
So the count I want is
$$N=\sum_{p\mid s}c_s=\frac1p\sum_{j=0}^{p-1}G(\omega^{j}).$$
And $G(y)$ was just the coefficient of $x^p$ in $F(x,y)$, so $G(\omega^j)=[x^p]\,F(x,\omega^j)$, giving
$$N=\frac1p\sum_{j=0}^{p-1}[x^{p}]\,F(x,\omega^{j})=\frac1p\sum_{j=0}^{p-1}[x^{p}]\prod_{k=1}^{2p}\bigl(1+x\,\omega^{jk}\bigr).$$
The huge polynomial in $y$ never has to be expanded — I just evaluate $F$ at $p$ specific values of $y$ on the unit circle and average. The projection I needed is exactly this average over roots of unity. Now I have to evaluate the $p$ pieces.

Start with the easy one, $j=0$. Then $\omega^{0}=1$, so every $y^{k}$ becomes $1^{k}=1$ and the product collapses:
$$F(x,1)=\prod_{k=1}^{2p}(1+x)=(1+x)^{2p}.$$
Its coefficient of $x^p$ is, by the binomial theorem, $\binom{2p}{p}$. So the $j=0$ term contributes $\binom{2p}{p}$ — this is the "count everything ignoring the congruence" term, which makes sense: at $j=0$ the filter doesn't discriminate by sum at all, it just counts all $p$-subsets.

Now the $p-1$ terms with $1\le j\le p-1$, which is where the actual work is. Fix such a $j$. Since $p$ is prime and $1\le j\le p-1$, $\omega^{j}$ is itself a *primitive* $p$-th root of unity — its powers $\omega^{0j},\omega^{1j},\dots,\omega^{(p-1)j}$ run through all $p$ roots of unity, just reshuffled, because $jk\bmod p$ is a bijection on residues when $\gcd(j,p)=1$. I have to evaluate
$$\prod_{k=1}^{2p}\bigl(1+x\,\omega^{jk}\bigr).$$
The exponents are $jk$ for $k=1,2,\dots,2p$. Reduce $k$ mod $p$: as $k$ runs $1,2,\dots,2p$, it covers every residue class mod $p$ *exactly twice* — because $2p$ is two full blocks of $p$ consecutive integers, and within each block $k$ hits each residue once. This is the spot where the specific number $2p$ in the problem bites: the ground set being *twice* as long as the modulus means each $p$-th root appears in the product with multiplicity exactly $2$. So the product splits as a square,
$$\prod_{k=1}^{2p}\bigl(1+x\,\omega^{jk}\bigr)=\left[\prod_{k=1}^{p}\bigl(1+x\,\omega^{jk}\bigr)\right]^{2},$$
and one block $k=1,\dots,p$ already sweeps $\omega^{jk}$ through *all* the $p$-th roots of unity once each (the value at $k=p$ is $\omega^{jp}=1$, the "$\omega^0$" root). So
$$\prod_{k=1}^{p}\bigl(1+x\,\omega^{jk}\bigr)=\prod_{r=0}^{p-1}\bigl(1+x\,\omega^{r}\bigr),$$
the product over *every* $p$-th root of unity, regardless of which primitive $\omega^j$ I started from. That already tells me every nontrivial $j$ will give the *same* value — the dependence on $j$ has evaporated, because multiplying the exponents by $j$ just permutes the roots and the product is over all of them.

So I need the single quantity $\prod_{r=0}^{p-1}(1+x\,\omega^{r})$. The $p$-th roots of unity are precisely the roots of $z^p-1$, so
$$z^{p}-1=\prod_{r=0}^{p-1}(z-\omega^{r}).$$
I want $\prod_r(1+x\,\omega^{r})$, which has $\omega^r$ multiplied by $x$ and a $+1$, not a $z-\omega^r$ shape. Let me massage it. Factor an $x$ out of each term: $1+x\,\omega^{r}=x\bigl(\tfrac1x+\omega^{r}\bigr)=-x\bigl(-\tfrac1x-\omega^{r}\bigr)=-x\bigl((-\tfrac1x)-\omega^{r}\bigr)$. So
$$\prod_{r=0}^{p-1}\bigl(1+x\,\omega^{r}\bigr)=\prod_{r=0}^{p-1}(-x)\bigl((-\tfrac1x)-\omega^{r}\bigr)=(-x)^{p}\prod_{r=0}^{p-1}\bigl((-\tfrac1x)-\omega^{r}\bigr).$$
The remaining product is exactly $z^p-1$ evaluated at $z=-\tfrac1x$:
$$\prod_{r=0}^{p-1}\bigl((-\tfrac1x)-\omega^{r}\bigr)=\Bigl(-\tfrac1x\Bigr)^{p}-1.$$
So
$$\prod_{r=0}^{p-1}\bigl(1+x\,\omega^{r}\bigr)=(-x)^{p}\left[\Bigl(-\tfrac1x\Bigr)^{p}-1\right]=(-x)^{p}\Bigl(-\tfrac1x\Bigr)^{p}-(-x)^{p}.$$
Now $(-x)^{p}\bigl(-\tfrac1x\bigr)^{p}=\bigl((-x)(-\tfrac1x)\bigr)^{p}=1^{p}=1$, and $(-x)^{p}=-x^{p}$ because $p$ is odd — and there's the second place oddness of $p$ matters: it flips the sign of $(-x)^p$ to $-x^p$, so that the $-(-x)^p$ becomes $+x^p$. Putting it together,
$$\prod_{r=0}^{p-1}\bigl(1+x\,\omega^{r}\bigr)=1-(-x)^{p}=1+x^{p}.$$
A startlingly clean collapse: the whole product over all $p$-th roots is just $1+x^p$. (Check on $p=3$: $\prod_{r=0}^{2}(1+x\omega^r)=(1+x)(1+x\omega)(1+x\omega^2)$. The constant term is $1$; the $x^3$ term is $x^3\,\omega^{0+1+2}=x^3\omega^3=x^3$; the $x^1$ and $x^2$ coefficients are $1+\omega+\omega^2=0$ and $\omega^0\omega^1+\omega^0\omega^2+\omega^1\omega^2=\omega+\omega^2+\omega^3=\omega+\omega^2+1=0$. So it's $1+x^3$.)

Therefore, for each nontrivial $j$,
$$F(x,\omega^{j})=\left[\prod_{r=0}^{p-1}(1+x\,\omega^{r})\right]^{2}=(1+x^{p})^{2}=1+2x^{p}+x^{2p},$$
and the coefficient of $x^{p}$ is exactly $2$. Every single one of the $p-1$ nontrivial roots contributes $2$, and now I see *why* they all contribute equally — it's not a coincidence to be checked case by case, it's because each primitive root, raised through $k=1,\dots,2p$, sweeps the full set of $p$-th roots twice, and the product over the full set is the same $1+x^p$ no matter which primitive root you entered with.

Now I assemble the average:
$$N=\frac1p\left[\,\underbrace{\binom{2p}{p}}_{j=0}+\underbrace{(p-1)\cdot 2}_{j=1,\dots,p-1}\,\right]=\frac1p\left[\binom{2p}{p}+2p-2\right].$$
Let me put it in the form my $p=3$ experiment suggested. Split off the $2p$:
$$N=\frac{\binom{2p}{p}+2p-2}{p}=\frac{\binom{2p}{p}-2}{p}+\frac{2p}{p}=\frac{\binom{2p}{p}-2}{p}+2.$$
There is the same integer form that the $p=3$ case hinted at, now coming from the root average rather than guesswork. And the check at $p=3$: $\binom{6}{3}=20$, so $N=\frac{20-2}{3}+2=\frac{18}{3}+2=6+2=8$. Matches the eight triples I found by hand.

One thing nags: I've written $N=\frac{\binom{2p}{p}-2}{p}+2$, and $N$ is a count, so it had better be an integer — which forces $p\mid\bigl(\binom{2p}{p}-2\bigr)$. Is that actually true, or have I made an error that happens to work at $p=3$? Let me see it directly. Rewrite the middle binomial coefficient as
$$
\binom{2p}{p}=\frac{(p+1)(p+2)\cdots(2p)}{1\cdot2\cdots p}
=2\prod_{i=1}^{p-1}\frac{p+i}{i}.
$$
For each $1\le i\le p-1$, the denominator $i$ is invertible mod $p$, and $p+i\equiv i\pmod p$, so each factor $\frac{p+i}{i}$ is congruent to $1$ mod $p$. Therefore $\binom{2p}{p}\equiv2\pmod p$, hence $p\mid\bigl(\binom{2p}{p}-2\bigr)$, and $N$ is an integer: the $\frac{\binom{2p}{p}-2}{p}$ piece is a genuine whole number, the $+2$ a genuine integer correction.

I have to be careful about the visible $2$ in the nontrivial-root computation. The two natural blocks $\{1,2,\dots,p\}$ and $\{p+1,p+2,\dots,2p\}$ do have sums divisible by $p$: the first has sum $1+2+\cdots+p=\frac{p(p+1)}2=p\cdot\frac{p+1}2$, and the second has sum $\frac{p(p+1)}2+p^{2}$. Algebraically, after the ground set is split into two full residue cycles, the coefficient of $x^p$ in $(1+x^p)^2$ comes from $x^p\cdot1$ or $1\cdot x^p$, matching those two full-block choices. But the final count is still the root average; it is not a partition into a uniform part plus those two subsets. The precise statement is that every nontrivial root contributes the coefficient $[x^p](1+x^p)^2=2$, and after the average this is what rewrites as $\frac{\binom{2p}{p}-2}{p}+2$.

So the count is
$$N=\frac{\dbinom{2p}{p}-2}{p}+2,$$
equivalently $N=\dfrac1p\Bigl[\dbinom{2p}{p}+2(p-1)\Bigr]$.
