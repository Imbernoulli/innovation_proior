I keep running into the same wall, and it is a wall between two kinds of statement. On one side I can say things like "the probability that the $n$-th digit of a random number in $[0,1]$ deviates from its expected frequency is small," and I can even say that this probability tends to $0$ as $n$ grows. On the other side is the statement I actually want: "for almost every number, the digit frequencies eventually settle down and stay settled — the deviations stop happening for good." That second statement is not about any single stage $n$. It is about a fixed outcome $\omega$ and the *entire infinite tail* of what happens to it. The weak law and convergence in probability live entirely on the first side. They tell me, stage by stage, that a bad thing is improbable; they are silent about whether, along one fixed random path, the bad thing recurs forever or eventually quits. I need a bridge.

So let me set it up cleanly. I have events $A_1,A_2,A_3,\dots$, with $A_n$ = "the bad thing happens at stage $n$." For a fixed outcome $\omega$, the honest question is: how many of the $A_n$ contain $\omega$? If only finitely many do, then past some point the bad thing never happens again — that is "eventually good, forever." If infinitely many do, the bad thing keeps coming back. So the object I care about is the set of $\omega$ that lie in infinitely many $A_n$. Let me name it and pin it down set-theoretically, because everything will hinge on getting this object exactly right.

"$\omega$ is in infinitely many $A_n$" means: for every $n$, there is some $k\ge n$ with $\omega\in A_k$. "For every $n$ there exists $k\ge n$" — that is an intersection over $n$ of a union over $k\ge n$:
$$\{\omega \text{ in infinitely many } A_n\} = \bigcap_{n=1}^{\infty}\bigcup_{k=n}^{\infty} A_k.$$
Stare at the inner union $\bigcup_{k\ge n}A_k$ — call it $B_n$. As $n$ increases I am throwing away terms from the front, so $B_1\supseteq B_2\supseteq B_3\supseteq\cdots$; the $B_n$ are *decreasing*. And $\bigcap_n B_n$ is their limit. This is exactly the construction people write as $\limsup_n A_n$, and the reason for that name clicks: $1_{\bigcap_n B_n} = \lim_n \sup_{k\ge n} 1_{A_k} = \limsup_n 1_{A_n}$ — the limsup of the indicator sequence is the indicator of this set. Good. So my target is $P(\limsup_n A_n)$, and I will write "$A_n$ i.o." (infinitely often) for the event $\limsup_n A_n$. The complementary statement — bad thing happens only finitely often — is $\liminf_n A_n^c = \bigcup_n \bigcap_{k\ge n} A_k^c$, "eventually $A_k^c$ forever."

Now what do I actually have to work with? Only the marginal numbers $P(A_n)$, and the basic axioms: monotonicity, the union bound $P(\bigcup_k C_k)\le \sum_k P(C_k)$, and continuity of $P$ along monotone sequences of sets (which is just countable additivity restated). Let me try to get *some* grip on $P(\limsup A_n)$ from these alone.

The decreasing structure is handing me continuity from above for free. $B_n=\bigcup_{k\ge n}A_k$ decreases to $\bigcap_n B_n = \limsup A_n$, so
$$P(\limsup_n A_n) = P\!\left(\bigcap_n B_n\right) = \lim_{n\to\infty} P(B_n) = \lim_{n\to\infty} P\!\left(\bigcup_{k=n}^{\infty} A_k\right).$$
I have reduced the limsup to a limit of union-probabilities. And on each union I can throw the union bound:
$$P\!\left(\bigcup_{k=n}^{\infty}A_k\right) \le \sum_{k=n}^{\infty} P(A_k).$$
So $P(\limsup A_n)\le \lim_{n\to\infty}\sum_{k=n}^{\infty}P(A_k)$. Now look hard at that right-hand side. $\sum_{k\ge n}P(A_k)$ is the *tail* of the series $\sum_k P(A_k)$. If that series converges — if $\sum_k P(A_k)<\infty$ — then by the very definition of convergence its tail goes to $0$ as $n\to\infty$. So the whole right-hand side is $0$, which forces
$$P(\limsup_n A_n) = 0.$$
That is a complete argument, and it needed *nothing* about how the $A_n$ relate to each other — no independence, no structure at all. Just subadditivity, which is one-directional and free, plus the elementary fact that a convergent series has a vanishing tail. So: **if $\sum_n P(A_n)<\infty$, then almost surely only finitely many $A_n$ occur.** The convergence of the series of probabilities is exactly the brake on the bad event.

Let me re-derive that a second way, because the second way tells me *why* it is true rather than just *that* it is, and I suspect the "why" will matter later. Instead of chasing the limsup set, count occurrences directly. Let $N(\omega)=\sum_{k=1}^{\infty}1_{A_k}(\omega)$ be the total number of events that happen along $\omega$ — a nonnegative integer or $+\infty$. Its expectation, by countable additivity of the integral over nonnegative terms (Tonelli — I can swap $E$ and $\sum$ freely for nonnegative summands), is
$$E\,N = E\sum_{k}1_{A_k} = \sum_k E\,1_{A_k} = \sum_k P(A_k).$$
If $\sum_k P(A_k)<\infty$ then $E\,N<\infty$, and a nonnegative random variable with finite expectation must be finite almost surely (if $N=\infty$ on a set of positive probability, its expectation would be $\infty$). So $N<\infty$ a.s. — finitely many events occur a.s. — which is the same conclusion. This version says it plainly: the *expected number* of occurrences is the sum of the probabilities, and a finite average count cannot tolerate an infinite count on a non-null set. I like this proof; it is the honest reason the first half works.

Now the hard direction, and the one I genuinely don't know the shape of yet. The question writes itself: is the converse true? If $\sum_n P(A_n)=\infty$, must infinitely many $A_n$ occur? It would be beautiful — a perfect dichotomy keyed on convergence vs. divergence of a single series. Let me first see whether the dichotomy can possibly be unconditional, because I have a nagging memory of a sequence that should kill it.

Take $\Omega=(0,1)$ with Lebesgue measure, and the *nested* events $A_n=(0,1/n)$. Then $P(A_n)=1/n$, and $\sum_n 1/n=\infty$ — the series diverges as hard as the harmonic series. So if the converse were unconditional, infinitely many $A_n$ would occur a.s. But what is $\limsup_n A_n$ here? An $\omega$ is in infinitely many $(0,1/n)$ only if $\omega< 1/n$ for infinitely many $n$, i.e. $\omega\le 0$ — impossible in $(0,1)$. So $\limsup_n A_n=\varnothing$ and $P(\limsup A_n)=0$. Divergent sum, yet the bad event happens for *no* $\omega$ eventually. The converse is false in general.

So I hit the wall, and the counterexample tells me exactly what the wall is made of. These nested events are piling all their probability mass onto the *same* outcomes — the small ones near $0$ — over and over. Event $A_{n+1}$ doesn't reach any new $\omega$; it just re-uses a subset of the $\omega$'s that $A_n$ already covered. The divergence of $\sum P(A_n)$ is being "spent" redundantly on the same region. For divergence to *force* recurrence, I need the events to keep spreading their mass around, to keep reaching fresh outcomes, so that the accumulated probability cannot all hide inside one shrinking set. Some hypothesis that prevents the events from conspiring to overlap. The natural candidate — the structural assumption that says events don't coordinate — is independence. Let me try assuming the $A_n$ are independent and see if divergence now bites.

I want to show $P(\limsup A_n)=1$, i.e. that the complement is null. The complement of "$A_n$ i.o." is "$A_n$ only finitely often" $=\bigcup_n \bigcap_{k\ge n}A_k^c$. A countable union is null iff each piece is null, so it suffices to show $P\!\left(\bigcap_{k=n}^{\infty}A_k^c\right)=0$ for every fixed $n$. Equivalently — and this is the cleaner thing to chase — I'll show $P(B_n)=P(\bigcup_{k\ge n}A_k)=1$ for every $n$; then $\limsup A_n=\bigcap_n B_n$ is an intersection of probability-one events, hence probability one. Either framing reduces the limsup to the tail intersection $\bigcap_{k\ge n}A_k^c$.

Here is where independence finally has something to do. The $A_k$ are independent, so their complements $A_k^c$ are independent too (a standard fact — independence is preserved under complementation within the family). And independence is precisely the rule that turns the probability of an intersection into a *product*. But the full intersection is infinite, and I don't want to invoke any "product over infinitely many factors" as an axiom — let me work with a finite intersection and then pass to the limit by continuity. For $n\le N$,
$$P\!\left(\bigcap_{k=n}^{N}A_k^c\right) = \prod_{k=n}^{N}P(A_k^c) = \prod_{k=n}^{N}\bigl(1-P(A_k)\bigr).$$
And $\bigcap_{k=n}^{N}A_k^c$ decreases (in $N$) to $\bigcap_{k=n}^{\infty}A_k^c$, so by continuity from above,
$$P\!\left(\bigcap_{k=n}^{\infty}A_k^c\right) = \lim_{N\to\infty}\prod_{k=n}^{N}\bigl(1-P(A_k)\bigr).$$
So everything now rides on a single question: does this product of factors $(1-P(A_k))$, each just below $1$, collapse to $0$ when $\sum_k P(A_k)=\infty$?

And this is the part where it would be easy to lose the answer. Each factor $1-P(A_k)$ is in $[0,1)$; multiplying many of them shrinks the product, but how do I know it shrinks *all the way to $0$* rather than leveling off at some positive limit? An infinite product $\prod(1-p_k)$ can be positive — for instance if $\sum p_k<\infty$ it converges to a positive number. So the divergence of $\sum p_k$ has to be the thing that drives the product to $0$, and I need a clean way to *see* the sum inside the product. The product is awkward; the sum is what I have a hypothesis about. I need to convert one into the other.

The bridge I want is a pointwise inequality that bounds each factor $1-p$ by something whose product is itself an exponential of a sum. The exponential is the unique function that turns products into sums of exponents: $\prod_k e^{-p_k}=e^{-\sum_k p_k}$. So if I can dominate $1-p$ by $e^{-p}$, the product of the $(1-p_k)$ is dominated by $e^{-\sum p_k}$, and a divergent sum sends that to $0$. Is $1-p\le e^{-p}$ true? Let $g(x)=e^{-x}-(1-x)$. Then $g(0)=0$, $g'(x)=-e^{-x}+1=1-e^{-x}$, which is $\ge 0$ for $x\ge 0$ and $\le 0$ for $x\le 0$, so $g$ has a global minimum at $x=0$ where it equals $0$; hence $g(x)\ge 0$ for all real $x$, i.e.
$$1-x\le e^{-x}\quad\text{for every } x\in\mathbb R,$$
with equality only at $x=0$. (Same thing read off the Taylor series: $e^{-x}=1-x+\tfrac{x^2}{2}-\cdots$, and for the inequality the convexity argument above is airtight without worrying about the sign of the remainder.) It is tight near $0$, which is exactly the regime that matters — when the $p_k$ are small, almost no slack is lost. This is the right tool: one elementary inequality, no theory of infinite products needed.

Apply it factor by factor with $x=P(A_k)$:
$$\prod_{k=n}^{N}\bigl(1-P(A_k)\bigr)\le \prod_{k=n}^{N}e^{-P(A_k)} = \exp\!\left(-\sum_{k=n}^{N}P(A_k)\right).$$
Now let $N\to\infty$. The tail $\sum_{k=n}^{N}P(A_k)\to\sum_{k=n}^{\infty}P(A_k)=\infty$, because the full series diverges and chopping off the finite front $\sum_{k<n}$ leaves a still-divergent tail. So the exponent $\to-\infty$ and the bound $\to e^{-\infty}=0$. Therefore
$$P\!\left(\bigcap_{k=n}^{\infty}A_k^c\right) = \lim_{N\to\infty}\prod_{k=n}^{N}\bigl(1-P(A_k)\bigr) \le \lim_{N\to\infty}\exp\!\left(-\sum_{k=n}^{N}P(A_k)\right) = 0.$$
That holds for every $n$. The countable union $\bigcup_n\bigcap_{k\ge n}A_k^c$ of these null sets is null, so its complement has probability $1$:
$$P(\limsup_n A_n)=1.$$
So under independence the divergence of $\sum P(A_n)$ *does* force infinitely many occurrences. And the inequality $1-x\le e^{-x}$ is doing the entire load-bearing work: it is the hinge that converts the product I can't evaluate into the exponential of the sum I can. The exact spot independence enters is the single line where the intersection-probability becomes a product; remove independence and that line collapses, and the nested counterexample shows the conclusion collapses with it.

Let me re-examine the counterexample through this lens to make sure I understand *why* it slips through. For $A_n=(0,1/n)$, the complements $A_n^c=[1/n,1)$ are very far from independent — they are nested *increasing*, totally positively dependent. $P(\bigcap_{k=n}^N A_k^c)=P(A_N^c)=1-1/N$, which does *not* factor as $\prod(1-1/k)$, and its limit is $1-0=1$, not $0$. So $\bigcap_{k\ge n}A_k^c$ has full probability, the bad event happens never-eventually, exactly as I computed. The product formula — the one good line — is the precise thing that fails without independence. That settles that independence is not a convenience but the working hypothesis of the converse.

I now have a clean dichotomy in two halves: convergence $\Rightarrow$ finitely often (always), and divergence $+$ independence $\Rightarrow$ infinitely often. Before I lock it in, let me push on both edges, because two things nag at me. First, mutual independence felt like *more* than I used — I only ever multiplied complement-probabilities; do I really need every finite sub-collection to factor, or would something weaker do? Second, the conclusion "infinitely often" is qualitative; the counting proof of the first half suggests I might be able to say *how many* times, not just "infinitely."

Take the weakening first. Suppose the $A_m$ are only *pairwise* independent — equivalently the indicators $X_m=1_{A_m}$ are uncorrelated. Set $S_n=\sum_{m=1}^n X_m$, the number of occurrences up to $n$, with mean $ES_n=\sum_{m=1}^n P(A_m)$, which $\to\infty$ since the series diverges. Uncorrelatedness gives the variance of the sum as the sum of the variances:
$$\operatorname{Var}(S_n)=\sum_{m=1}^n \operatorname{Var}(X_m).$$
Each $X_m\in\{0,1\}$, so $\operatorname{Var}(X_m)\le E X_m^2=E X_m=P(A_m)$, whence $\operatorname{Var}(S_n)\le \sum_{m=1}^n P(A_m)=ES_n$. Now Chebyshev on the deviation of $S_n$ from its mean, at relative scale $\delta$:
$$P\bigl(|S_n-ES_n|>\delta\,ES_n\bigr)\le \frac{\operatorname{Var}(S_n)}{(\delta ES_n)^2}\le \frac{ES_n}{\delta^2 (ES_n)^2}=\frac{1}{\delta^2\,ES_n}\to 0,$$
because $ES_n\to\infty$. So $S_n/ES_n\to 1$ *in probability*. In particular $S_n\to\infty$ in probability, which already says the count is unbounded — but "in probability" is the weak side again, and I want an almost-sure statement about a fixed path. To upgrade, the trick from the records-and-runs world: prove it on a fast-enough subsequence where the deviation probabilities are *summable*, then sandwich the gaps by monotonicity (since $S_n$ is nondecreasing).

Pick the subsequence by the growth of the mean: let $n_k=\inf\{n: ES_n\ge k^2\}$, and write $T_k=S_{n_k}$. Because each increment of $ES_n$ is at most $P(A_{n_k})\le 1$, overshooting $k^2$ by at most $1$, I get $k^2\le ET_k\le k^2+1$. Now Chebyshev at the subsequence, using $ET_k\ge k^2$:
$$P\bigl(|T_k-ET_k|>\delta ET_k\bigr)\le \frac{1}{\delta^2 ET_k}\le \frac{1}{\delta^2 k^2}.$$
And $\sum_k 1/(\delta^2 k^2)<\infty$ — summable. So by the first half of the dichotomy (the one that needs no independence), $P(|T_k-ET_k|>\delta ET_k \text{ i.o.})=0$, i.e. $T_k/ET_k\to 1$ a.s., for every $\delta>0$, hence a.s. To fill the gaps, fix an $\omega$ with $T_k/ET_k\to 1$, and for $n_k\le n<n_{k+1}$ use monotonicity of $S_n$:
$$\frac{T_k}{ET_{k+1}}\le \frac{S_n}{ES_n}\le \frac{T_{k+1}}{ET_k},$$
which I rewrite as
$$\frac{ET_k}{ET_{k+1}}\cdot\frac{T_k}{ET_k}\le \frac{S_n}{ES_n}\le \frac{T_{k+1}}{ET_{k+1}}\cdot\frac{ET_{k+1}}{ET_k}.$$
Both end factors $T_k/ET_k$, $T_{k+1}/ET_{k+1}\to 1$, so I only need $ET_{k+1}/ET_k\to 1$; and that follows from $k^2\le ET_k\le ET_{k+1}\le (k+1)^2+1$, giving $ET_{k+1}/ET_k\le \{(k+1)^2+1\}/k^2 = 1+2/k+2/k^2\to 1$. So both bounds $\to 1$ and $S_n/ES_n\to 1$ a.s. This is strictly stronger and strictly more general than what I had: it needs only *pairwise* independence, and it doesn't merely say "infinitely often" — it says the count $S_n$ is asymptotic to its mean $\sum_{m\le n}P(A_m)$ almost surely. Since that mean $\to\infty$, infinitely-often falls out as the crudest consequence ($S_n\to\infty$). The mutual-independence version I proved first is the special case where I cared only that $S_n\to\infty$.

Now the second edge: can I drop independence *entirely* and replace it by a quantitative condition? The pairwise proof leaned only on $\operatorname{Var}(S_n)$ being small relative to $(ES_n)^2$. So the real engine is a second-moment estimate, and I can ask for the bare minimum that makes the second-moment method give a *lower* bound on $P(\limsup A_n)$. Here is the Paley–Zygmund / Cauchy–Schwarz route. For the count $S_n=\sum_{k=1}^n 1_{A_k}$ with $ES_n=\sum_{k\le n}P(A_k)\to\infty$, and $E S_n^2=\sum_{1\le j,k\le n}P(A_j\cap A_k)$, Cauchy–Schwarz on $S_n=S_n\cdot 1_{\{S_n>0\}}$ gives $(ES_n)^2=\bigl(E[S_n 1_{\{S_n>0\}}]\bigr)^2\le E S_n^2\cdot P(S_n>0)$, so
$$P(S_n>0)\ge \frac{(ES_n)^2}{E S_n^2}=\frac{\bigl(\sum_{k\le n}P(A_k)\bigr)^2}{\sum_{1\le j,k\le n}P(A_j\cap A_k)}.$$
The event $\{S_n>0\}=\bigcup_{k\le n}A_k$ increases, and $\bigcup_{k\ge m}A_k\supseteq \bigcup_{m\le k\le n}A_k$ for each $m$, so taking $\limsup$ along $n$ of the right side bounds $P(\bigcup_{k\ge m}A_k)$ from below; letting $m\to\infty$,
$$P(\limsup_n A_n)\ge \limsup_{n\to\infty}\frac{\bigl(\sum_{k\le n}P(A_k)\bigr)^2}{\sum_{1\le j,k\le n}P(A_j\cap A_k)}.$$
If the events are independent (or just have $P(A_j\cap A_k)\le P(A_j)P(A_k)$ off-diagonal plus controlled diagonal) the denominator is $\approx (\sum P(A_k))^2$, the ratio $\to 1$, and I recover $P(\limsup A_n)=1$. So the independent second lemma is the $\alpha=1$ corner of a quantitative statement: whenever that limsup-ratio is some $\alpha>0$, the bad event recurs with probability at least $\alpha$. The exact form of the engine — second moment relative to mean — is now exposed, and independence was only ever a sufficient condition for it.

Good. Now let me cash the dichotomy out on the thing that started all this — turn a per-stage probability bound into an almost-sure law. The cleanest target is the strong law of large numbers under a fourth-moment assumption, because the fourth moment is exactly what makes the deviation probabilities *summable*, which is what the first half eats. Let $X_1,X_2,\dots$ be i.i.d. with $EX_i=0$ (subtract the mean) and $EX_i^4<\infty$, $S_n=\sum_{i\le n}X_i$. Expand
$$E S_n^4=E\!\left(\sum_{i=1}^n X_i\right)^4=\sum_{1\le i,j,k,\ell\le n}E[X_iX_jX_kX_\ell].$$
By independence and $EX_i=0$, any term with an index appearing exactly once has a lone $EX_i=0$ factor and vanishes; so do the all-distinct, the $X_i^3X_j$, and the $X_i^2X_jX_k$ terms. The survivors are the $X_i^4$ (there are $n$ of them, contributing $nEX_1^4$) and the $X_i^2X_j^2$ with $i\ne j$ (each equal to $(EX_1^2)^2$, and there are $3n(n-1)$ of them: choose the pair of indices in $n(n-1)/2$ ways, times $\binom{4}{2}=6$ ways to assign the four slots). So
$$E S_n^4 = nEX_1^4 + 3n(n-1)(EX_1^2)^2 \le C n^2$$
for a constant $C<\infty$. Chebyshev with the fourth power:
$$P(|S_n|>n\varepsilon)\le \frac{ES_n^4}{(n\varepsilon)^4}\le \frac{Cn^2}{n^4\varepsilon^4}=\frac{C}{n^2\varepsilon^4},$$
and $\sum_n 1/(n^2\varepsilon^4)<\infty$. The first half of the dichotomy then gives $P(|S_n|>n\varepsilon \text{ i.o.})=0$ for each $\varepsilon$, i.e. eventually $|S_n|\le n\varepsilon$; since $\varepsilon$ is arbitrary, $S_n/n\to 0$ a.s. There it is — an almost-sure statement, a true law for a fixed path, pulled straight out of summable per-stage probabilities by the convergent half. The fourth moment was needed precisely to get the $n^{-2}$ decay that makes the sum finite; a mere second moment gives only $n^{-1}$, which is *not* summable, and the bridge would not carry.

The same machine runs in the divergent direction to show that a finite mean is *necessary*. If $X_1,X_2,\dots$ are i.i.d. with $E|X_i|=\infty$, then using $E|X_1|=\int_0^\infty P(|X_1|>x)\,dx\le \sum_{n\ge 0}P(|X_1|>n)$, the divergence of $E|X_1|$ forces $\sum_n P(|X_n|\ge n)=\infty$; the $\{|X_n|\ge n\}$ are independent (the $X_n$ are), so the second half gives $P(|X_n|\ge n \text{ i.o.})=1$. And $|X_n|\ge n$ infinitely often is incompatible with $S_n/n$ converging to a finite limit (if $S_n/n\to L$ then $X_n/n=S_n/n-\tfrac{n-1}{n}\cdot S_{n-1}/(n-1)\to 0$, so $|X_n|\ge n$ can hold only finitely often) — so the strong law *fails* whenever the mean is infinite. The two halves of the dichotomy are the two directions of "finite mean $\Leftrightarrow$ strong law."

Let me also confirm the dichotomy on the bare independent test profiles, where I can see it without any application dressing. Independent $A_n$ with $P(A_n)=1/n^2$: $\sum 1/n^2<\infty$, convergent half $\Rightarrow$ only finitely many occur, so along a path the occurrences stop. Independent $A_n$ with $P(A_n)=1/n$: $\sum 1/n=\infty$, divergent half $\Rightarrow$ infinitely many occur, so the occurrences never stop — and by the pairwise sharpening the *count* $S_n$ tracks $\sum_{k\le n}1/k\sim\log n$. Two profiles, opposite verdicts, decided purely by whether one series converges.

It's worth one small concrete check that the criterion behaves as claimed, just to feel the dichotomy on a sampled path rather than only in the abstract — the kind of thing a single realization makes vivid. Simulate one trajectory of independent events with each profile and watch how far out the occurrences keep happening. For $1/n^2$ the last occurrence sits at some tiny index and nothing happens thereafter; for $1/n$ the occurrences keep cropping up arbitrarily far out, and the running count divided by its mean $\sum_{k\le n}1/k$ hovers near $1$. I can also numerically confirm the one inequality the whole divergent half rests on, $1-x\le e^{-x}$ on $[0,1]$, by checking $(1-x)-e^{-x}\le 0$ across a grid. None of this proves anything — the theorems are the proofs above — but it makes the two verdicts tangible.

```python
import math, random

def last_occurrence_and_count(prob_fn, N, seed=0):
    """One realization of independent events A_1..A_N with P(A_n)=prob_fn(n).
    Returns (number that occurred, largest index that occurred)."""
    rng = random.Random(seed)
    occurred, last = 0, 0
    for n in range(1, N + 1):
        if rng.random() < prob_fn(n):          # draw 1_{A_n} ~ Bernoulli(P(A_n))
            occurred += 1
            last = n
    return occurred, last

def running_frequency(prob_fn, N, seed=0):
    """S_n / E[S_n] for the divergent profile: the count tracked against its mean
    (the strong-law sharpening: S_n / sum_{k<=n} P(A_k) -> 1)."""
    rng = random.Random(seed)
    S, ES, snap = 0.0, 0.0, {}
    for n in range(1, N + 1):
        p = prob_fn(n)
        ES += p                                 # E[S_n] = sum_{k<=n} P(A_k)
        if rng.random() < p:
            S += 1
        if n in (1000, 10000, 100000, N):
            snap[n] = S / ES
    return snap

if __name__ == "__main__":
    # the hinge of the divergent half: 1 - x <= e^{-x}, i.e. (1-x) - e^{-x} <= 0
    worst = max((1 - x) - math.exp(-x) for x in [i / 100 for i in range(101)])
    print(f"max of (1-x) - e^(-x) on [0,1]: {worst:.1e}   (<= 0 confirms 1-x <= e^(-x))")

    N = 200000
    summable  = lambda n: 1.0 / (n * n)         # sum 1/n^2 < inf  -> first lemma: finitely often
    divergent = lambda n: 1.0 / n               # sum 1/n  = inf, indep -> second lemma: infinitely often
    oc, lc = last_occurrence_and_count(summable, N)
    od, ld = last_occurrence_and_count(divergent, N)
    print(f"P(A_n)=1/n^2 : occurrences={oc:6d}, last index seen={lc:7d}   (settles -> finite)")
    print(f"P(A_n)=1/n   : occurrences={od:6d}, last index seen={ld:7d}   (keeps recurring)")
    print("divergent S_n / E[S_n] (should approach 1):",
          {n: round(v, 3) for n, v in sorted(running_frequency(divergent, N).items())})
```

The causal chain, start to finish: I wanted almost-sure tail statements but had only per-stage probabilities, so I encoded "bad thing infinitely often" as the set $\limsup_n A_n=\bigcap_n\bigcup_{k\ge n}A_k$ and asked for $P$ of it. Continuity from above turned that into a limit of union-probabilities; the union bound plus the vanishing tail of a convergent series gave, with no structural assumption, that $\sum P(A_n)<\infty\Rightarrow P(\limsup A_n)=0$ — equivalently, the expected number of occurrences $\sum P(A_n)$ being finite forces finitely many occurrences a.s. The converse failed unconditionally — nested events show divergence can coexist with the bad event never recurring — and the failure mode (events reusing the same outcomes) named the fix: independence, the assumption that lets the intersection of complements factor as $\prod(1-P(A_k))$; the inequality $1-x\le e^{-x}$ collapsed that product into $\exp(-\sum P(A_k))\to 0$ on a divergent sum, giving $\sum P(A_n)=\infty$ with independence $\Rightarrow P(\limsup A_n)=1$. Pushing the second moment showed mutual independence was overkill — pairwise independence already yields the stronger $S_n/\sum_{k\le n}P(A_k)\to 1$ a.s. — and the Cauchy–Schwarz lower bound $P(\limsup A_n)\ge \limsup (\sum P(A_k))^2/\sum P(A_j\cap A_k)$ exposed the second-moment engine of which independence is just the cleanest sufficient case. Feeding the convergent half summable fourth-moment deviation bounds delivered the strong law $S_n/n\to\mu$ a.s., and the divergent half showed a finite mean is necessary for it.
