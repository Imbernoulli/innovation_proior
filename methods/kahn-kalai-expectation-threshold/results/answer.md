# The expectation threshold is within a log factor of the threshold

**Problem.** For an increasing property $\mathcal F\subseteq 2^X$ on a finite set $X$ ($|X|=n$), the *threshold* $p_c(\mathcal F)$ is the unique $p$ with $\mu_p(\mathcal F)=1/2$, where $\mu_p$ is the product measure including each element with probability $p$. Call $\mathcal F$ *$p$-small* if it has a cheap cover: a family $\mathcal G$ with $\mathcal F\subseteq\langle\mathcal G\rangle=\bigcup_{S\in\mathcal G}\{T:T\supseteq S\}$ and $\sum_{S\in\mathcal G}p^{|S|}\le 1/2$. The *expectation threshold* $q(\mathcal F)$ is the largest $p$ for which $\mathcal F$ is $p$-small. Since $\mu_p(\mathcal F)\le\sum_{S\in\mathcal G}p^{|S|}$ (union bound, and $\sum_S p^{|S|}=\mathbb E|\{S\in\mathcal G:S\subseteq X_p\}|$), one has $q(\mathcal F)\le p_c(\mathcal F)$ for free. The question is how far above this naive lower bound the true threshold can lie.

**Key idea.** Take the contrapositive and avoid the "spread"/LP-duality machinery used for the fractional version entirely. If $\mathcal F$ has *no* cheap cover, sample a random set in increments and, at each round, peel off part of the hypergraph using the **minimum fragment** $T(S,W)$ — the *smallest* leftover $S'\setminus W$ of an edge $S'$ sitting inside $W\cup S$. Minimality makes the cover automatically cheap (a two-step counting argument with no spread hypothesis) and shrinks edges geometrically. After $\Theta(\log\ell)$ rounds the process degenerates into an exhaustive dichotomy: either the accumulated sample already contains a member of $\mathcal F$, or the peeled-off pieces assemble into a cheap cover. The second is forbidden (no cheap cover exists), except with vanishing probability by Markov, so the sample succeeds with high probability.

## Theorems

Let $\ell(\mathcal F)=\max\big(2,\ \text{size of the largest minimal member of }\mathcal F\big)$.

**Theorem (Kahn–Kalai expectation-threshold).** There is a universal constant $K$ such that for every finite set $X$ and every nontrivial increasing $\mathcal F\subseteq 2^X$,
$$p_c(\mathcal F)\ \le\ K\,q(\mathcal F)\,\log\ell(\mathcal F).$$

This follows from the hypergraph reformulation (all logarithms base $2$; floors/ceilings suppressed; $\langle\mathcal H\rangle=\bigcup_{S\in\mathcal H}\{T:T\supseteq S\}$; $\mathcal H$ is *$\ell$-bounded* if every edge has size $\le\ell$):

**Theorem (reformulation).** Let $\ell\ge 2$. There is a universal constant $L$ such that for any nonempty $\ell$-bounded hypergraph $\mathcal H$ on $X$ that is **not** $p$-small, a uniformly random $\big((Lp\log\ell)\,|X|\big)$-element subset of $X$ belongs to $\langle\mathcal H\rangle$ with probability $1-o_{\ell\to\infty}(1)$. (Quantitatively the exceptional probability is $(\log\ell)^{-c}$ for some $c>0$.)

## Proof of the reformulation

Write $n=|X|$, $w=Lpn$, and let $W\in\binom{X}{w}$.

**Minimum fragment.** For $S\in\mathcal H$ and $W\subseteq X$, let $T=T(S,W)$ be the smallest set (possibly empty) of the form $S'\setminus W$ with $S'\in\mathcal H$ and $S'\subseteq W\cup S$; set $t=|T|$. (It exists: $S$ itself works.) Define
$$\mathcal G(W)=\{S\in\mathcal H: t(S,W)\ge 0.9\ell\},\quad \mathcal U(W)=\{T(S,W):S\in\mathcal G(W)\},\quad \mathcal H'(W)=\{T(S,W):S\in\mathcal H\setminus\mathcal G(W)\}.$$
Each $T(S,W)\subseteq S$, so $\mathcal U(W)$ covers $\mathcal G(W)$ and $\mathcal H\setminus\mathcal G(W)\subseteq\langle\mathcal H'(W)\rangle$; and $\mathcal H'(W)$ is $0.9\ell$-bounded.

**Cost Lemma.** For $W$ uniform from $\binom{X}{w}$,
$$\mathbb E\Big[\sum_{U\in\mathcal U(W)}p^{|U|}\Big]<L^{-0.8\ell}\quad(\text{for }L\text{ a large enough absolute constant}),$$
equivalently $\sum_{W\in\binom{X}{w}}\sum_{U\in\mathcal U(W)}p^{|U|}<\binom{n}{w}L^{-0.8\ell}$.

*Proof.* For $m\ge 0.9\ell$ let $\mathcal U_m(W)=\{T(S,W):t(S,W)=m\}$; every $U\in\mathcal U_m(W)$ has $|U|=m$, so
$$\sum_{W}\sum_{U\in\mathcal U_m(W)}p^{|U|}=p^m\cdot\big|\{(W,T(S,W)):W\in\tbinom{X}{w},\ S\in\mathcal H,\ t(S,W)=m\}\big|.$$
Count the pairs $(W,T)$ in two steps.

- **Step 1 (specify $Z=W\cup T$).** $W,T$ are disjoint and $|T|=m$, so $|Z|=w+m$ and
$$\binom{n}{w+m}=\binom{n}{w}\prod_{j=0}^{m-1}\frac{n-w-j}{w+j+1}\le\binom{n}{w}\Big(\frac{n}{w}\Big)^{m}=\binom{n}{w}(Lp)^{-m},$$
since each factor is $\le n/w=1/(Lp)$ (numerator decreasing, denominator increasing).
- **Step 2 (specify $T$ inside $Z$).** $Z=W\cup T$ contains an edge of $\mathcal H$ (the $S'$ realizing $T$); choose an edge $\hat S\subseteq Z$ as a function of $Z$ only (free given $Z$). Minimality forces $T\subseteq\hat S$: otherwise $\hat S\setminus W\subsetneq T$ would be a strictly smaller fragment of $(S,W)$. So $T$ is one of at most $2^{|\hat S|}\le 2^\ell$ subsets of $\hat S$.

Then $W=Z\setminus T$ is determined, so the number of pairs is $\le\binom{n}{w}(Lp)^{-m}2^\ell$, and the size-$m$ contribution is $\le p^m\binom{n}{w}(Lp)^{-m}2^\ell=\binom{n}{w}L^{-m}2^\ell$. Summing,
$$\sum_{m\ge 0.9\ell}\binom{n}{w}L^{-m}2^\ell=\binom{n}{w}\,2^\ell\,\frac{L^{-0.9\ell}}{1-1/L}\le\binom{n}{w}L^{-0.8\ell},$$
the last inequality holding once $L^{0.1}>2$ (then $2^\ell L^{-0.9\ell}\le L^{-0.8\ell}$ with room to spare). Dividing by $\binom{n}{w}$ gives the lemma. $\qquad\blacksquare$

*(Remark.* Specifying $Z=W\cup T$ instead of $Z=W\cup S$, with $T$ the minimum fragment, is what makes Step 2's containment automatic and removes the "pathological/non-pathological" case split of the earlier sunflower-style arguments.)*

**Iteration.** Let $\ell_i=0.9^i\ell$ and $\gamma=\lfloor\log_{0.9}(1/\ell)\rfloor+1$, so $0.9\le\ell_\gamma<1$. Set $X_0=X$; for $i=1,\dots,\gamma$ let $W_i\in\binom{X_{i-1}}{w_i}$, $X_i=X_{i-1}\setminus W_i$, $w_i=L_ipn$ with
$$L_i=\begin{cases}L & i<\gamma-\sqrt{\log_{0.9}(1/\ell)},\\ L\sqrt{\log\ell} & \gamma-\sqrt{\log_{0.9}(1/\ell)}\le i\le\gamma.\end{cases}$$
Let $\mathcal H_0=\mathcal H$ and $\mathcal H_i=\mathcal H_{i-1}'(W_i)$, with $\mathcal G_i=\mathcal G_i(W_i)$, $\mathcal U_i=\mathcal U_i(W_i)$. Then $\mathcal H_i$ is $\ell_i$-bounded, and $W:=\bigcup_{i=1}^\gamma W_i$ is a uniformly random $(CLp\log\ell)n$-subset for an absolute $C$.

*Capture (by induction on $i$):* for all $S_i\in\mathcal H_i$, $(\bigcup_{j\le i}W_j)\cup S_i\in\langle\mathcal H\rangle$. Indeed $S_i=T(S_{i-1},W_i)=S_{i-1}'\setminus W_i$ for some $S_{i-1}'\in\mathcal H_{i-1}$, so $(\bigcup_{j\le i}W_j)\cup S_i\supseteq(\bigcup_{j\le i-1}W_j)\cup S_{i-1}'\in\langle\mathcal H\rangle$.

*Dichotomy:* since $\ell_\gamma<1$, $\mathcal H_\gamma=\emptyset$ or $\mathcal H_\gamma=\{\emptyset\}$.
- If $\mathcal H_\gamma=\emptyset$: tracing $S=S_0,S_1,\dots$ ($S_i=T(S_{i-1},W_i)$), each $S$ is peeled at some step $j<\gamma$ (i.e. $S_j\in\mathcal G_j$, covered by $\mathcal U_j$), so $\mathcal U:=\bigcup_{i\le\gamma}\mathcal U_i$ covers $\mathcal H$.
- If $\mathcal H_\gamma=\{\emptyset\}$: capture with $S_\gamma=\emptyset$ gives $W=\bigcup_i W_i\in\langle\mathcal H\rangle$.

Hence $\Pr(W\in\langle\mathcal H\rangle)+\Pr(\mathcal E)\ge 1$, where $\mathcal E=\{\mathcal U\text{ covers }\mathcal H\}$.

**Bounding $\Pr(\mathcal E)$.** Because $\mathcal H$ is *not* $p$-small, any cover has $\sum_U p^{|U|}>1/2$; so $\mathcal E\Rightarrow\sum_{U\in\mathcal U}p^{|U|}>1/2$. By the Cost Lemma applied at each $(\ell_i,L_i)$,
$$\mathbb E\Big[\sum_{U\in\mathcal U}p^{|U|}\Big]<\sum_{i\le\gamma}L_i^{-0.8\ell_i}.$$
There is $c>0$ with $\ell_i>\exp(c\sqrt{\log\ell})$ for $i<\gamma-\sqrt{\log_{0.9}(1/\ell)}$, so the early terms sum to $\le 2L^{-0.8\exp(c\sqrt{\log\ell})}$. The late range has $O(\sqrt{\log\ell})$ terms with $\ell_i\ge 0.9$, summing to $O\big((L\sqrt{\log\ell})^{-c'}\big)$. Thus
$$\mathbb E\Big[\sum_{U\in\mathcal U}p^{|U|}\Big]<2L^{-0.8\exp(c\sqrt{\log\ell})}+O\big((L\sqrt{\log\ell})^{-c'}\big)=(\log\ell)^{-c''},$$
and by Markov $\Pr(\mathcal E)\le 2\,\mathbb E[\sum_U p^{|U|}]=(\log\ell)^{-c''}=o_{\ell\to\infty}(1)$. Therefore $\Pr(W\in\langle\mathcal H\rangle)\ge 1-(\log\ell)^{-c''}$. $\qquad\blacksquare$

## Deriving the main theorem from the reformulation

Let $\mathcal F$ be nontrivial increasing and $q>q(\mathcal F)$. Let $\mathcal H$ be the minimal elements of $\mathcal F$ (so $\langle\mathcal H\rangle=\mathcal F$); $\mathcal H$ is $\ell(\mathcal F)$-bounded and not $q$-small. Pick absolute $C$ so that with $\ell=C\ell(\mathcal F)$ the exceptional probability above is $<1/4$. Set $m=(Lq\log\ell)n$, $p'=2Lq\log\ell$, and choose $K$ so $p:=Kq\log\ell(\mathcal F)\ge p'$. The bound is vacuous if $p'>1$, so assume $p'\le1$. Then
$$\Pr(X_{p'}\in\langle\mathcal H\rangle)\ \ge\ \Pr(X_m\in\langle\mathcal H\rangle)\,\Pr(|X_{p'}|\ge m)\ \ge\ \tfrac34\,\Pr(|X_{p'}|\ge m)\ >\ \tfrac12,$$
where the last step is concentration of $|X_{p'}|\sim\mathrm{Bin}(n,p')$ about its mean $p'n=2m$, valid because $m$ is large: $\mathcal H$ not $q$-small implies $nq>1/2$ (the singleton cover $\{\{x\}\}$ has cost $nq$), so $m=(L\log\ell)\,qn>(L\log\ell)/2$. As $\mathcal F$ is increasing and $p\ge p'$, $\Pr(X_p\in\mathcal F)\ge\Pr(X_{p'}\in\mathcal F)>1/2$, i.e. $p_c(\mathcal F)\le Kq\log\ell(\mathcal F)$. Letting $q\downarrow q(\mathcal F)$ gives $p_c(\mathcal F)\le K\,q(\mathcal F)\log\ell(\mathcal F)$. $\qquad\blacksquare$

## Why it works (one paragraph)

The expectation threshold is the union-bound lower bound on $p_c$. The proof upgrades it to a matching (up to $\log\ell$) upper bound by a direct dichotomy that never invokes spread or LP duality: an incremental random sample, with the **minimum fragment** as the one new object. Minimality buys an automatic containment ($T\subseteq$ any edge inside $W\cup T$), which collapses the cover-cost count to two steps and yields $\mathbb E[\sum_U p^{|U|}]<L^{-0.8\ell}$ per round; peeling large fragments and recursing on small ones shrinks edges by $0.9$ each round; after $\Theta(\log\ell)$ rounds the hypergraph degenerates, so either the sample already contains a member or the peeled pieces form a cheap cover; the cheap cover is impossible because $\mathcal H$ is not $p$-small, so (Markov) the sample wins with probability $1-(\log\ell)^{-c}$.
