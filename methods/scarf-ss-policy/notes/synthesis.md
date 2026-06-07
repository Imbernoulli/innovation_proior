# Synthesis — Scarf (s,S) optimality via K-convexity

## Problem
Single product, periodic review, T periods, random demand $w_t$ (i.i.d. or independent), backlogging.
State $x$ = inventory level at start of period. Order $u\ge0$, delivered instantly, post-order level $y=x+u$.
Costs per period: linear ordering $cu$ PLUS a FIXED cost $K>0$ whenever $u>0$; convex holding+shortage
$L(x)=c_H x^+ + c_B x^-$ charged on end-of-period inventory $y-w$. Discount $\gamma$.
Minimize expected total cost over horizon. Terminal cost $v_N$ convex nonneg.

## Bellman recursion
$J_N^*(x)=v_N(x)$.
$J_k^*(x)=\min_{y\ge x}\{K\delta(y-x)+c(y-x)+L(y)+\gamma E_w[J_{k+1}^*(y-w)]\}$, $\delta=1$ if arg>0.
Define $G_k(y)=cy+\bar L(y)+\gamma E_w[J_{k+1}^*(y-w)]$ where $\bar L(y)=E_w L(y-w)$.
Then $J_k^*(x)=\min\{G_k(x),\ \min_{y>x}(K+G_k(y))\}-cx$.
(stockpyl uses $\theta_t(x)=\min_{y\ge x}\{K\delta+c(y-x)+g(y)+\gamma E\theta_{t+1}(y-D)\}$, same thing; $H_t(y)=g(y)+\gamma E[\theta_{t+1}(y-D)]$ precomputed.)

## Without K: base-stock optimal
Lemma: if $J_{k+1}^*$ convex nonneg → $G_k$ convex coercive ($\lim_{|y|\to\infty}G_k=\infty$ from $c+c_H>0$ and $c_B>c$).
So $G_k$ has unconstrained min $S_k$. Constrained min over $y\ge x$: if $x<S_k$ go to $S_k$, else stay → base-stock.
Then $J_k^*(x)=G_k(\max(x,S_k))-cx$ is convex nonneg → induction closes. Base-stock policy optimal.

## With K: convexity breaks
$J_k^*(x)$ has a downward jump of size up to $K$ at $s_k$ (cost-of-not-ordering meets cost-of-ordering), so NOT convex.
A convex-preservation induction fails. Need a weaker property preserved by the DP operator that still forces (s,S).

## K-convexity (Scarf)
Def (Def 3.3.1): $f$ is $K$-convex ($K\ge0$) if for all $y\le y'$, $0\le\theta\le1$:
$f(\theta y+(1-\theta)y')\le \theta f(y)+(1-\theta)(K+f(y'))$.
Equivalent (Lemma 3.3.2.1, the "Scarf inequality"), for all $y\in\mathbb R$, $a\ge0$, $b>0$:
$$K+f(y+a)\ \ge\ f(y)+\frac{a}{b}\,[f(y)-f(y-b)].$$
Differentiable form: $K+f(y)\ge f(x)+f'(x)(y-x)$ for $x\le y$.
$K=0$ ⇒ ordinary convexity. K-convex fn need not be convex/quasiconvex; can have several local minima;
discontinuities must jump DOWN and be $\le K$.

### Preservation properties (Lemma 3.3.2)
- (3) If $f_1$ $K$-convex and $f_2$ $L$-convex, $\alpha,\beta>0$, then $\alpha f_1+\beta f_2$ is $(\alpha K+\beta L)$-convex.
- (4) If $f$ $K$-convex and $w$ r.v. (integrable), $E_w[f(y-w)]$ is $K$-convex.
- (also) Lemma 3.3.1: $f$ $K$-convex, $y<y'$, $f(y)=K+f(y')$ ⇒ $f(z)\le K+f(y')$ on $[y,y']$.
  (crosses level $K+f(y)$ at most once on $(-\infty,y)$.)

### (s,S) from K-convexity (Lemma 3.3.3)
If $f$ continuous coercive $K$-convex, exist $s\le S$ with:
1. $S$ minimizes $f$.
2. $f(S)+K=f(s)$ and $f(y)>f(s)$ for $y<s$.  ($s$ = smallest $z\le S$ with $f(S)+K=f(z)$)
3. $f$ decreasing on $(-\infty,s)$.
4. $f(y)\le f(y')+K$ for all $s\le y\le y'$.
Apply to $G_k$ ⇒ for $x<s_k$ order to $S_k$ (gain $>K$), for $x\ge s_k$ don't (no $y>x$ beats $x$ by $K$). (s,S) optimal.

### Closing the induction (Lemma 3.3.4 / 3.3.5)
$J_{k+1}^*$ $K$-convex nonneg cont ⇒ $G_k=cy+\bar L+\gamma E[J_{k+1}^*]$ is $K$-convex (props 3 & 4; $cy$ and convex $\bar L$ are 0-convex), coercive, continuous.
Then $J_k^*(x)+cx =: \tilde G_k(x) = K+G_k(S_k)$ for $x\le s_k$ (constant), $=G_k(x)$ for $x\ge s_k$.
Show $\tilde G_k$ $K$-convex by Def 3.3.1, three cases on $y\le y'$:
- $s_k\le y\le y'$: $\tilde G_k=G_k$, $K$-convex. ✓
- $y\le y'\le s_k$: $\tilde G_k$ constant. ✓
- $y<s_k<y'$ (the hard case): $\tilde G_k(y)=G_k(s_k)=K+G_k(S_k)$. The chord goes from $(y,G_k(s_k))$ to $(y',G_k(y')+K)$. Since $G_k(y')\ge G_k(S_k)$, $K+G_k(y')\ge G_k(s_k)=\tilde G_k(y)$ → chord is "increasing". For $z\in[y,s_k]$: $\tilde G_k$ constant $=G_k(s_k)\le$ chord (chord increasing, starts at $G_k(s_k)$). For $z\in[s_k,y']$: by $K$-convexity of $G_k$ on $[s_k,y']$, $G_k(z)$ lies below chord $(s_k,G_k(s_k))\to(y',G_k(y')+K)$; that chord lies below the chord of interest (same right endpoint, left endpoint $s_k\ge y$ at equal height $G_k(s_k)$ on an increasing line). ✓
So $J_k^*$ $K$-convex nonneg cont → induction holds for all $k$. Base $J_N=v_N$ convex(=0-convex) nonneg. □

## (s,S) characterization for code
$S_k=\arg\min_y G_k(y)$. $s_k=$ smallest/largest $x\le S_k$ s.t. $G_k(x)=K+G_k(S_k)$.
Order iff $x<s_k$, up to $S_k$. Without $K$: $s_k=S_k$ (base-stock).
stockpyl extracts: $S_t$=OUL at smallest x; $s_t$=largest $x$ with OUL$(x)=S_t$.

## Ancestors / lineage
- Newsvendor (Edgeworth 1888 banking; formalized Arrow–Harris–Marschak 1951): single period, critical fractile $F(S^*)=p/(p+h)$. No fixed cost; one period.
- Arrow, Harris, Marschak 1951 (Econometrica 19:250-272, RAND 1950 Logistics conf): static + dynamic uncertainty; identified "best max stock S and reorder point s" as functions of demand dist, order cost, depletion penalty. Posed the (S,s)-shaped policy but did not prove general optimality.
- Bellman 1957 Dynamic Programming: the DP/functional-equation machinery; principle of optimality; backward recursion. The tool that turns the multi-period problem into a one-step operator.
- Dvoretzky–Kiefer–Wolfowitz, "The inventory problem I/II" Econometrica 20 (1952); "On the optimal character of the (s,S) policy in inventory theory" Econometrica 21 (1953): proved (s,S) optimality but under restrictive/structural conditions and case analyses, not a single clean structural property preserved by the recursion.
- Scarf 1960 ("The optimality of (S,s) policies in the dynamic inventory problem", in Arrow/Karlin/Suppes eds., Mathematical Methods in the Social Sciences 1959, Stanford UP, pp. 196-202): introduces K-convexity, the property preserved by min+expectation that yields (s,S) in general.

## Code (canonical = stockpyl finite_horizon.finite_horizon_dp)
Backward DP over discretized x. Precompute $H_t(y)=g(y)+\gamma E[\theta_{t+1}(y-D)]$ once per (t,y).
$\theta_t(x)=\min_{y\ge x}\{K\delta(y-x)+c(y-x)+H_t(y)\}$. Extract reorder_points[t]=$s_t$, order_up_to_levels[t]=$S_t$.
If K=0, $s_t=S_t$. Doctest: finite_horizon_dp(5,1,20,1,20,2,50,100,20) → s=[_,110,110,110,110,111], S=[_,133,133,133,133,126], cost≈1558.69.

## Unsourced/uncertain
- Exact page numbers (196-202) of Scarf 1960 from secondary search; symposium year 1959/pub 1960.
- DKW exact theorem conditions not read in original; characterized from secondary survey statements.
