# Cauchy's Functional Equation and Its Monsters

Find all functions $f:\mathbb{R}\to\mathbb{R}$ satisfying
$$f(x+y)=f(x)+f(y)\qquad(x,y\in\mathbb{R}).$$

The complete answer has two parts.

Without any regularity assumption, the solutions are exactly the $\mathbb{Q}$-linear maps
$\mathbb{R}\to\mathbb{R}$. Equivalently, choose a Hamel basis
$H=\{e_\alpha\}$ of $\mathbb{R}$ over $\mathbb{Q}$, choose arbitrary real values
$v_\alpha$, and define
$$
f\Big(\sum_i q_i e_{\alpha_i}\Big)=\sum_i q_i v_{\alpha_i},
\qquad q_i\in\mathbb{Q}.
$$
This is additive, and every additive solution is obtained this way. The special choices
$v_\alpha=c e_\alpha$ for one fixed real $c$ give the ordinary solutions
$$f(x)=cx.$$
Every other choice gives a wild solution. Its graph is dense in $\mathbb{R}^2$, so it is
discontinuous everywhere, unbounded above and below on every interval, non-monotone,
and non-Lebesgue-measurable. There are $2^{\mathfrak c}$ wild solutions, while the
ordinary line solutions form a family of size $\mathfrak c$.

With any one of the following regularity hypotheses, the only solutions are
$$f(x)=cx,\qquad c=f(1):$$

- $f$ is continuous at one point;
- $f$ is monotone;
- $f$ is bounded on some nondegenerate interval;
- $f$ is Lebesgue measurable.

## Proof

First, the equation forces rational linearity. From $x=y=0$,
$f(0)=0$. From $y=-x$, $f(-x)=-f(x)$. By induction,
$f(nx)=nf(x)$ for every integer $n$, and from
$f(x)=f(n\cdot x/n)=n f(x/n)$,
$$
f\!\left(\frac mn x\right)=\frac mn f(x)
\qquad(m\in\mathbb{Z},\ n\in\mathbb{Z}_{>0}).
$$
Thus, with $c=f(1)$, $f(q)=cq$ for every rational $q$.

This is as far as the equation alone reaches. View $\mathbb{R}$ as a vector space over
$\mathbb{Q}$. Zorn's Lemma, equivalently the usual Choice principle used for arbitrary
vector-space bases, gives a Hamel basis $H$: every real $x$ has a unique finite
representation
$$
x=\sum_i q_i e_{\alpha_i},\qquad q_i\in\mathbb{Q},\ e_{\alpha_i}\in H.
$$
Assigning arbitrary values on $H$ and extending $\mathbb{Q}$-linearly gives an additive
function. Conversely, every additive function is $\mathbb{Q}$-linear by the rational
scaling just proved, so it is determined by its values on $H$.

If $f$ is not of the form $cx$, its graph is dense. Choose nonzero $x_1,x_2$ with
$f(x_1)/x_1\ne f(x_2)/x_2$. Then
$$
\det\begin{pmatrix}x_1&x_2\\ f(x_1)&f(x_2)\end{pmatrix}\ne0,
$$
so $v_1=(x_1,f(x_1))$ and $v_2=(x_2,f(x_2))$ are an $\mathbb{R}$-basis of the plane.
For rationals $q_1,q_2$,
$$
q_1v_1+q_2v_2=
\big(q_1x_1+q_2x_2,\ f(q_1x_1+q_2x_2)\big),
$$
so the graph contains all rational combinations of $v_1,v_2$. Since $\mathbb{Q}^2$ is
dense in $\mathbb{R}^2$, the graph is dense in the plane.

Each regularity condition kills the dense-graph alternatives.

If $f$ is continuous at $x_0$, then it is continuous everywhere: for $x_n\to a$,
$$
f(x_n)=f(x_n-a+x_0)+f(a-x_0)\to f(x_0)+f(a-x_0)=f(a).
$$
Then, for rationals $q_n\to x$,
$$f(x)=\lim_n f(q_n)=\lim_n cq_n=cx.$$

If $f$ is bounded on $[a,b]$, translate the bound to the origin. For
$t=x-a\in[0,b-a]$, $f(t)=f(x)-f(a)$, so $f$ is bounded on $[0,b-a]$, and by oddness
on $[-\delta,\delta]$ for $\delta=b-a$. Say $|f|\le M$ there. If
$|t|<\delta/N$, then $Nt\in[-\delta,\delta]$ and
$$
|f(t)|=\frac1N|f(Nt)|\le \frac MN.
$$
Given $\varepsilon>0$, choose $N>M/\varepsilon$; this proves continuity at $0$, hence
linearity.

If $f$ is monotone, the nonincreasing case follows by applying the nondecreasing case
to $-f$. So assume $f$ is nondecreasing. Then $c=f(1)\ge0$. For any real $x$, choose
rationals $p_n\uparrow x$ and $q_n\downarrow x$. Since $p_n\le x\le q_n$,
$$
cp_n=f(p_n)\le f(x)\le f(q_n)=cq_n.
$$
Both bounds converge to $cx$, so $f(x)=cx$.

If $f$ is Lebesgue measurable, Lusin's theorem gives continuity on a large compact set,
and additivity pulls that continuity back to $0$. Fix $\varepsilon>0$. Choose compact
$F\subset[0,1]$ with $m=\mu(F)>3/4$ such that $f$ is continuous on $F$. Uniform
continuity on $F$ gives $\eta>0$ such that
$|f(x)-f(y)|<\varepsilon$ whenever $x,y\in F$ and $|x-y|<\eta$. Put
$$\delta=\min(\eta,2m-1)>0.$$
For $0<h<\delta$, the sets $F$ and $F-h$ both lie in $[-h,1]$. They cannot be
disjoint, because then their union would have measure $2m$, while $[-h,1]$ has
measure $1+h<2m$. Hence some $x_0\in F\cap(F-h)$ satisfies $x_0,x_0+h\in F$, and
therefore
$$
|f(h)|=|f(x_0+h)-f(x_0)|<\varepsilon.
$$
Oddness gives the same estimate for negative $h$, so $f$ is continuous at $0$, and the
continuity argument gives $f(x)=cx$.

The only obstruction to the line solutions is the Choice-built freedom to assign values
independently on a Hamel basis. Order, topology, boundedness, or measurability removes
that freedom and leaves only $f(x)=cx$.
