# Galois theory: the solvability-by-radicals criterion

## Problem

Decide, for a separable polynomial $f(x)\in F[x]$ over a characteristic-$0$ base field, whether its roots can be written as a *formula in radicals* — built from the coefficients by $+,-,\times,\div$ and extraction of $m$-th roots — and in particular settle whether a *single general formula* exists for every equation of degree $n$.

## Key idea

Attach to $f$, relative to $F$, the **Galois group** $G=\operatorname{Gal}(K/F)=\operatorname{Aut}_F(K)$, where $K$ is the splitting field of $f$: the automorphisms of $K$ fixing $F$ pointwise, viewed as permutations of the roots. The governing fixed-field principle is:

$$\text{a quantity in }K\text{ is rationally known over }F \iff \text{it is fixed by all of }G.$$

For every subgroup $H\le G$, the fixed quantities $K^H$ form an intermediate field $F\subseteq K^H\subseteq K$; every intermediate field $E$ gives the subgroup $\operatorname{Gal}(K/E)$. This Galois correspondence reverses inclusion, and $H\trianglelefteq G$ exactly when $K^H/F$ is Galois, with quotient $G/H\simeq\operatorname{Gal}(K^H/F)$. Thus the analytic question "what can I express in radicals" becomes the group question "what does this finite group fix." One prime-degree radical extraction, with the matching root of unity present, corresponds to a normal subgroup with cyclic quotient.

## Main theorem (Solvability in Radicals)

> Let $F$ have characteristic $0$, and let $f\in F[x]$ be separable with splitting field $K$. Then $f$ is solvable by radicals over $F$ if and only if its Galois group $G=\operatorname{Gal}(K/F)$ is a **solvable group** — i.e. there is a chain
> $$G = G_0 \;\trianglerighteq\; G_1 \;\trianglerighteq\; \cdots \;\trianglerighteq\; G_k = \{e\}$$
> with each $G_{i+1}$ normal in $G_i$ and each quotient $G_i/G_{i+1}$ abelian; equivalently, after refinement, each quotient is cyclic of prime order.

### Supporting facts and proof structure

**Kummer / simple radical extension.** If $F$ contains the $m$-th roots of unity and $\operatorname{char}F\nmid m$, then the splitting field of $x^m-a$ over $F$ has cyclic Galois group of order dividing $m$. Conversely, if $K/F$ is cyclic of degree $m$ and $F$ contains a primitive $m$-th root of unity, then $K=F(\beta)$ with $\beta^m\in F$.
*Proof (forward):* for a chosen root $\beta^m=a$, every automorphism sends $\beta$ to $\zeta\beta$ for some $\zeta\in\mu_m$. The map $\sigma\mapsto\sigma(\beta)/\beta$ embeds the Galois group into the cyclic group $\mu_m$.
*Proof (converse):* with $\sigma$ a generator and $\zeta$ a primitive $m$-th root of unity, choose $\alpha\in K$ so the Lagrange resolvent $\beta=\sum_{i=0}^{m-1}\zeta^i\sigma^i(\alpha)$ is nonzero. Then $\sigma(\beta)=\zeta^{-1}\beta$, so $\beta^m\in F$ and the conjugates of $\beta$ span the cyclic extension; hence $K=F(\beta)$.

**Solvable groups.** A finite group is *solvable* iff it has a subnormal chain with abelian quotients; refining the finite abelian quotients gives a chain with cyclic prime-order quotients. Subgroups and quotients of solvable groups are solvable, and if $N\trianglelefteq G$ with $N$ and $G/N$ solvable, then $G$ is solvable.

**($\Rightarrow$, $f$ solvable $\Rightarrow$ $G$ solvable).** If the roots lie in a radical tower, first adjoin the needed roots of unity; that is still a radical extension, and its Galois group is abelian. Over this enlarged base each radical step has cyclic Galois closure by the Kummer calculation, so a finite Galois extension $L/F$ containing the roots has solvable Galois group. Since $K\subseteq L$, restriction gives a surjection $\operatorname{Gal}(L/F)\twoheadrightarrow\operatorname{Gal}(K/F)$, and a quotient of a solvable group is solvable.

**($\Leftarrow$, $G$ solvable $\Rightarrow$ $f$ solvable).** Refine a solvable series to cyclic prime quotients. The Galois correspondence gives fixed fields
$$F=K^{G_0}\subset K^{G_1}\subset\cdots\subset K^{G_k}=K,$$
and each step $K^{G_{i+1}}/K^{G_i}$ is cyclic of prime degree. Let $E$ be $F$ with the needed roots of unity adjoined. In the compositum tower
$$E K^{G_0}\subset E K^{G_1}\subset\cdots\subset E K^{G_k}=EK,$$
each step remains cyclic, possibly of smaller degree, and Kummer realizes it by adjoining one radical. Since $E/F$ is also radical, the roots lie in a radical tower over $F$.

## Corollary (Abel–Ruffini): the general quintic is unsolvable

> For $n\ge 5$ the general equation of degree $n$ is not solvable by radicals.

The general equation over $F(a_1,\dots,a_n)$ (indeterminate coefficients) has Galois group the full symmetric group $S_n$. $S_n$ is **not solvable** for $n\ge 5$: any chain would force its normal subgroup $A_n$ to be solvable, but $A_n$ is **simple and non-abelian** for $n\ge 5$.

**$A_5$ is simple (counting).** $|A_5|=60$; conjugacy-class sizes are $1$ (identity), $15$ (double transpositions), $20$ (3-cycles), $12+12$ (5-cycles split into two classes). A normal subgroup is a union of classes containing $1$ with order dividing $60$; the only sub-sums of $\{1,15,20,12,12\}$ including $1$ that divide $60$ are $1$ and $60$. So $A_5$ is simple, and non-abelian. Thus $S_5$ (and $S_n$, $n\ge5$, since $A_n$ is simple for $n\ge5$) is not solvable.

**Concrete witness over $\mathbb{Q}$.** $f(t)=t^5-4t+2$ is irreducible by Eisenstein at $2$. Let $a=(4/5)^{1/4}$; $f'(t)=5t^4-4$, so $f$ is increasing on $(-\infty,-a)$ and $(a,\infty)$ and decreasing on $(-a,a)$. Since $f(-a)=2+16a/5>0$, $f(0)=2>0$, and $f(a)=2-16a/5<0$, it has exactly three real roots and one complex-conjugate pair. Complex conjugation therefore acts as a transposition. Irreducibility gives a transitive subgroup of $S_5$, so $5\mid |G|$ and Cauchy's theorem gives a $5$-cycle. A $5$-cycle and any transposition generate $S_5$: conjugating the transposition by powers of the cycle gives transpositions along a connected graph on the five letters. Hence $G=S_5$, so this quintic is not solvable by radicals.

## Application to prime degree (Galois's headline)

> An irreducible equation of prime degree $p$ is solvable by radicals $\iff$ its Galois group consists only of affine substitutions $x_k\mapsto x_{ak+b}\ (\mathrm{mod}\ p)$ $\iff$ **every root is a rational function of any two of them.**

The affine map fixes two letters only if it is the identity, so solvability forces "two roots determine the rest"; conversely that condition caps the group at $p(p-1)$ elements with a single cyclic substitution, pinning it to affine form, which is solvable ($C_p \rtimes C_{p-1}$ has the chain $\{e\}\trianglelefteq C_p \trianglelefteq \text{affine}$ with cyclic quotients).

The Galois group is the structural invariant that makes "solvable by radicals" identical to "the group is solvable," recovering Cardano/Ferrari ($S_3,S_4$ solvable: $S_4\trianglerighteq A_4\trianglerighteq V_4\trianglerighteq C_2\trianglerighteq\{e\}$) and barring the quintic ($S_5$ not solvable).
