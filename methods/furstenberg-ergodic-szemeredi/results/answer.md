# The Furstenberg ergodic-theoretic proof of Szemerédi's theorem

## Problem

**Szemerédi's theorem.** If $A \subseteq \mathbb{Z}$ has positive upper density
$\bar d(A) = \limsup_N |A \cap [-N,N]|/(2N+1) > 0$, then for every $k$, $A$ contains a
$k$-term arithmetic progression $a, a+n, \dots, a+(k-1)n$ with $n \neq 0$.

## Key idea

Raw upper density is shift-invariant but not a measure. The correspondence principle
chooses a density-realizing averaging scheme, refines it to a shift-invariant Banach
mean on the relevant shift algebra, and realizes that mean as an invariant probability
measure on a compact shift system. This turns arithmetic progressions in $A$ into a
multiple-recurrence statement. Multiple recurrence is then proved by a
structure-vs-randomness dichotomy: weak mixing gives product-limit behavior, compact
systems give almost-periodic returns, and the general system is reduced to a finite
distal tower built from compact extensions.

## The correspondence principle

Given $A$ with $\bar d(A) > 0$, identify $A$ with $a \in \{0,1\}^{\mathbb{Z}}$. Let
$T$ be the left shift, $(Tx)(m)=x(m+1)$ (on subsets, $B\mapsto B-1$). Let
$X = \overline{\{T^n a : n \in \mathbb{Z}\}}$ be the orbit closure, and let
$E=\{x\in X:x(0)=1\}$ be the clopen cylinder. Then $T^m a \in E$ iff $m\in A$.
Form the empirical measures
$$\mu_N = \frac{1}{2N+1}\sum_{n=-N}^{N} \delta_{T^n a}, \qquad \mu_N(E) = \frac{|A\cap[-N,N]|}{2N+1}.$$
Pick $N_j \to \infty$ with $\mu_{N_j}(E) \to \bar d(A)$, and let $\mu$ be a weak-$*$
limit (Banach–Alaoglu). Then:
- $\mu(E) = \bar d(A) > 0$ (since $1_E$ is continuous);
- $\mu$ is $T$-invariant, because $\|T\mu_N - \mu_N\|_{\mathrm{TV}} = \frac{1}{2N+1}\|\delta_{T^{N+1}a} - \delta_{T^{-N}a}\| \le \frac{2}{2N+1} \to 0$.

So $(X,\mathcal{B},\mu,T)$ is a measure-preserving system with $\mu(E) > 0$. If
$E \cap T^{-n}E \cap \dots \cap T^{-(k-1)n}E$ is nonempty, some orbit point
$T^m a$ lies in that clopen intersection, and
$m,m+n,\dots,m+(k-1)n$ is a non-degenerate arithmetic progression in $A$.

**Equivalence with multiple recurrence (both directions).** Recurrence $\Rightarrow$
Szemerédi as above. Conversely, if $\mu(E) > 0$, define
$$\delta_N(x)=\frac{1}{2N+1}\big|\{|n|\le N:T^n x\in E\}\big|.$$
Since $\int \delta_N\,d\mu=\mu(E)$, the sets
$A_N=\{\delta_N\ge \mu(E)/2\}$ have measure bounded below by a positive constant, and
$F=\limsup_N A_N$ has positive measure. Every $x\in F$ has a return set of positive
upper density. Szemerédi applied to those return sets, plus countability of the
possible progressions $(a,n)$ and an invariant shift by $T^a$, gives
$\mu(E\cap T^{-n}E\cap\dots\cap T^{-(k-1)n}E)>0$.

## The theorem to prove

**Furstenberg multiple recurrence.** For every measure-preserving
$(X,\mathcal{B},\mu,T)$, every $B$ with $\mu(B) > 0$, and every $k \ge 1$, there is
$n > 0$ with
$$\mu\big(B \cap T^{-n}B \cap \dots \cap T^{-(k-1)n}B\big) > 0,$$
and in fact $\liminf_{N\to\infty}\frac{1}{N}\sum_{n=1}^{N}\mu(B \cap T^{-n}B \cap \dots \cap T^{-(k-1)n}B) > 0$.

- $k=2$ is **Poincaré recurrence** ($B, T^{-1}B, \dots$ have equal measure in a
  probability space, so two overlap).

## Proof of the $k=3$ case (Roth), in full

Reduce to ergodic $(X,\mathcal{B},\mu,T)$ by ergodic decomposition.

**Weak-mixing case.** If $T$ is weakly mixing then $T^q$ is ergodic for all $q$, the
diagonal measure on $X^k$ equidistributes to $\mu^k$ under $T\times T^2\times\dots\times
T^k$ (induction + a mean ergodic theorem along the singular generic measure), so
$\frac{1}{N}\sum_n \mu(B \cap T^{-n}B \cap T^{-2n}B) \to \mu(B)^3 > 0$.

**Kronecker (compact) case.** $X = G$ compact abelian, $Tx = g_0 x$, $g_0$ generating a
dense subgroup, $\mu = $ Haar. For $0 \le f \in L^\infty$, $f \not\equiv 0$, set
$$\phi(z') = \int_G f(z)\,f(zz')\,f(zz'^2)\,dm_G(z).$$
$\phi$ is continuous ($z' \mapsto f(\cdot z')$ is $L^2$-continuous), nonnegative, and
$\phi(e) = \int f^3 > 0$. By unique ergodicity, $\frac1N\sum_n \phi(g_0^n) \to \int_G
\phi\,dm_G > 0$, so $\frac1N\sum_n \int f\cdot T^n f\cdot T^{2n}f\,d\mu \to$ a positive
number.

**General ergodic $X$.** The eigenfunctions of $T$ span $\mathcal{H}_e(X,T) \cong
L^2(G)$ for the maximal **Kronecker factor** $G$. On $X \times X$ with $\tau = T \times
T^2$, the $\tau$-invariant functions are built from a $T$-eigenfunction on the first
coordinate and a $T^2$-eigenfunction with the reciprocal eigenvalue on the second; those
$T^2$-eigenfunctions still come from the same Kronecker factor. Thus the limit of
$\frac1N\sum_n \int h(x)f(T^n x)g(T^{2n}x)\,d\mu$ depends only on the projections
$Q_T h, Q_T f, Q_T g$ onto $\mathcal{H}_e$. For $h=f=g=1_B$, that projection is
$p=E(1_B\mid G)$, a nonnegative function on the Kronecker factor with
$\int p\,dm_G=\mu(B)>0$, so the compact argument gives a positive limit. Hence the
$3$-AP average exists and is positive for every ergodic $X$, recovering Roth's theorem.
The dividing line is **eigenfunctions**:
they are the only obstruction to the weak-mixing computation (the counterexample
$f=\phi$, $g=\phi^{-2}$, $h=\phi$ gives $f\cdot T^n g\cdot T^{2n}h \equiv 1$ while
$\int\phi = 0$), and they are exactly the content of the Kronecker factor.

## General $k$ (structure)

For $k \ge 4$ a single Kronecker split is insufficient: skew-product and group-extension
behavior can be compact relative to a base without being visible as ordinary
eigenfunctions over a point. Relativize:

- **Compact (isometric) extension** $X \to Y$: $X = Y \times G/L$, $T(y,s) =
  (Ty,\gamma(y)s)$; equivalently $\mathcal{H}_g(X/Y,T) = L^2(X)$, where generalized
  eigenfunctions are finite-rank $T$-invariant $Y$-modules ($TH = A(y)H$, $A(y)$ unitary
  matrix). A **distal** system is an isometric tower from the trivial system.
- **Relatively weak-mixing extension**: $\mathcal{H}_g(X/Y,T) = L^2(Y)$, equivalently
  $X \times_Y X$ ergodic.
- **Distal structure theorem**: every ergodic system has a maximal distal
  factor — a tower $Z_0 \to Z_1(=\text{Kronecker}) \to Z_2 \to \dots \to X_{\max}$ of
  isometric extensions — over which $X$ is relatively weak mixing. (Compact
  sub-extensions are manufactured from a non-trivial $T$-invariant symmetric kernel on
  $X \times_Y X$ via the fibrewise self-adjoint Hilbert–Schmidt operators $A_y$, whose
  finite-dimensional eigenspaces are $T$-invariant finite-rank $Y$-modules.)

**Two facts drive the conclusion.**
1. **Characteristic factor is finite-order distal.** The generic limit $\mu^*_k$ of the
   diagonal measure under $T \times \dots \times T^k$ is a conditional product measure
   relative to $Z_{k-2}(X)^k$ ("defined over $Z_{k-2}$"; induction on $k$ via conditional
   product measures). So multiple recurrence for $X$ reduces to multiple recurrence for
   the order-$(k-2)$ distal factor $Z_{k-2}(X)$. (For $k=3$: $Z_1$ = Kronecker.)
2. **Finite-order distal systems are SZ.** Induct on distal order. A strict group
   extension $X = Y \times G$ of an SZ system $Y$ is SZ: with the diagonal subgroup
   $G_\Delta \subset G^k$ ($\lambda_\Delta * \mu^*_k = \mu^*_k$), the set $\Sigma$ of
   $g \in G$ with $\mu^\Delta \prec \lambda_V * \mu^*_k$ for arbitrarily small $V \ni g$
   is a non-empty closed subsemigroup of compact $G$, hence a group containing $e$, giving
   $\mu^\Delta \prec \lambda_V*\mu^*_k$ for $V \ni e$; a uniform-continuity argument on the
   $L^1(G)$-valued fibre map then forces $\mu^\Delta \prec \mu^*_k$, i.e. positivity.

Combining: reduce $X$ to $Z_{k-2}(X)$ (fact 1), which is SZ (fact 2). Therefore every
ergodic system satisfies, for $0 \le f \in L^\infty$, $f \not\equiv 0$,
$$\liminf_{N-M\to\infty}\frac{1}{N-M}\sum_{n=M+1}^{N}\int f\cdot T^n f \cdots T^{(k-1)n}f\,d\mu > 0,$$
hence (ergodic decomposition) so does every system, hence multiple recurrence holds,
hence — by the correspondence principle — Szemerédi's theorem.

## Why it is the right proof

It names the mechanism: progressions are forced because the diagonal averages are
controlled by a finite distal factor, built by iterating compact extensions, while the
weak-mixing part supplies no further obstruction. On that finite structured factor,
almost-periodicity and the group-extension induction force a positive average.
