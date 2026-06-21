# The no-go theorem on combining spacetime and internal symmetries

## The problem

In a relativistic quantum theory the symmetry generators fall into two families that never seem to talk to each other: the **Poincaré generators** — the four-momentum $P_\mu$ (translations) and $J_{\mu\nu}$ (Lorentz rotations and boosts) — and the **internal generators** (isospin, $SU(3)$-flavour) that act only on particle-type labels and commute with all of Poincaré. Because internal charges commute with $P_\mu$, every member of an internal multiplet has the same mass. Two hopes drove the search for a non-trivial mixing of the two families: that a generator with $[P_\mu, I]\neq 0$ could *explain* the observed mass splittings (e.g. $m_p\neq m_n$), and that a fully relativistic spin-flavour group ($SU(6)$ completed to contain the Lorentz group) could place different-spin, different-mass particles in one irreducible multiplet. Every such construction failed.

The question this answers: **for a relativistic theory with a non-trivial scattering matrix, what is the most general Lie algebra of symmetry generators?**

## The key idea

Stop arguing inside abstract group theory (where earlier no-go results needed unphysical hypotheses — finite-parameter groups, normality of translations — and saw only the one-particle spectrum, so loophole groups escaped). Instead demand that the symmetry **commute with a non-trivial, analytic S-matrix** and use the *multiparticle* scattering content directly.

The physics in one line: a symmetry generator acts **additively** on multiparticle states, so any conserved charge carrying uncontracted Lorentz indices (a tensor charge beyond $P_\mu$ and $J_{\mu\nu}$) would impose conservation laws *on top of* energy–momentum conservation, freezing elastic scattering to isolated angles. Analyticity of the amplitude in the scattering angle then forces it to vanish identically — contradicting the premise that scattering occurs. Hence every conserved generator beyond Poincaré must be a Lorentz scalar: an internal symmetry. The algebra splits as Poincaré $\oplus$ internal.

## The theorem

Let $G$ be a connected symmetry group of the $S$-matrix, i.e. a group of unitary operators that (i) map one-particle states to one-particle states, (ii) act on many-particle states as tensor products (additive generators), and (iii) commute with $S$. Assume:

1. **(Lorentz invariance)** $G$ contains a subgroup locally isomorphic to the Poincaré group.
2. **(Particle-finiteness)** All particle types are positive-energy Poincaré representations, and below any finite mass there are only finitely many particle types.
3. **(Weak elastic analyticity)** Elastic-scattering amplitudes are analytic in $s$ and $t$ in a neighborhood of the physical region (away from normal thresholds).
4. **(Occurrence of scattering)** Any two one-particle momentum eigenstates scatter into something, except perhaps at isolated values of $s$ (so $S\neq 1$).
5. **(Technical)** The generators of $G$, as integral operators in momentum space, have distribution kernels.

**Then $G$ is locally isomorphic to the direct product of the Poincaré group and an internal symmetry group**, and (if there are finitely many particle types) the internal group has compact closure:
$$
\boxed{\;\mathfrak g \;=\; \mathcal P \,\oplus\, \mathfrak g_{\rm internal},\qquad [\,\mathcal P,\ \mathfrak g_{\rm internal}\,]=0,\quad \mathfrak g_{\rm internal}\ \text{compact}.\;}
$$
Spacetime and internal symmetries cannot mix non-trivially.

## The proof

Split the generators into $\mathfrak B$ (those commuting with $P_\mu$) and $\mathfrak A$ (those that do not).

### Intuition (kinematic over-constraint)

Suppose a conserved symmetric traceless tensor charge $Q_{\mu\nu}$, not built from $J$ or $\eta$. Lorentz covariance on a one-particle state forces $\langle p|Q_{\mu\nu}|p\rangle \propto p_\mu p_\nu - \tfrac14\eta_{\mu\nu}p^2$. Additivity gives, on $|p_1,p_2\rangle$, the value $(p_1)_\mu(p_1)_\nu+(p_2)_\mu(p_2)_\nu$ (trace pieces fixed by masses). For elastic $p_1,p_2\to q_1,q_2$, conservation of $P$ and of $Q$ read
$$
p_1+p_2=q_1+q_2,\qquad (p_1)_\mu(p_1)_\nu+(p_2)_\mu(p_2)_\nu=(q_1)_\mu(q_1)_\nu+(q_2)_\mu(q_2)_\nu.
$$
Writing $q_1=p_1+a,\ q_2=p_2-a$ and substituting, the quadratic pieces cancel and
$$
a_\mu(p_1-p_2)_\nu + a_\nu(p_1-p_2)_\mu + 2a_\mu a_\nu = 0,
$$
which for generic momenta forces $a_\mu=0$: only forward scattering. With analyticity, an amplitude nonzero only at isolated angles vanishes identically — contradicting assumption 4.

### Part 1 — generators commuting with $P_\mu$ ($\mathfrak B$) are internal

A $B_\alpha\in\mathfrak B$ cannot move momentum, so on one-particle states it acts as a $p$-dependent matrix $b_\alpha(p)$ on the (finite, by assumption 2) discrete indices, Hermitian (unitary symmetry), and $b_\alpha(p,q)=b_\alpha(p)\otimes 1 + 1\otimes b_\alpha(q)$. The $B_\alpha$ and the $b_\alpha(p)$ share structure constants, so $B_\alpha\mapsto b_\alpha(p)$ is a homomorphism.

**Isomorphism via similarity transforms.** From $[B_\alpha,S]=0$ on elastic $p,q\to p',q'$,
$$
b_\alpha(p',q')\,M = M\,b_\alpha(p,q)\ \Longrightarrow\ b_\alpha(p',q') = M\,b_\alpha(p,q)\,M^{-1},
$$
with $M$ the amplitude, invertible at almost all momenta (assumptions 3, 4). So $c^\alpha b_\alpha(p,q)=0$ propagates to all connected $(p',q')$. The two-particle structure then gives $c^\alpha b_\alpha(p')\otimes 1 = -1\otimes c^\alpha b_\alpha(q')$, forcing each side $\propto\mathbb 1$ — a residual trace piece survives.

**Killing the trace.** $\mathrm{Tr}\,b_\alpha(p,q)=N(m_q)\,\mathrm{tr}\,b_\alpha(p)+N(m_p)\,\mathrm{tr}\,b_\alpha(q)$ is similarity-invariant, hence an additive conserved charge, hence $\mathrm{tr}\,b_\alpha(p)/N(m_p)=a_\alpha^\mu p_\mu$ (linear in $p$, no constant). Define $B^\#_\alpha\equiv B_\alpha - a_\alpha^\mu P_\mu$, still in $\mathfrak B$, with **traceless** $b^\#_\alpha(p)$. Now $c^\alpha b^\#_\alpha(p')\propto\mathbb 1$ and traceless $\Rightarrow c^\alpha b^\#_\alpha(p')=0$.

**Chaining off momentum conservation.** From $c^\alpha b^\#_\alpha(p,q)=0$ one gets vanishing at $p,q,p',q'$; then scattering $p,q'\to k,(p+q'-k)$ (total momentum $p+q'$) makes the outgoing $\mathbf k$ free, so $c^\alpha b^\#_\alpha(k)=0$ for almost all on-shell $k$, then all $k$ (isolated bad points excluded by analyticity, other mass shells reached by inelastic processes + assumption 2). Hence $c^\alpha B^\#_\alpha=0$: the map is an **isomorphism**. Consequently $\mathfrak B^\#$ is finite-dimensional, and being a Lie algebra of finite Hermitian matrices it is **compact semisimple $\oplus$ $u(1)$'s**.

**Commuting with Lorentz.** For each $u(1)$ generator $B^\#_i$: Jacobi $\Rightarrow [J,B^\#_i]\in\mathfrak B^\#$; since $B^\#_i$ is central in $\mathfrak B^\#$, $[B^\#_i,[J,B^\#_i]]=0$, and its two-particle expectation value $0=2\sum(\sigma'-\sigma)|(b^\#_i)_{\cdot\cdot}|^2$ forces $b^\#_i$ to commute with $J$. For the semisimple part, $U(\Lambda)B_aU(\Lambda)^{-1}=D(\Lambda)_a{}^bB_b$ is a finite-dimensional **unitary** representation of the **non-compact** Lorentz group, which must be trivial, so $[B_a,U(\Lambda)]=0$. Thus all of $\mathfrak B^\#$ commutes with $P_\mu$ and $J_{\mu\nu}$: internal.

### Part 2 — generators not commuting with $P_\mu$ ($\mathfrak A$) are Lorentz + internal

The kernel $A_\alpha(p',p)$ vanishes off shell. Smearing $A_\alpha^f=\int d^4x\,e^{iP\cdot x}A_\alpha e^{-iP\cdot x}f(x)$ with $\tilde f$ peaked at $\Delta\neq 0$ shifts momentum by $\Delta$; on generic scattering this kills three legs but not the fourth, forbidding generic scattering — contradiction. So $A_\alpha(p',p)$ is supported at $p'=p$, hence a finite sum of $\delta^4(p'-p)$ derivatives up to order $D_\alpha$ (assumption 5 bounds $D_\alpha$).

Take the $D$-fold commutator $B_\alpha^{\mu_1\cdots\mu_D}=[P^{\mu_1},[\cdots[P^{\mu_D},A_\alpha]\cdots]]$: each $[\,\cdot\,,P]$ pulls a factor $(p'-p)$, and $D+1$ such factors against $\le D$ derivatives of $\delta$ vanish at $p'=p$, so $B_\alpha^{\mu_1\cdots\mu_D}\in\mathfrak B$, giving $b_\alpha^{\mu_1\cdots\mu_D}(p)=(b^\#_\alpha)^{\mu_1\cdots\mu_D}+a_\alpha^{\mu\mu_1\cdots\mu_D}p_\mu$. It is **symmetric** in $\mu_1\dots\mu_D$ (the $P$'s commute). The mass-shell condition $[P^2,A_\alpha]=0$ gives, for $D\ge 1$, $p_{\mu_1}b_\alpha^{\mu_1\cdots\mu_D}(p)=0$, hence $(b^\#_\alpha)^{\mu_1\cdots\mu_D}=0$ and **antisymmetry in the first two indices** $a_\alpha^{\mu\mu_1\cdots}= -a_\alpha^{\mu_1\mu\cdots}$.

- $D=0$: $A_\alpha=B_\alpha\in\mathfrak B$ (nothing new).
- $D\ge 2$: antisymmetry in $(\mu,\mu_1)$ + symmetry in $(\mu_1,\dots,\mu_D)$ forces, by index chasing,
$$
a^{\mu\mu_1\mu_2\cdots}=-a^{\mu_1\mu\mu_2\cdots}=-a^{\mu_1\mu_2\mu\cdots}=+a^{\mu_2\mu_1\mu\cdots}=+a^{\mu_2\mu\mu_1\cdots}=-a^{\mu\mu_2\mu_1\cdots}=-a^{\mu\mu_1\mu_2\cdots},
$$
so $a=-a=0$: no generators of order $\ge 2$.
- $D=1$: $a_\alpha^{\mu\nu}=-a_\alpha^{\nu\mu}$ and $[P^\mu,A_\alpha]=a_\alpha^{\mu\nu}P_\nu$. Comparing with $[P^\mu,J^{\rho\sigma}]=i(\eta^{\mu\sigma}P^\rho-\eta^{\mu\rho}P^\sigma)$ gives $[P^\mu,-\tfrac{i}{2}a_\alpha^{\rho\sigma}J_{\rho\sigma}]=a_\alpha^{\mu\nu}P_\nu$, so $A_\alpha+\tfrac{i}{2}a_\alpha^{\rho\sigma}J_{\rho\sigma}\in\mathfrak B$, i.e.
$$
A_\alpha = -\tfrac{i}{2}\,a_\alpha^{\mu\nu}J_{\mu\nu} + B_\alpha .
$$

Every generator is therefore a translation, a Lorentz transformation, or an internal symmetry, and the internal generators commute with all of Poincaré. $\blacksquare$

## Where the proof leans (loopholes)

- **Massless particles** ($p^2=0$): the mass-shell power-matching that gave antisymmetry for $D\ge 2$ collapses, so the conformal generators (dilatation $D$, special conformal $K_\mu$) *can* mix with Poincaré. The theorem correctly does not forbid conformal symmetry of massless theories.
- **$1+1$ dimensions**: only forward/backward scattering, so the scattering angle is not an analytic variable — assumption 3 is empty and integrable models can carry infinitely many higher conserved charges.
- **Symmetries invisible to $S$**: spontaneously broken and discrete symmetries are outside the scope (they don't act on $S$ in the required way).
- **The Lie bracket itself**: every step manipulates *commutators* of bosonic Hermitian charges forming a Lie algebra. Charges that close under *anticommutators* — spinorial generators of a graded (super-)algebra — are not even statements in this proof, and so are not excluded. This is the one door left open.
