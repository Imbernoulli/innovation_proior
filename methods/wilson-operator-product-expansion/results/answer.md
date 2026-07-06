# The Operator Product Expansion

## Problem it solves

Current algebra fixes the *equal-time* commutators of the SU(3)×SU(3) currents and yields low-energy theorems, but it is powerless for the questions that turn on the **short-distance behavior of a product of two local operators** (equivalently, the **high-energy** behavior of its Fourier transform): convergence of the Weinberg sum rules, divergences in radiative corrections, $\eta\to3\pi$ and $\pi^0\to2\gamma$, and the Bjorken limit of deep-inelastic scattering. The equal-time commutator lives only on the slice $x_0=y_0$, has $\delta$-function coefficients in the three-vector $\mathbf x-\mathbf y$, can carry infinite (Schwinger-term) c-number constants, and forces an integer power series in $q_0^{-1}$ — too coarse and sometimes the wrong functional form.

## Key idea

Replace the equal-time commutator by an expansion of the **ordinary product** as the four-vector $y\to x$, over a complete set of local operators with c-number coefficient functions, and **fix those coefficients by assuming scale invariance at short distances**, so each local operator carries a renormalization-determined (generally non-integer) **scale dimension**. Two ingredients, neither tied to a Lagrangian or to canonical fields:

1. **The expansion.** For two local fields $A,B$ and a complete, linearly independent, countable set of local fields $\{O_n(x)\}$ ordered by dimension (with $O_0=I$),
$$\boxed{\;A(x)\,B(y)=\sum_n C_n(x-y)\,O_n(x),\qquad y\to x\;}$$
where $C_n(x-y)$ are **c-number functions of the four-vector** $x-y$ — distributions singular on the light cone, of the form $[(x-y)^2-i\epsilon(x_0-y_0)]^{-p}$ (possibly with non-integer $p$ and logarithms), finite away from $x=y$ and carrying **no** infinite constants. The expansion holds in the weak sense (sandwiched between fixed states) for $y$ near $x$; to any finite order in $x-y$ only finitely many terms contribute. It exists for free scalar and spinor theories and for renormalized interacting fields to all orders in perturbation theory, for $T$-products and commutators alike.

2. **Scale covariance fixes the singularities.** Assume short-distance invariance under dilatations $U(s)$ with each field of definite dimension,
$$U^\dagger(s)\,O_n(x)\,U(s)=s^{d(n)}\,O_n(sx).$$
Applying $U(s)$ to the expansion and matching the (linearly independent) operators term by term forces each coefficient to be **homogeneous**:
$$\boxed{\;C_n(s\,(x-y))=s^{-d_A-d_B+d(n)}\,C_n(x-y)\;}$$
i.e. for a scalar coefficient $C_n(x-y)\propto |x-y|^{-d_A-d_B+d(n)}$. The singularity strength of each term is set entirely by $d_A+d_B-d(n)$: the identity ($d=0$) gives the leading singularity, terms get less singular as $d(n)$ grows, and only finitely many are singular at a given order. The dimensions $d(n)$ are **not** the canonical integers but renormalization-shifted real numbers (the field-theory analogue of critical exponents), the correction that makes the coefficient-function rules correct in the strong-coupling regime where canonical counting fails.

## What follows from it

**Equal-time commutator recovered (the $\delta$-reduction).** With the leading commutator coefficient $E_0(z)=E[(-z^2+i\epsilon z_0)^{-p}-(-z^2-i\epsilon z_0)^{-p}]$ (which vanishes for spacelike $z$), and converting to $\delta$-functions of $\mathbf z$ via $\int_{\mathbf z}\delta^3\rho=\rho(0)$, $\int_{\mathbf z}(\nabla\delta^3)\rho=-\nabla\rho(0)$ and a Taylor expansion:
$$E_0(z)=F_0(z_0)\,\delta^3(\mathbf z)+\mathbf F_1(z_0)\cdot\nabla\delta^3(\mathbf z)+\cdots,\quad F_0(z_0)=f_0\,z_0^{-2p+3},\ \mathbf F_1(z_0)=\mathbf f_1\,z_0^{-2p+4}.$$
The equal-time coefficient of $\delta^3(\mathbf x-\mathbf y)\,I$ is
$$F_0(0)=\begin{cases}0,&p<3/2\\ f_0,&p=3/2\\ \infty,&p>3/2\end{cases}$$
($\nabla\nabla\delta^3$ coefficient nonzero for $p\ge5/2$). The Gell-Mann case is $p=3/2$; the Schwinger-term infinities are exactly the $p>3/2$ cases — artifacts of squeezing a four-dimensional singularity onto the three-dimensional slice.

**Free-field check (Wick's theorem).** For a free scalar $\phi$ ($d=1$), $D(x-y)\sim(x-y)^{-2}$:
$$:\!\phi^2(x)\!:\,:\!\phi^2(y)\!:=2\,[D(x-y)]^2\,I+4\,D(x-y)\,:\!\phi(x)\phi(y)\!:+\,:\!\phi^2(x)\phi^2(y)\!:.$$
The leading $I$ term goes as $(x-y)^{-4}=$ power $-d_A-d_B+d_C=-2-2+0$; expanding $:\!\phi(x)\phi(y)\!:=\,:\!\phi^2(x)\!:+(y-x)_\mu:\!\phi(x)\nabla^\mu\phi(x)\!:+\cdots$ gives the $:\!\phi^2\!:$ term at power $-2-2+2=-2$. Each coefficient is homogeneous of the dimension the scaling law predicts.

**Conserved currents are rigid.** If $Q=\int j_0$ generates a symmetry, $[A,Q]=qA$ with $U^\dagger QU=Q$ forces $U^\dagger(s)j_0(x)U(s)=s^3 j_0(sx)$ — a conserved current has dimension $3$. Conservation makes $k_n=\int_{\mathbf z}K_{n0}(z_0,\mathbf z)$ time-independent, hence finite as $z_0\to0$, and $qA=\sum_n k_n O_n$ pins the current's dimension at $3$ even with interactions; likewise $P_\mu$ has dimension $1$, the stress tensor $\theta_{\mu\nu}$ dimension $4$. So **current algebra survives intact** while non-conserved operator dimensions shift. The pion-field multiplet is constrained to $1\le\Delta<4$ ($\Delta<4$ for PCAC, $\Delta\ge1$ from Källén–Lehmann).

**Bjorken limit (the headline application).** For $T_{\alpha\beta}(q)=\langle\alpha|\int d^4x\,e^{iq\cdot x}\,T A(x)B(0)|\beta\rangle$, large $q^2$ makes the integral small-$x$ dominated, so
$$T_{\alpha\beta}(q)\simeq\sum_n R_n(q)\,\langle\alpha|O_n(0)|\beta\rangle,\qquad R_n(q)\sim q^{\,d_A+d_B-d(n)-4}.$$
Because Fourier transform in four dimensions trades the homogeneity degree $-\alpha$ for $\alpha-4$, a **non-integer** $d(n)$ produces a **fractional** power of $q$ — impossible in Bjorken's equal-time-commutator theorem (integer powers of $q_0^{-1}$), but exactly what the Thirring model exhibits. Fractional powers are unimportant when $A,B$ are both currents (dimension $3$) but matter when $A,B$ both belong to the pion-field multiplet (non-integer $\Delta$) and $O_n$ is a current.

## The final artifact

A new language for short-distance physics: talk about **operator-product expansions** of products of operators near the same point instead of equal-time commutators; talk about the **dimension** of an operator instead of how it is built from canonical fields; carry out analyses of divergences in **position space** rather than momentum space. The two boxed equations — the expansion plus the scale-covariance constraint $C_n(s(x-y))=s^{-d_A-d_B+d(n)}C_n(x-y)$ — are the whole content: they fix every short-distance singularity up to constants in terms of the operator dimensions, recover the equal-time commutator and the Bjorken limit as special cases, and let the conserved-current algebra survive while the rest of the operator spectrum is renormalized to non-canonical dimensions.
