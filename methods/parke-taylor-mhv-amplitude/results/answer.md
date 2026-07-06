# The MHV (Parke-Taylor) n-gluon amplitude

## Problem

Tree-level $n$-gluon scattering in $SU(N)$ Yang-Mills is needed for multi-jet
QCD predictions, but the Feynman-diagram count grows faster than factorially
(4, 25, 220, 2485, 34300, 559405, 10525900 for $n = 2,\dots,8$), so brute force
gives no closed form past a handful of legs. Goal: a formula for the squared,
color-summed amplitude valid for all $n$.

## Key idea

Three moves collapse the problem.

1. **Helicity basis + spinor variables.** Compute fixed-helicity amplitudes and
   sum $|A|^2$ incoherently. Write everything in two-component spinors with
   $\langle ij\rangle$, $[ij]$; since $\langle ij\rangle[ji]=s_{ij}$ and
   $|\langle ij\rangle|^2=s_{ij}$, the brackets are complex square roots of the
   invariants and natively encode collinear singularities ($1/\sqrt{s_{ab}}$
   with an azimuthal phase, from the angular-momentum mismatch in a gauge
   collinear limit).

2. **The simplest nonvanishing sector is two-minus.** Supersymmetry Ward
   identities (valid for tree gluon amplitudes since no scalar/fermion runs
   internally) force the all-plus and one-minus amplitudes to vanish. The first
   nonzero configuration — maximal helicity violation that is actually allowed —
   has exactly two negative-helicity gluons (MHV).

3. **Little-group weight + factorization fix the form uniquely.** Color
   decomposition reduces the colored amplitude to a cyclic partial amplitude
   $m(1,\dots,n)$. Little-group scaling forces a fourth-power numerator
   $\langle ij\rangle^4$ on the two negative legs $i,j$; mass dimension and
   cyclic symmetry force an $n$-bracket denominator; counting negative helicities
   shows the MHV partial amplitude has *no* multi-particle poles (a factorization
   channel would need four negative helicities but only three are available), so
   the denominator is the nearest-neighbor cyclic chain. Altarelli-Parisi
   collinear factorization fixes the residues and is satisfied for all $n$.

## Result

The color-ordered MHV partial amplitude, with the two negative-helicity gluons
labeled $i,j$ and the rest positive (all outgoing):

$$\boxed{\,m(1^+,\dots,i^-,\dots,j^-,\dots,n^+) \;=\; i\,g^{\,n-2}\,
\frac{\langle i\,j\rangle^4}{\langle 1\,2\rangle\langle 2\,3\rangle\cdots\langle n\,1\rangle}\,}$$

The full color amplitude is
$\mathcal{M}_n = i\,g^{n-2}\,\langle ij\rangle^4\sum_{P'}
\mathrm{tr}(\lambda^{a_1}\cdots\lambda^{a_n})/(\langle 12\rangle\cdots\langle n1\rangle)$
over the $(n{-}1)!$ non-cyclic orderings. Squared and summed over colors at
leading order in $N$, and over the two helicity sectors $(--+\cdots+)$ and
$(++-\cdots-)$ (using $|\langle ij\rangle|^2=s_{ij}=2p_i\cdot p_j$):

$$\sum |\mathcal{M}_n|^2 \;=\; 2\,g^{\,2n-4}\,N^{\,n-2}(N^2-1)\,
\sum_{i>j} s_{ij}^4 \;\sum_{P'} \frac{1}{s_{12}\,s_{23}\cdots s_{n1}},$$

where the primed sum runs over the $(n{-}1)!$ non-cyclic orderings of the
cyclic denominator and $s_{ij}=2p_i\cdot p_j$.

**Properties.** Correct mass dimension $4-n$ and little-group weight; cyclic and
Bose symmetric; reproduces the known $n=4,5,6$ amplitudes; obeys soft and
collinear (Altarelli-Parisi) factorization in every adjacent pair, for all $n$.
The collinear splitting amplitudes that drop out are
$\mathrm{Split}_-(a^+,b^+;z)=1/(\sqrt{z(1-z)}\,\langle ab\rangle)$ and its
helicity partners — the square roots of the AP $g\to gg$ kernels. Every
multi-particle propagator present in the Feynman diagrams cancels, leaving only
nearest-neighbor brackets.

## Numerical verification

```python
import numpy as np

def random_null(n, rng):
    """n outgoing null four-momenta p=(E, px, py, pz), E=|vec p|."""
    return [np.array([np.linalg.norm(v := rng.normal(size=3)), *v]) for _ in range(n)]

def mink(p, q):                       # (+---) Minkowski product
    return p[0]*q[0] - np.dot(p[1:], q[1:])

def spinors(p):
    """Factor p_{a adot} = lambda_a * lambdatilde_adot (lambdatilde = lambda* up to phase)."""
    E, px, py, pz = p
    M = np.array([[E + pz, px - 1j*py],
                  [px + 1j*py, E - pz]], dtype=complex)
    lam = M[:, 0]/np.sqrt(M[0, 0]) if abs(M[0, 0]) > 1e-9 else M[:, 1]/np.sqrt(M[1, 1])
    a = 0 if abs(lam[0]) >= abs(lam[1]) else 1
    return lam, M[a, :]/lam[a]

class Kinematics:
    def __init__(self, ps):
        self.ps = ps; self.S = [spinors(p) for p in ps]
    def ang(self, i, j):              # <ij>
        li, _ = self.S[i]; lj, _ = self.S[j]
        return li[0]*lj[1] - li[1]*lj[0]
    def sq(self, i, j):               # [ij]
        _, ti = self.S[i]; _, tj = self.S[j]
        return ti[0]*tj[1] - ti[1]*tj[0]
    def s(self, i, j):                # s_ij = 2 p_i . p_j
        return 2*mink(self.ps[i], self.ps[j])

def mhv_partial(K, n, neg):
    """i * <ij>^4 / (<12><23>...<n1>); neg=(i,j) the two negative legs, g set to 1."""
    i, j = neg
    den = 1.0
    for k in range(n):
        den *= K.ang(k, (k + 1) % n)
    return 1j * K.ang(i, j)**4 / den

def mhv_square_via_s(K, n, neg):
    """The squared form as a ratio of invariants: s_ij^4 / (s_12 s_23 ... s_n1)."""
    i, j = neg
    den = 1.0
    for k in range(n):
        den *= K.s(k, (k + 1) % n)
    return K.s(i, j)**4 / den

def collinear_recursion(n, neg, rng, z=0.37):
    """Legs a=(n-2), b=(n-1) made collinear as zP, (1-z)P (both positive helicity);
       expect |m_n|^2 -> |Split(a+,b+)|^2 |m_{n-1}|^2, |Split|^2 = 1/(z(1-z)|<ab>|^2)."""
    base = random_null(n - 1, rng); P = base[-1]; eps = 1e-6
    kick = np.array([0.0, eps, -eps, 0.0])
    pa = z*P + kick; pa[0] = np.linalg.norm(pa[1:])
    pb = (1 - z)*P - kick; pb[0] = np.linalg.norm(pb[1:])
    Kn = Kinematics(base[:-1] + [pa, pb]); a, b = n - 2, n - 1
    Km1 = Kinematics(base)
    split2 = 1.0/(z*(1 - z)*abs(Kn.ang(a, b))**2)
    return abs(mhv_partial(Kn, n, neg))**2, split2*abs(mhv_partial(Km1, n - 1, neg))**2

if __name__ == "__main__":
    rng = np.random.default_rng(2024)
    for n in (5, 6, 7):
        K = Kinematics(random_null(n, rng)); neg = (0, 1)
        br = max(abs(abs(K.ang(a, b)*K.sq(b, a)) - abs(K.s(a, b)))
                 for a in range(n) for b in range(n) if a != b)
        sqid = abs(abs(mhv_partial(K, n, neg))**2 - abs(mhv_square_via_s(K, n, neg)))
        lhs, rhs = collinear_recursion(n, neg, rng)
        print(f"n={n}: |<ab>[ba]|-|s_ab|={br:.1e}  |m|^2-ratio={sqid:.1e}  collinear ratio={lhs/rhs:.4f}")
```

Output: bracket-vs-invariant residual $\sim10^{-15}$, squaring identity holds to
$\sim10^{-15}$, collinear ratio $=1.0000$ at $n=5,6,7$.
