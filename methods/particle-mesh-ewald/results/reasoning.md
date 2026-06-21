I have a solvated protein in a periodic box — call it twenty, thirty thousand
atoms — and every single time step I need the electrostatic energy and the force
on every atom. Millions of steps. The electrostatics is the wall. If I just
truncate the Coulomb interaction at some cutoff the simulation looks plausible
for a while and then quietly goes wrong: water orders artificially at the cutoff
shell, the protein drifts, the ions sit in the wrong places. The 1/r tail
matters, and for a periodic system the honest object is the Ewald lattice sum.
So the question isn't whether to do Ewald, it's how to do it without it eating
the entire compute budget.

Let me write down exactly what Ewald gives me so I know what I'm trying to make
fast. The raw periodic Coulomb sum over a neutral unit cell U with N charges,

  E = ½ Σ′_n Σ_{i,j} q_i q_j / |r_i − r_j + n|,

with n = n₁a₁+n₂a₂+n₃a₃ running over all lattice translations and the prime
dropping i=j at n=0, is only conditionally convergent — its value depends on how
I let the box of images grow and on the surroundings of the crystal. Ewald's
fix, the theta-transform, is to screen each charge with a Gaussian of width set
by a parameter β and split the sum into two absolutely convergent pieces plus
corrections. Writing the pair potential as ψ(r;β) = Φ_dir(r;β) + Φ_rec(r;β),

  Φ_dir(r;β) = Σ_n erfc(β|r+n|)/|r+n|,
  Φ_rec(r;β) = (1/πV) Σ_{m≠0} [exp(−π²m²/β²)/m²] exp(2πi m·r),

where V = a₁·(a₂×a₃) is the cell volume and m = m₁a₁*+m₂a₂*+m₃a₃* runs over the
reciprocal lattice. The total energy comes out as

  E = Σ_{i<j} q_i q_j erfc(βr_ij)/r_ij − (β/√π) Σ_i q_i²
        + ½ Σ_{i,j} q_i q_j Φ_rec(r_j − r_i;β) + J(D).

Four terms. The first is the direct sum of *screened* charges; erfc kills it past
a few Ångström. The second is the self-energy — each charge talks to its own
screening Gaussian and I subtract that off. The fourth, J(D), is the de Leeuw–
Perram–Smith surface term, depending on the cell dipole D and the boundary
conditions; it's an O(N) piece of bookkeeping. The crucial thing I notice is
that ψ depends on β only by an additive constant, so for a neutral cell the
*total* E is independent of β. β doesn't change the physics; it only moves work
between the direct and reciprocal sums. Crank β up and erfc(βr) collapses so the
direct sum, term one, becomes just the minimum-image pairs inside a 9 Å cutoff —
O(N) with the Verlet list I already have. Self and surface are O(N) too.

So everything is cheap except the third term,

  E_rec = ½ Σ_{i,j} q_i q_j Φ_rec(r_j − r_i;β).

That's a sum over all N² ordered pairs, every step. *That* is the O(N²)
bottleneck — and pushing β up to cheapen the direct sum only shovels more weight
into exactly this term. Tuning β to balance the two gets me to N^{3/2} at best.
For thirty thousand atoms, no.

OK. Stare at E_rec. The first instinct in the Ewald literature is to rewrite it
with the structure factor S(m) = Σ_{j} q_j exp(2πi m·r_j):

  E_rec = (1/2πV) Σ_{m≠0} [exp(−π²m²/β²)/m²] S(m) S(−m).

This is a real improvement in framing — the kernel exp(−π²m²/β²)/m² decays fast,
so only a modest shell of m-vectors matters, and the pair sum has collapsed into
|S(m)|². But it hasn't escaped the scaling: each S(m) is itself a sum over all N
charges, and I need it for O(N) reciprocal vectors, so I'm back at O(N²). The
expensive thing is evaluating S(m), the sum Σ_j q_j exp(2πi m·r_j), again and
again for many m. That sum — charges, times complex exponentials of m·r_j,
summed over j — has the exact shape of a Fourier transform of a charge density.
If only the r_j sat on a regular grid, S(m) for all m would be one FFT, O(M log
M) instead of O(N·M). But the atoms are at arbitrary positions, not on a grid.

That's the gap, and it's a familiar one. The plasma and cosmology people hit it
years ago and answered it with particle–mesh: Hockney and Eastwood's P³M takes
the off-grid charges and *assigns* them to a regular mesh with some smooth window
function — nearest-grid-point, cloud-in-cell, triangular-shaped-cloud — then
solves the long-range Poisson problem on the grid with the FFT, and interpolates
the field back to the particles. Spread → FFT → solve → interpolate back, all
O(N log N). The idea fits my problem like a glove in spirit: the reciprocal sum
*is* the long-range Poisson solution for Gaussian sources in periodic boundary
conditions, and the FFT is exactly the tool for a convolution on a periodic grid.

But the P³M assignment windows are low order. They were built for plasmas where
moderate accuracy is fine and the particles are roughly uniform. People have
tried them for macromolecules and the verdict is that high accuracy is hard to
get this way, especially with the lumpy, nonuniform charge distribution of a
protein — and the low-order windows give forces that aren't very smooth, which
is bad news for integrating equations of motion. I don't want a fixed,
low-accuracy assignment. I want a knob that lets me dial the accuracy up as far
as I like, and I want smooth forces. So I can't just lift P³M wholesale. I take
its skeleton — get the charges onto a grid, do the long-range part by FFT — and
I need to redesign the assignment so accuracy is tunable and the result is
continuous.

Where does the off-grid-ness actually bite? It's in the complex exponentials.
Φ_rec(r_j − r_i;β), written in the fractional coordinates f_k = a_k*·r of the
cell, is

  Φ_rec(f₁,f₂,f₃;β) = (1/πV) Σ_{m≠0} [exp(−π²m²/β²)/m²]
                        exp[2πi(m₁f₁ + m₂f₂ + m₃f₃)].

The whole position dependence lives in exp[2πi(m₁f₁+m₂f₂+m₃f₃)], a product of
three one-dimensional complex exponentials exp(2πi m_k f_k). And here's the
thing about Φ_rec: it depends only on the *difference* of fractional coordinates,
and it's the same function at every step and for every pair. So I don't have to
recompute it. Pick grid sizes K₁, K₂, K₃ and, once at the start of the
simulation, evaluate Φ_rec — and its gradient — on the regular grid of fractional
coordinates (l₁/K₁, l₂/K₂, l₃/K₃). Store those arrays. Now Φ_rec is *known on the
grid*, exactly. The only remaining job is to get from any atom's actual
fractional position to a value of Φ_rec, and the natural bridge is interpolation
from the grid points.

So: interpolate. But how, given I just rejected low-order windows? The objects
to interpolate are the smooth complex exponentials exp(2πi m_k f_k) — or
equivalently Φ_rec away from grid points. Smooth periodic functions are exactly
where polynomial interpolation shines: the error of order-q interpolation of a
smooth function on a grid of spacing h shrinks like h^q, geometrically in the
order. So let me use high-order Lagrange interpolation and let the *order* be my
accuracy knob.

Naively that's a trap. High-order *global* polynomial interpolation is
ill-conditioned — Runge's phenomenon, wild oscillations near the ends of the
interval, the interpolant fighting itself. If I just throw a degree-9 polynomial
through ten grid points and evaluate near an endpoint, the error explodes. So I
have to be careful about *where* I evaluate. The escape is to always keep the
evaluation point in the *centre* of the stencil. For a point x I take the
(2p−1)th-order Lagrange interpolation on the local window of 2p grid points
straddling x, arranged so x sits in the central interval. Centred, local,
fixed-width — none of the endpoint pathology.

Let me build the weights cleanly. The barycentric form of Lagrange
interpolation on the grid points {k/K} is numerically the right way to write it.
For a real argument x, let [x] be the floor, and define the integer offset that
locates the left edge of the centred stencil,

  k_{p,K}(x) = [Kx] − p + 1.

On the canonical window, define for 0 ≤ k ≤ 2p−1

  φ_{p,K}(x,k) = [ (−1)^k C(2p−1,k) · 1/(x − k/K) ]
                 ───────────────────────────────────────
                 Σ_{l=0}^{2p−1} (−1)^l C(2p−1,l) · 1/(x − l/K),

with φ = 0 otherwise — this is exactly the barycentric weight of (2p−1)th-order
Lagrange interpolation through the points k/K, k = 0,…,2p−1, where C(2p−1,k) =
(2p−1 choose k). Then shift the window to centre on x:

  θ_{p,K}(x,k) = φ_{p,K}[ x − k_{p,K}(x), k − k_{p,K}(x) ].

Three properties I need and can check. First, θ_{p,K}(x,k) is nonzero for only
2p values of k — local stencil, cheap. Second, Σ_k θ_{p,K}(x,k) = 1: the weights
are a partition of unity, which is what makes it reproduce constants and
conserve charge. Third — and this is the property I was chasing — θ_{p,K}(k/K,k)
= 1 and θ_{p,K}(l/K,k) = 0 for l ≠ k, so as a function of x each θ is continuous
(and because x is always pinned to the central interval, there's no
high-order ill-conditioning). Continuity of θ in x is what will make the
interpolated energy and forces continuous functions of the atomic positions —
the property the cell-multipole methods can't offer, because an atom crossing a
hierarchy boundary makes them jump.

Now interpolate Φ_rec at a pair. Atom j has fractional coords f_{j1},f_{j2},f_{j3}
and atom i has f_{i1},f_{i2},f_{i3}. The argument of Φ_rec is the difference, and
the exponential factorizes over the three axes, so the interpolation factorizes
too. Approximate Φ_rec at the (i,j) separation by interpolating from the
precomputed grid array:

  Φ̂_{rec,p}(f_{j1}−f_{i1}, … ;β)
     = Σ_{k₁,k₂,k₃} θ_{p,K₁}(f_{j1},k₁) θ_{p,K₂}(f_{j2},k₂) θ_{p,K₃}(f_{j3},k₃)
       · Σ_{l₁,l₂,l₃} θ_{p,K₁}(f_{i1},l₁) θ_{p,K₂}(f_{i2},l₂) θ_{p,K₃}(f_{i3},l₃)
       · Φ_rec((l₁−k₁)/K₁, (l₂−k₂)/K₂, (l₃−k₃)/K₃),

using Φ_rec(−r) = Φ_rec(r). Each atom contributes a little cloud of 2p weights
per axis onto the grid; the two clouds are tied together through the precomputed
Φ_rec evaluated at the difference of grid indices.

Stare at the structure of that double sum. Atom j's weights touch grid index
(k₁,k₂,k₃); atom i's weights touch (l₁,l₂,l₃); and they're coupled only through
Φ_rec evaluated at the *difference* (l−k). That is the signature of a
convolution. Let me make it explicit by defining a single gridded charge array
that absorbs all the atoms' interpolation weights at once:

  Q(l₁/K₁, l₂/K₂, l₃/K₃)
     = Σ_{j=1}^N q_j θ_{p,K₁}(f_{j1},l₁) θ_{p,K₂}(f_{j2},l₂) θ_{p,K₃}(f_{j3},l₃).

This is just "spread every charge onto the grid through its 2p×2p×2p weight
cloud" — building Q costs O((2p)²·N), a few hundred grid touches per atom,
linear in N. Now the total interpolated reciprocal energy, summing the pairwise
Φ̂ over all i,j with the ½, collapses into

  E_rec ≈ ½ Σ_{grid points} Q · (Φ_rec * Q),

where Φ_rec * Q is the discrete convolution of the precomputed grid array Φ_rec
with the charge grid Q. Per atom this reads

  Ê_{rec,p}(i) = ½ Σ_{k₁,k₂,k₃} q_i θ_{p,K₁}(f_{i1},k₁) θ_{p,K₂}(f_{i2},k₂)
                   θ_{p,K₃}(f_{i3},k₃) · (Φ_rec * Q)(k₁/K₁, k₂/K₂, k₃/K₃),

i.e. spread the charges to make Q, convolve with the stored Φ_rec to get a
potential on the grid, then gather that potential back to each atom through the
*same* interpolation weights, weighted by q_i. The O(N²) pair sum has turned into
a convolution on a grid of M = K₁K₂K₃ points.

And a convolution on a periodic grid is the one thing the FFT does for free: a
convolution in real space is a pointwise product in Fourier space. Transform Q,
multiply by the transform of Φ_rec, transform back. The FFT is O(M log M).
Concretely: at the start of the simulation I FFT the precomputed Φ_rec array
(and the three gradient-component arrays) once, and store them. Each step I build
Q as a complex array, FFT it, multiply the transformed Q pointwise by the
precomputed transformed Φ_rec at every grid point, inverse-FFT to get Φ_rec*Q on
the grid, and gather. The convolution costs O(K₁K₂K₃ log(K₁K₂K₃)); building Q and
reading off the per-atom energies costs O((2p)²·N). The reciprocal sum is done.

Forces. I need −∂E_rec/∂r_i, smoothly. Because the only position dependence in
Q and in the gather step is through the θ weights, and θ_{p,K}(x,k) is continuous
in x, I can differentiate term by term. The clean way is to interpolate the
*gradient* of Φ_rec the same way I interpolate Φ_rec: I precompute Φ_rec's three
Cartesian gradient components on the grid alongside Φ_rec, transform them once,
and the convolution machinery delivers the gradient field on the grid; gathering
that through the atom's weights gives the force. Pack Φ_rec and its three
gradient components into two complex arrays, transform once, and every step do
the multiply-and-back-transform on the pair. Interpolating energy and forces
symmetrically this way has a bonus: the reciprocal forces sum to zero to machine
precision, so the scheme obeys Newton's third law exactly even though it's an
approximation.

Now the honest question: how good is this, and does it really scale? I asserted
high-order Lagrange interpolation of a smooth function is accurate — let me
bound the error and see what controls it, because that's what tells me how K and
p have to grow with the system. The thing I'm actually interpolating per axis is
the complex exponential exp(2πi m x). For (2p−1)th-order Lagrange interpolation
on the grid {k/K}, the standard remainder estimate for a smooth function gives,
for all x,

  | exp(2πi m x) − Σ_{k∈Z} exp(2πi m k/K) θ_{p,K}(x,k) | < 2 C(2p,p) (m/4K)^{2p},

where C(2p,p) = (2p)!/(p!)². The interpolation error of a band-limited
exponential of frequency m on a grid of spacing 1/K is controlled by (m/K)^{2p}
— geometric in the order p, and shrinking with finer grid. Good: p is exactly
the accuracy knob I wanted.

Feed that into Φ_rec. The interpolated Φ̂_{rec,p} differs from the exact Φ_rec by
the interpolation error applied inside the reciprocal sum. Applying the
one-dimensional bound per axis and summing,

  | Φ̂_{rec,p} − Φ_{rec,p} | < 4 C(2p,p) (1/πV) Σ_{m≠0} [exp(−π²m²/β²)/m²]
        · [ (m₁/4K₁)^{2p} + (m₂/4K₂)^{2p} + (m₃/4K₃)^{2p} ],

pointwise. The Gaussian kernel exp(−π²m²/β²) tames the high-m terms; the
interpolation factor (m_k/K_k)^{2p} grows with m but the Gaussian wins. To see
the dependence on the grid sizes cleanly, replace the sum by an integral. Change
variables to x_k = a*_{1k}m₁ + a*_{2k}m₂ + a*_{3k}m₃ (so m_k = a_{k1}x₁ +
a_{k2}x₂ + a_{k3}x₃), go to spherical coordinates in x, and use Cauchy–Schwarz on
the components m_k together with the standard moments of a Gaussian. Working that
through, the right-hand side is bounded above by

  | Φ̂_{rec,p} − Φ_{rec,p} | ≤ 8 C(2p,p) ((2p)!/p!) (β/√π) (β/8π)^{2p}
        · [ (a₁/K₁)^{2p} + (a₂/K₂)^{2p} + (a₃/K₃)^{2p} ].

This is the result I wanted to read off. The error is governed by (a_k/K_k)^{2p}
— the *grid spacing* a_k/K_k raised to twice the order. So if I fix the grid
spacing a_k/K_k to something below one Ångström and then choose p large enough,
the error goes to zero geometrically, for any tolerance I name. And here's the
scaling argument: holding the spacing a_k/K_k fixed means K₁K₂K₃ grows in
proportion to the cell volume a₁a₂a₃, which grows in proportion to the number of
atoms N. So M ≈ K₁K₂K₃ is O(N). The convolution is O(M log M) = O(N log N), and
the spread/gather is O((2p)²N) = O(N). For any fixed accuracy, the whole
reciprocal evaluation is O(N log N). That's the win — and it's a *proof*, not a
hope: accuracy and cost are decoupled, accuracy set by p, cost set by N.

A few choices I should pin down so the thing actually runs. The screening β: I
want erfc(βr)/r negligible beyond my 9 Å cutoff so the direct sum is truly
minimum-image and O(N); a value around β = 0.386 Å⁻¹ does that for the crystals
I care about, and the total energy doesn't care about the exact value since it's
β-invariant. The grid spacing a_k/K_k I take around 0.5–1 Å, fine enough that the
(a_k/K_k)^{2p} factor is small. The order p: the error bound says larger p buys
accuracy geometrically, but each atom touches (2p)³ grid points so the
spread/gather cost grows as p³ — there's a sweet spot to find. For MD I only need
the force error down around 10⁻⁴ or so, and the bound says a modest order on a
sub-Ångström grid should get there cheaply, with a couple of extra orders
available whenever I want to push toward near-exact; that's exactly the
accuracy-on-demand I was after, and what I'd want to confirm by measuring the rms
force error against the exact pair sum as I turn p up. The grid costs more memory
than a plain neighbor-list method, but memory is cheap and getting cheaper, so
I'll spend it.

Let me say the whole chain back to myself in one breath. The exact periodic
electrostatics is the Ewald sum; with β large the direct, self, and surface terms
are all O(N), and the only expensive piece is the reciprocal sum ½ Σ_{ij} q_i q_j
Φ_rec(r_j−r_i), which is O(N²). Φ_rec depends only on the difference of fractional
coordinates and is fixed, so I precompute it (and its gradient) once on a regular
grid. To get Φ_rec at arbitrary atomic positions I interpolate with centred,
high-order Lagrange weights θ that are continuous in position and tunable in
accuracy through the order p. Because the weights factorize over the three axes,
the interpolated pair sum collapses into a discrete convolution of a gridded
charge array Q (spread the charges through their weight clouds) with the
precomputed Φ_rec grid — and a periodic convolution is an FFT, O(M log M).
Spread, FFT, multiply by the transformed Φ_rec, inverse-FFT, gather; do the same
with the gradient arrays for the forces. The interpolation error bound goes as
(grid-spacing)^{2p}, so fixing the spacing and choosing p makes the error as
small as I like while keeping the grid size — hence the cost — proportional to N.
The reciprocal sum has gone from O(N²) to O(N log N) at any accuracy, with smooth
forces. That's the method.

One simplification for the implementation. By Parseval, ½ Σ_grid Q·(Φ_rec*Q) is
exactly (1/2πV) Σ_{m≠0} [exp(−π²m²/β²)/m²] |F(Q)(m)|², where F(Q) is the FFT of
the gridded charge array — i.e. F(Q)(m) is the *approximate* structure factor
S(m), and the reciprocal kernel just multiplies its squared magnitude in Fourier
space. So I don't even need to precompute Φ_rec on the grid and FFT it; I can
apply the reciprocal kernel directly to F(Q). The energy is one FFT of Q,
multiply by the kernel, sum. For the forces I still want the gradient field on
the grid, so I form F(Q) times the kernel, multiply by i·2π·m to differentiate,
inverse-FFT to get the three Cartesian gradient components, and gather them
through the same weights — the convolution and the structure-factor pictures are
the same computation seen two ways.

Here is the reciprocal-space evaluator, the spread → FFT → influence → inverse-
FFT → gather pipeline, in the canonical form.

```python
import numpy as np
from math import comb

def lagrange_weights(u, p):
    """Centred (2p-1)th-order Lagrange interpolation weights theta_{p,K}(x,k)
    for a point at scaled fractional coordinate u = K*f, on its 2p-point window.
    Returns (left_index, weights[2p]). Barycentric form; x sits in the central
    interval so high-order interpolation stays well conditioned."""
    floor_u = int(np.floor(u))
    left = floor_u - p + 1                       # k_{p,K}(x) = [Ku] - p + 1
    nodes = np.arange(2 * p)                      # window nodes 0..2p-1
    x = u - left                                  # x relative to window start
    bary = np.array([(-1) ** k * comb(2 * p - 1, k) for k in nodes], float)
    diff = x - nodes
    on_node = np.isclose(diff, 0.0)
    if on_node.any():                            # x coincides with a node
        w = np.where(on_node, 1.0, 0.0)
    else:
        terms = bary / diff
        w = terms / terms.sum()                  # sum_k theta = 1 (partition of unity)
    return left, w                               # nonzero on 2p nodes only

class ReciprocalSpaceEvaluator:
    def __init__(self, cell, beta, grid_dims, order):
        self.cell = np.asarray(cell, float)      # 3x3 lattice vectors (columns)
        self.a_star = np.linalg.inv(self.cell).T # rows a_k*: f = a_star @ r
        self.V = abs(np.linalg.det(self.cell))
        self.beta = beta
        self.K = tuple(grid_dims)                # (K1, K2, K3)
        self.p = order
        self.M = self.K[0] * self.K[1] * self.K[2]
        self.setup()

    def setup(self):
        # Precompute, once: the reciprocal Ewald kernel exp(-pi^2|m|^2/beta^2)
        # /(pi V |m|^2) on the reciprocal grid (the influence function in k-space).
        K1, K2, K3 = self.K
        m1 = np.fft.fftfreq(K1, d=1.0 / K1)      # integer reciprocal indices
        m2 = np.fft.fftfreq(K2, d=1.0 / K2)
        m3 = np.fft.fftfreq(K3, d=1.0 / K3)
        M1, M2, M3 = np.meshgrid(m1, m2, m3, indexing="ij")
        A = self.a_star                          # m = m1 a1* + m2 a2* + m3 a3*
        mx = M1 * A[0, 0] + M2 * A[1, 0] + M3 * A[2, 0]
        my = M1 * A[0, 1] + M2 * A[1, 1] + M3 * A[2, 1]
        mz = M1 * A[0, 2] + M2 * A[1, 2] + M3 * A[2, 2]
        m2sq = mx ** 2 + my ** 2 + mz ** 2
        kern = np.zeros_like(m2sq)
        nz = m2sq > 0
        kern[nz] = np.exp(-(np.pi ** 2) * m2sq[nz] / self.beta ** 2) \
                   / (np.pi * self.V * m2sq[nz])
        self.kern = kern                         # Fourier-space reciprocal kernel
        self.m_vec = (mx, my, mz)                # for the analytic gradient (forces)

    def _spread(self, positions, charges):
        # Gridded charge array Q: each charge spread through its 2p x 2p x 2p
        # centred-Lagrange weight cloud. O((2p)^2 N).
        K1, K2, K3 = self.K
        Q = np.zeros((K1, K2, K3), dtype=complex)
        info = []
        for r, q in zip(positions, charges):
            f = self.a_star @ r                  # fractional coords
            f = f - np.floor(f)                  # wrap into [0,1)
            l1, w1 = lagrange_weights(f[0] * K1, self.p)
            l2, w2 = lagrange_weights(f[1] * K2, self.p)
            l3, w3 = lagrange_weights(f[2] * K3, self.p)
            i1 = (l1 + np.arange(2 * self.p)) % K1
            i2 = (l2 + np.arange(2 * self.p)) % K2
            i3 = (l3 + np.arange(2 * self.p)) % K3
            Q[np.ix_(i1, i2, i3)] += q * w1[:, None, None] * w2[None, :, None] * w3[None, None, :]
            info.append((i1, i2, i3, w1, w2, w3, q))
        return Q, info

    def energy_and_forces(self, positions, charges):
        Q, info = self._spread(positions, charges)
        Qhat = np.fft.fftn(Q)                     # F(Q)(m) = approximate structure factor S(m)
        # E_rec = 1/2 sum Q.(Phi_rec*Q) = 1/(2 pi V) sum kernel * |S(m)|^2  (Parseval)
        E_rec = 0.5 * np.sum(self.kern * np.abs(Qhat) ** 2)
        # gradient field on the grid: F(Phi_rec*Q) = kernel * Qhat, then i*2*pi*m to differentiate
        pot_hat = self.kern * Qhat
        mx, my, mz = self.m_vec                   # *M restores numpy's ifft normalization
        gx = np.fft.ifftn(pot_hat * (1j * 2 * np.pi * mx)).real * self.M
        gy = np.fft.ifftn(pot_hat * (1j * 2 * np.pi * my)).real * self.M
        gz = np.fft.ifftn(pot_hat * (1j * 2 * np.pi * mz)).real * self.M
        F = np.zeros_like(positions)
        for k, (i1, i2, i3, w1, w2, w3, q) in enumerate(info):
            wc = w1[:, None, None] * w2[None, :, None] * w3[None, None, :]
            b = np.ix_(i1, i2, i3)               # gather gradient through same weights
            F[k] = -q * np.array([np.sum(wc * gx[b]),
                                  np.sum(wc * gy[b]),
                                  np.sum(wc * gz[b])])  # force = -grad of E_rec
        return E_rec, F
```
