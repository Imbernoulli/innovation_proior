The problem I set out to solve is to look inside a large, dense, optically opaque object — a sealed cargo container, a truck, a shielded cask — and decide whether it conceals a lump of high-$Z$ material (uranium, plutonium, tungsten, lead), without opening it, without an artificial radiation source, and in an operationally useful time. The difficulty is brutal in its symmetry: the object I am hunting is exactly the object that defeats my probe. X-rays and gamma rays die in a few centimetres of steel, so a shielded actinide core is a black hole to them. Active interrogation with a high-energy beam penetrates better, but it needs an accelerator, it doses the cargo, and it imposes a one-sided source/detector geometry that a closed box does not give me; and like every absorption method it returns a 2D line-integral, not a 3D map, and it is only weakly $Z$-discriminating. The historical passive option — cosmic-ray absorption radiography, as George did in 1955 over a tunnel and Alvarez did under the Pyramid of Chephren — counts the muons that survive along each line of sight. But that survivor count is a line integral of areal density and nothing more: no depth, so no 3D localization; weak $Z$-sensitivity, so a thick slab of water impersonates a thin slab of lead; and it needs enormous exposures or a mountain of overburden to build a statistically significant deficit. None of these meets all four requirements at once — genuinely penetrating, strongly $Z$-sensitive, 3D-localizing, and passive-and-fast.

What is freely available everywhere on Earth's surface is a steady drizzle of cosmic-ray muons — heavy charged leptons (about 207 electron masses, mean life $2.2\,\mu$s), the secondaries of hadronic cascades through $\pi \to \mu\,\nu$, with mean energy 3–4 GeV and a $\cos^2\theta$ zenith dependence, arriving at roughly $1\ \mathrm{cm}^{-2}\,\mathrm{min}^{-1}$. A few-GeV muon loses energy only by ionization, about $2\ \mathrm{MeV}$ per $\mathrm{g\,cm}^{-2}$, so it sails through metres of rock or iron. The flux rate is a gift twice over: low enough that I can catch and reconstruct muons one at a time — no pileup, every muon its own little experiment — yet high enough to accumulate real statistics over minutes-to-hours. The channel I exploit is not energy loss but multiple Coulomb scattering. As a muon crosses material it is kicked many times by elastic Rutherford scattering off nuclei, $d\sigma/d\Omega \sim Z^2/\sin^4(\theta/2)$, so the kicks are mostly tiny-angle, there are very many of them, and the rate scales as $Z^2$. By the central limit theorem the net projected deflection is zero-mean Gaussian, with width given by the Highland / Lynch–Dahl form

$$\theta_0 = \frac{13.6\ \mathrm{MeV}}{\beta c\,p}\,z\,\sqrt{\frac{L}{X_0}}\,\Big[1 + 0.038\,\ln\!\big(L z^2/(X_0\beta^2)\big)\Big] \approx \frac{15\ \mathrm{MeV}}{p}\,\sqrt{\frac{L}{X_0}}\quad(\beta\approx 1),$$

where $p$ is the muon momentum, $z=1$ its charge, $L$ the thickness, and $X_0$ the radiation length. The whole $Z$ dependence hides in $X_0$, and $X_0$ plummets with atomic number — about 36 cm in water, 11 cm in concrete, 1.8 cm in iron, 0.56 cm in lead, 0.32 cm in uranium — so per centimetre, high-$Z$ material scatters muons enormously more. For a 3 GeV muon through 10 cm the projected-angle RMS is about 2.6 mrad in water, 12 in iron, 21 in lead, 28 in uranium: an order of magnitude of contrast between ordinary cargo and a high-$Z$ core, and exactly the discriminant absorption lacks. And it is practical: a milliradian deflection is read straight off position, by fitting an incoming straight track above the object and an outgoing straight track below it (each tracking station has enough position planes to fit a line in two projected views), so millimetre positions over a metre lever arm give milliradian angles with no magnetic spectrometer and no energy measurement.

I propose muon scattering tomography, with a fast non-iterative reconstructor I call PoCA (Point of Closest Approach) and two statistical upgrades, MLS and MLSD. First I have to fix the quantity I am imaging, because the angle itself is not a material property — it also depends on momentum and thickness. Squaring the Highland width gives $\theta_0^2 = (15/p)^2\,(1/X_0)\,L$: thickness enters linearly, momentum as $1/p^2$, material as $1/X_0$. Since tracking alone does not measure each muon's momentum, I fix a nominal $p_0 \approx 3$ GeV and define the scattering density

$$\lambda := \Big(\frac{15}{p_0}\Big)^2\frac{1}{X_0} = \frac{\theta_0^2}{L},$$

the mean-square projected scatter per unit length for a nominal muon. This is a clean material fingerprint: the thickness is normalized out, and because it scales as $1/X_0$ it inherits the sharp $Z$ separation. My target is a voxelized 3D map of $\lambda(x,y,z)$.

The honest tomographic statement is subtle, and it is why prior CT machinery does not transfer directly. In X-ray CT each ray's signal is a deterministic line integral $s_i = \sum_j w_{ij} f_j$, solved by back-projection, ART, or ML-EM. The naive move would set $f_j=\lambda_j$ and $\theta_i = \sum_j L_{ij}\lambda_j$ — but $\mathbb{E}[\theta_i]=0$, so the angle is not a line integral. The thing linear in $\lambda$ is the *variance*: $\mathrm{Var}(\theta_i)=\sum_j L_{ij}\lambda_j$, because the muon's total scatter is a sum of independent per-voxel scatters and variances add. So the raysum is the right skeleton but on the wrong moment, and I observe one sample $\theta_i$, not its variance. Before paying for a full inversion, I take the cheapest thing that localizes. If a small dense object sits in otherwise thin material, almost all the bending happens right where the muon crosses it, so I make the boldest simplification: pretend *all* the scattering happened at a single point — one kink, straight legs on either side. That kink is the intersection of the extended in/out tracks. In a 2D slice the two lines cross; in 3D they are generally skew, so I take the point of closest approach. With $r_1(t)=p_\mathrm{in}+t\,v_\mathrm{in}$, $r_2(s)=p_\mathrm{out}+s\,v_\mathrm{out}$, minimizing $|r_1-r_2|^2$ requires the connecting vector to be perpendicular to both directions; writing $w_0=p_\mathrm{in}-p_\mathrm{out}$, $a=v_\mathrm{in}\!\cdot\!v_\mathrm{in}$, $b=v_\mathrm{in}\!\cdot\!v_\mathrm{out}$, $c=v_\mathrm{out}\!\cdot\!v_\mathrm{out}$, $d=v_\mathrm{in}\!\cdot\!w_0$, $e=v_\mathrm{out}\!\cdot\!w_0$ gives the $2\times 2$ system whose solution is $t=(be-cd)/(ac-b^2)$, $s=(ae-bd)/(ac-b^2)$, and the vertex is the midpoint $\tfrac12(r_1(t)+r_2(s))$. The denominator $ac-b^2$ vanishes exactly when the tracks are parallel — no kink to localize — so I skip that muon. What I deposit there is a variance sample: the unbiased estimate of a zero-mean Gaussian's variance from a single draw is the draw squared, so in one projected view $s=(\theta_\mathrm{out}-\theta_\mathrm{in})^2$, and in 3D I average the two independent projected views, $s=\tfrac12[(\Delta\theta_x)^2+(\Delta\theta_y)^2]$. I add $s$ to the voxel containing the PoCA point, and I count every voxel crossed by the estimated straight path (a muon whose vertex lands elsewhere still contributes a genuine zero to the voxels it merely traversed). After $M$ muons each voxel holds a sum of assigned samples and a path count $I_j$, and

$$\hat\lambda(j) = \frac{\sum_{\text{assigned}} s}{I_j\,L},$$

the mean projected $\theta^2$ per crossing divided by $L$ to turn variance-per-crossing into variance-per-unit-length. This is just the single-slab estimator $\hat\lambda=(1/L)(1/M)\sum\theta_i^2$, except PoCA has routed each muon's nonzero contribution into one voxel and left zeros along the rest of its path. It is $O(M)$, non-iterative, and sharp for an isolated high-$Z$ lump — but the single-scatter bet is a lie whenever scattering is distributed (two objects on one path, or a thick extended mass): then the in/out lines cross at a geometric compromise that need not lie inside any real scatterer, the image smears and grows ghost density, and the method wastes the lateral-displacement information entirely.

The honest fix returns to $\mathrm{Var}(\theta_i)=\sum_j L_{ij}\lambda_j$ and refuses to collapse it to a point — MLS (Maximum Likelihood, Scattering). I model each projected angle as a zero-mean Gaussian whose variance is the path-length raysum, $P(\theta_i\mid\lambda)=\frac{1}{\sqrt{2\pi v_i}}\exp(-\theta_i^2/2v_i)$ with $v_i=(L\lambda)_i$, treat the two projected views as two independent samples sharing the same path-length row, multiply over independent rays, take $-2\log$ and drop constants, giving

$$\hat\lambda = \arg\min_\lambda \sum_i\Big[\ln v_i + \frac{\theta_i^2}{v_i}\Big],\quad v_i=(L\lambda)_i,\quad \lambda_j \ge \lambda_\mathrm{air}.$$

This is deliberately not least squares: because the unknown enters as the variance, the $\ln v$ term penalizes inflating the variance to fit, balanced against $\theta^2/v$ that rewards explaining the observed scatter — the correct negative log-likelihood of a Gaussian with unknown variance. Now every voxel on a path carries its share, and many overlapping rays from many angles jointly pin down which voxels are dense, resolving distributed scattering that PoCA cannot. The non-negativity floor $\lambda_j\ge\lambda_\mathrm{air}$ is physical (negative scattering density is meaningless) and regularizes under-determined cells.

One degeneracy remains that the angle alone cannot break: $\mathrm{Var}(\theta)=\sum_j L_j\lambda_j$ does not care *where* on the path the scattering sat, so a thin dense layer near the top and the same layer near the bottom give identical total angle. The lateral displacement breaks it, because where the kick happens changes how much sideways offset it accumulates by the exit — this is MLSD. For a single slab, scattering and displacement are jointly zero-mean Gaussian with $\mathrm{Var}(\Delta\theta)=L\lambda$, $\mathrm{Var}(\Delta x)=(L^3/3)\lambda$, $\mathrm{Cov}(\Delta\theta,\Delta x)=(L^2/2)\lambda$ — the higher powers of $L$ are exactly the position information. Propagating through a stack, an upstream kick gets amplified into displacement by the lever arm of everything downstream of it: with $T_j$ the path length downstream of cell $j$, $\Delta\theta=\sum_j\Delta\theta_j$ and $\Delta x=\sum_j(\Delta x_j + T_j\Delta\theta_j)$. Taking variances (cross-layer terms vanish since layers scatter independently and zero-mean) yields three weight vectors, all linear in $\lambda$,

$$W_\theta(j)=L_j,\quad W_{\theta x}(j)=\frac{L_j^2}{2}+L_j T_j,\quad W_x(j)=\frac{L_j^3}{3}+T_j L_j^2 + T_j^2 L_j,$$

so each ray has a $2\times2$ covariance $\Sigma_i=\begin{bmatrix}W_\theta\!\cdot\!\lambda & W_{\theta x}\!\cdot\!\lambda\\ W_{\theta x}\!\cdot\!\lambda & W_x\!\cdot\!\lambda\end{bmatrix}$ and data $d_i=[\Delta\theta_i;\ (x_\mathrm{out}-x_\mathrm{proj})\cos\theta_\mathrm{avg}]$, where $x_\mathrm{proj}$ projects the incoming track forward to the exit plane and the cosine measures displacement perpendicular to the mean path. The reconstruction has the identical shape, with the scalar variance promoted to a determinant and quadratic form,

$$\hat\lambda = \arg\min_\lambda \sum_i\Big[\ln|\Sigma_i| + d_i^\top \Sigma_i^{-1} d_i\Big],\quad \lambda_j\ge\lambda_\mathrm{air}.$$

The single-slab angle/displacement correlation is $(L^2/2)/\sqrt{L\cdot L^3/3}=\sqrt{3}/2\approx0.866$, independent of material — strong but not perfect, so displacement genuinely adds an independent coordinate, and the $L^3/3$, $L^2/2$ moments together with the downstream lever arms encode where along each ray the scattering happened, lifting the top-versus-bottom degeneracy MLS leaves behind.

```python
import numpy as np
from scipy.optimize import minimize

P0_MEV = 3000.0  # nominal muon momentum, 3 GeV/c

def scattering_density(X0_cm, p0_mev=P0_MEV):
    """lambda = (15/p0)^2 / X0  (Highland width squared per unit length), rad^2/cm."""
    return (15.0 / p0_mev) ** 2 / X0_cm

LAMBDA_AIR = scattering_density(X0_cm=3.04e4)   # X0(air) ~ 304 m

class VoxelGrid:
    def __init__(self, lo, hi, n):
        self.lo, self.hi, self.n = np.asarray(lo, float), np.asarray(hi, float), np.asarray(n, int)
        self.size = (self.hi - self.lo) / self.n
        self.L = float(np.mean(self.size))
    def index(self, pt):
        idx = ((np.asarray(pt) - self.lo) / self.size).astype(int)
        if np.any(idx < 0) or np.any(idx >= self.n):
            return None
        return tuple(idx)

def line_voxels(p0, p1, grid):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / np.min(grid.size))) * 3)
    hits, seen = [], set()
    for alpha in np.linspace(0.0, 1.0, n):
        j = grid.index(p0 + alpha * (p1 - p0))
        if j is not None and j not in seen:
            seen.add(j)
            hits.append(j)
    return hits

# ----- 3D geometry: closest-approach vertex of the in/out tracks -----
def closest_approach_point(p_in, v_in, p_out, v_out):
    v_in = v_in / np.linalg.norm(v_in)
    v_out = v_out / np.linalg.norm(v_out)
    w0 = p_in - p_out
    a, b, c = v_in @ v_in, v_in @ v_out, v_out @ v_out
    d, e = v_in @ w0, v_out @ w0
    denom = a * c - b * b
    if abs(denom) < 1e-9:                  # parallel tracks: no kink to localize
        return None
    t = (b * e - c * d) / denom
    s = (a * e - b * d) / denom
    return 0.5 * ((p_in + t * v_in) + (p_out + s * v_out))

def projected_angles(v):
    v = v / np.linalg.norm(v)
    vz = max(abs(v[2]), 1e-12)
    return np.array([np.arctan2(v[0], vz), np.arctan2(v[1], vz)])

def projected_scattering_signal(v_in, v_out):
    dtheta = projected_angles(v_out) - projected_angles(v_in)
    return 0.5 * float(dtheta @ dtheta)

# ----- PoCA reconstruction -----
def reconstruct_poca(muons, grid):
    S = np.zeros(tuple(grid.n)); I = np.zeros(tuple(grid.n))
    for p_in, v_in, p_out, v_out in muons:
        p_in, v_in = np.asarray(p_in, float), np.asarray(v_in, float)
        p_out, v_out = np.asarray(p_out, float), np.asarray(v_out, float)
        for j in line_voxels(p_in, p_out, grid):
            I[j] += 1
        pt = closest_approach_point(p_in, v_in, p_out, v_out)
        if pt is None:
            continue
        j = grid.index(pt)
        if j is None:
            continue
        if I[j] == 0:
            I[j] += 1
        S[j] += projected_scattering_signal(v_in, v_out)
    lam = np.zeros_like(S); nz = I > 0
    lam[nz] = S[nz] / (I[nz] * grid.L)   # mean projected theta^2 per unit length
    return lam

# ----- Maximum-likelihood (MLS) reconstruction -----
def reconstruct_mls(signals, Lmat, lambda_air=LAMBDA_AIR, lam0=None):
    Lmat = np.asarray(Lmat, float)
    s2 = np.asarray(signals, float) ** 2
    n = Lmat.shape[1]
    def nll(lam):
        v = np.maximum(Lmat @ lam, 1e-12)
        return np.sum(np.log(v) + s2 / v)        # -2 logL of projected Gaussians
    x0 = np.full(n, lambda_air) if lam0 is None else np.asarray(lam0, float)
    return minimize(nll, x0, method="L-BFGS-B", bounds=[(lambda_air, None)] * n).x
```
