The question I set out to answer is whether relativistic scattering theory puts any ceiling at all on how fast a total cross section can grow with energy. For elastic two-body scattering $A+B\to A+B$ the central observable is $\sigma_{\rm tot}(s)$ at squared centre-of-mass energy $s$, and through the optical theorem $\sigma_{\rm tot}(s)=(4\pi/k)\,\mathrm{Im}\,f(s,0)$ this is the same as asking how fast the forward amplitude can grow. That, in turn, is precisely the question the Mandelstam double-dispersion representation cannot duck: written as a double integral over the spectral densities $\rho_{st},\rho_{su},\rho_{tu}$, it converges as it stands only if the amplitude dies at infinity, and if the amplitude grows one must subtract a polynomial in $s$ before dispersing — the number of subtractions being exactly the degree of growth one is forced to tolerate. So "how many subtractions?" and "how fast can $\sigma_{\rm tot}$ grow?" are one and the same question. The trouble is that none of the standard tools delivers an answer. Fixed-$t$ dispersion relations take the high-energy growth of the absorptive part as an *input*; the Mandelstam representation is agnostic about asymptotics and happily accepts whatever growth one feeds it; and Pomeranchuk's results, pointing to a common limit for a process and its charge-conjugate, *assume* boundedness rather than deriving it. The empirical near-constancy of high-energy cross sections is suggestive but proves nothing. What I want is a ceiling that follows rigorously from only the two things I genuinely trust — the analyticity encoded in the representation (with a *finite but unspecified* number of subtractions) and the conservation of probability — expressed through the physical mass scale of the theory, and reducing correctly under the optical theorem.

The result I propose is what I will call the Froissart bound: as $s\to\infty$, the total cross section cannot grow faster than the square of the logarithm of the energy,
$$\sigma_{\rm tot}(s)\ \le\ \frac{4\pi}{t_0}\,\ln^2\!\big(s/s_0\big)\ =\ \frac{\pi}{m_\pi^2}\,\ln^2\!\big(s/s_0\big),$$
where $t_0=4m_\pi^2$ is the two-pion threshold — the nearest $t$-channel singularity for pion processes — and $s_0$ is a scale that renders the logarithm dimensionless. The whole derivation turns on choosing the right object and then making two facts, which at first sight seem to be pulling in opposite directions, meet.

The first design choice is to push not on the full amplitude $f$ but on its $s$-channel absorptive part $A(s,t)=\mathrm{Im}_s f(s,t)$. The reason is unitarity. Probability conservation says $S_l=1+2i\,a_l$ obeys $|S_l|\le1$, which unpacks to $\mathrm{Im}\,a_l\ge|a_l|^2\ge0$, hence $0\le \mathrm{Im}\,a_l\le1$. This single line gives two gifts: the partial-wave imaginary parts are *positive*, and each is *capped at unity*. The full $a_l$ are complex with no fixed sign, so their phases could let cancellations hide growth; the imaginary parts cannot cancel, they all push the same way. The absorptive part is therefore the right object, because positivity means a single large term cannot be secretly offset by another, and a bound at one angle propagates to the whole sum. Writing $f(s,t)=(1/k)\sum_l(2l+1)\,a_l(s)P_l(z)$ with $z=\cos\theta=1+t/2k^2$, its imaginary part is $A(s,t)=(1/k)\sum_l(2l+1)\,\mathrm{Im}\,a_l(s)\,P_l(z)$, and at the forward point $z=1$, where $P_l(1)=1$, the optical theorem becomes
$$\sigma_{\rm tot}(s)=\frac{4\pi}{k^2}\sum_{l\ge0}(2l+1)\,\mathrm{Im}\,a_l(s).$$
So the entire game reduces to one question: how many of these $\mathrm{Im}\,a_l$ are appreciable?

The second input is analyticity in the angle. Microcausality plus the spectral conditions imply, at fixed physical $s$, that $A(s,\cos\theta)$ is analytic inside the Lehmann ellipse with foci at $\pm1$ and real semi-axis $z_0(s)=1+t_0/2k^2$. Here is the difficulty that nearly kills the program: because $k^2\to s/4$ for equal masses, $z_0\to1+2t_0/s\to1$ as $s\to\infty$. The ellipse collapses onto the physical segment $[-1,1]$ exactly in the limit I care about, so the one nontrivial analytic input seems to evaporate. The key realisation is that a thin ellipse does not destroy the constraint — it merely softens it. A positive Legendre series $\sum c_l P_l(z)$ convergent inside an ellipse of semi-axis $z_0$ behaves like a power series with positive coefficients: convergence at the boundary forces the coefficients to beat the boundary growth of $P_l$. From the Laplace integral $P_l(x)=(1/\pi)\int_0^\pi (x+\cos\phi\,\sqrt{x^2-1})^l\,d\phi$ the integrand peaks at $\phi=0$, so $P_l(x)\sim (x+\sqrt{x^2-1})^l/\sqrt{l}$ for $x>1$. Setting $z_0=1+\varepsilon$ with $\varepsilon=t_0/2k^2$, one has $\sqrt{z_0^2-1}\approx\sqrt{2\varepsilon}$ and $z_0+\sqrt{z_0^2-1}\approx1+\sqrt{2\varepsilon}$, so with $\sqrt{2\varepsilon}=\sqrt{t_0}/k$ the coefficients are damped as
$$\mathrm{Im}\,a_l(s)\ \lesssim\ s^N\,\exp\!\big(-l\,\sqrt{t_0}/k\big),$$
where the polynomial $s^N$ is whatever bounds $A$ on the contour — this is exactly where the finite-subtraction hypothesis $|A(s,t)|\le C(t)\,s^N$ enters. The collapsing ellipse did not kill the constraint; it replaced a hard cutoff by an exponential decay, and the decay rate $\sqrt{t_0}/k$ still carries the pion mass scale through $t_0$.

Now the two bounds on each $\mathrm{Im}\,a_l$ — the unitarity cap $\le1$ and the analyticity decay $\lesssim s^N\exp(-l\sqrt{t_0}/k)$ — combine into a sharp picture. For small $l$ the exponential is near one, so unitarity binds and $\mathrm{Im}\,a_l$ can be $O(1)$; for large $l$ the exponential crushes the wave far below one, so analyticity binds and the wave is negligible. The crossover is where $s^N\exp(-L\sqrt{t_0}/k)\approx1$, that is
$$L\ \approx\ \frac{k}{\sqrt{t_0}}\,N\ln s\ \approx\ \frac{N}{2}\,\frac{\sqrt{s}}{\sqrt{t_0}}\,\ln s,\qquad k\approx\sqrt{s}/2.$$
The number of significant partial waves is therefore not constant — the angular content keeps fattening with energy — but it fattens only logarithmically per unit of momentum, because the decay rate weakens like $1/\sqrt{s}$ while the polynomial only contributes a $\ln s$. Feeding this through the optical theorem, with $P_l(1)=1$ and each retained wave capped by unitarity,
$$\sigma_{\rm tot}(s)=\frac{4\pi}{k^2}\sum_l(2l+1)\,\mathrm{Im}\,a_l(s)\ \lesssim\ \frac{4\pi}{k^2}\sum_{l=0}^{L}(2l+1)=\frac{4\pi}{k^2}(L+1)^2\approx\frac{4\pi}{k^2}L^2.$$
The exponentially suppressed tail is parametrically subleading — summing $(2l+1)$ against the decaying exponential from $l\approx L$ gives a contribution of order $L^2/(N\ln s)$, smaller than the retained block by a whole factor of the logarithm — so it cannot change the scaling. Substituting $L\approx(N/2)(\sqrt{s}/\sqrt{t_0})\ln s$ and $k^2\approx s/4$,
$$\sigma_{\rm tot}(s)\ \lesssim\ \frac{4\pi}{s/4}\cdot\frac{N^2}{4}\frac{s}{t_0}\,\ln^2 s\ =\ \frac{4\pi N^2}{t_0}\,\ln^2 s.$$
The $\sqrt{s}$ in $L$ squares to $s$ and cancels the $1/k^2\sim1/s$; the $s$ disappears and what survives is the squared logarithm. With $t_0=4m_\pi^2$ one has $4\pi/t_0=\pi/m_\pi^2$, giving the bound stated above. Two features make me trust it. First, the subtraction count $N$ enters only as a prefactor $N^2$ (refined to $(n-1)^2$ in a sharper truncation, with $n=2$ following from a Phragmén–Lindelöf argument on the fixed-$t$ relations), never in the exponent of the logarithm — so the $\ln^2 s$ *law* is independent of the field-theoretic details and only the coefficient remembers them. Second, I did not assume what I set out to prove: the input was polynomial boundedness of $A$ at fixed $t$, which is *weaker* than bounding $\sigma_{\rm tot}$; the ceiling on $\sigma_{\rm tot}$ came out, it did not go in.

The physics behind the abstract manipulation is a logarithmically growing interaction disk. Partial wave $l$ corresponds semiclassically to impact parameter $b=l/k$, so the retained block reaches $b_{\max}=L/k\approx(N/\sqrt{t_0})\ln s$ — an interaction radius that is energy-independent except for the logarithm. The cross section is then of order the area of that disk, $\sigma\sim\pi b_{\max}^2\sim(\pi N^2/t_0)\ln^2 s$, matching the rigorous ceiling's form and $1/t_0$ scale up to an $O(1)$ geometric factor. Why only logarithmic growth? For an absorptive Yukawa exchange of mass $\kappa$ the interaction strength at impact parameter $a$ behaves like $g\,e^{-\kappa a}$; the partner is essentially fully absorbed where $g\,e^{-\kappa a}\gtrsim1$ and barely scattered where it is $\ll1$, so the edge of the disk sits at $a_{\max}=(1/\kappa)\ln|g|$ and $\sigma\approx\pi a_{\max}^2=(\pi/\kappa^2)\ln^2|g|$. If the effective coupling grows like a power of energy, then $\ln|g|\propto\ln s$ and $\sigma\propto(\pi/\kappa^2)\ln^2 s$. The exponential fall-off of a finite-range force — with $\kappa=m_\pi$, the lightest exchanged quantum — is exactly what turns a power-law growth of the coupling into a logarithmic growth of the radius, and area being radius squared supplies the square. A lighter exchange would mean a longer range, a smaller $t_0$, and a larger ceiling; the lightness of the pion is what keeps the cross section from running away faster.

A compact numerical illustration makes the law and its angular-momentum content concrete: it evaluates the number of significant partial waves and the resulting ceiling as functions of $s$, and confirms that the forward sum, capped by unitarity, tracks $\ln^2 s$ while $b_{\max}$ grows only logarithmically.

```python
import numpy as np

# pion-mass units; equal-mass elastic scattering
m_pi = 1.0
t0   = 4 * m_pi**2          # two-pion threshold = nearest t-channel singularity
def k2(s):                  # c.m. momentum squared
    return (s - 4*m_pi**2) / 4.0

def decay_rate(s):
    # Im a_l <~ poly(s) * exp(-l * rate);  rate = sqrt(t0)/k from z0 = 1 + t0/(2 k^2)
    return np.sqrt(t0) / np.sqrt(k2(s))

N_sub = 2                    # finite number of subtractions
def num_significant_partial_waves(s):
    return N_sub * np.log(s) / decay_rate(s)        # L ~ (k/sqrt(t0)) N ln s

def sigma_tot_ceiling_from_waves(s):
    # optical theorem; each retained wave capped by unitarity (Im a_l <= 1)
    L = int(num_significant_partial_waves(s))
    l = np.arange(L + 1)
    return (4*np.pi / k2(s)) * np.sum(2*l + 1)       # = (4 pi / k^2)(L+1)^2

def cross_section_ceiling(s, s0=1.0):
    return (4*np.pi / t0) * (N_sub**2) * np.log(s/s0)**2   # = (pi/m_pi^2) N^2 ln^2 s (t0=4)

for s in [1e2, 1e4, 1e6, 1e8]:
    summed = sigma_tot_ceiling_from_waves(s)
    closed = cross_section_ceiling(s)
    b_max  = num_significant_partial_waves(s) / np.sqrt(k2(s))
    print(f"s={s:.0e}  L~{num_significant_partial_waves(s):8.1f}  "
          f"wave-sum->{summed:10.2f}  (4pi/t0)N^2 ln^2->{closed:10.2f}  b_max={b_max:6.3f}")
# the wave-sum ceiling and the closed-form ceiling both grow like ln^2 s and agree
# up to the O(1) truncation constant; b_max grows only logarithmically.
```
