Let me start from the thing that is actually bothering me, which is bookkeeping. I have the double-dispersion representation in front of me — the amplitude written as a double integral over spectral densities ρ_st, ρ_su, ρ_tu, dividing the singularities of the four-point function into three clean pieces. Beautiful object. But the way I have it written, with no subtractions, it only makes sense if the amplitude dies at infinity. The integrals ∬ ρ/[(s′−s)(t′−t)] presuppose convergence. And I do not know that the amplitude dies. If it grows, I must subtract — pull out a polynomial in s before I disperse — and the number of subtractions is exactly the degree of growth I am forced to tolerate. So the innocuous-sounding question "how many subtractions?" is really "how fast can the amplitude grow with energy?" and I cannot answer it by staring at the representation, because the representation is happy to accept whatever growth I feed it.

And there is a sharp physical edge to this. Take the forward direction and the optical theorem: σ_tot = (4π/k) Im f(s, 0), with f the c.m. amplitude normalized so dσ/dΩ = |f|². If the forward amplitude grows like s^N, the total cross section grows like s^{N−1}. So "how many subtractions" is, through one identity, "how fast can σ_tot grow." Nobody has put a ceiling on that. Cross sections look roughly flat in the data and Pomeranchuk has arguments about a particle and its antiparticle approaching the same limit, but those *assume* boundedness, they don't derive it. I want to derive it. From what? From the only two things I genuinely trust: the analyticity that the representation encodes, and the conservation of probability.

So let me forget the representation as a calculational engine and ask what it *forbids*. The content I trust is: (a) at fixed t in some neighbourhood of zero, the amplitude — really its absorptive part — is bounded by a power of s, |A(s,t)| ≤ C(t) s^N for some finite N. That is just "a finite number of subtractions" said out loud. I'll keep N unspecified; if the answer depends only weakly on N, so much the better. And (b) analyticity in the *angle*: at fixed physical s the amplitude is analytic in cos θ inside an ellipse, Lehmann's ellipse, foci at ±1, real semi-axis z_0(s) = 1 + t_0/2k². Here t_0 is where the nearest t-channel singularity sits. For pion processes that is two-pion exchange, t_0 = 4m_π². So I have two facts: power-law boundedness at fixed t, and angular analyticity out to a known ellipse.

Now, which amplitude do I push on — f or its imaginary part A? Let me think about which one unitarity actually controls. The partial-wave statement of probability conservation is that S_l = 1 + 2i a_l has |S_l| ≤ 1, which unpacks to Im a_l ≥ |a_l|². The right-hand side is non-negative, so Im a_l ≥ 0, and combined with the cap it gives 0 ≤ Im a_l ≤ 1. Two gifts in one line: the imaginary parts are *positive*, and each is *bounded by one*. The full a_l are complex with no sign; their phases would let cancellations hide growth. The imaginary parts cannot cancel — they all push the same way. So I want to work with A(s,t) = Im_s f(s,t), whose partial-wave coefficients are exactly these positive, capped Im a_l. The absorptive part is the right object precisely because positivity means a single large term cannot be secretly offset by another, and a bound at one angle propagates to the whole sum.

Write it out. With f(s,t) = (1/k) Σ_{l≥0} (2l+1) a_l(s) P_l(z), its s-channel imaginary part is
  A(s, t) = (1/k) Σ_{l≥0} (2l+1) Im a_l(s) P_l(z),  z = cos θ = 1 + t/2k².
At the forward point z = 1, P_l(1) = 1, and the optical theorem σ_tot = (4π/k) Im f(s,0) is
  σ_tot(s) = (4π/k²) Σ_{l≥0} (2l+1) Im a_l(s).
So if I can say how many of these Im a_l are appreciable, and bound each by unitarity, I bound σ_tot. The whole game is: how many partial waves matter?

Here is where the two facts have to meet, and at first it looks hopeless. The Lehmann ellipse has real semi-axis z_0 = 1 + t_0/2k². But k² grows with s — for equal masses k² → s/4 — so z_0 → 1 + 2t_0/s → 1 as s → ∞. The ellipse of angular analyticity *collapses* onto the physical segment [−1,1]. My one nontrivial analytic input is evaporating exactly in the limit I care about. If the ellipse shrinks to nothing, what constraint is left on the partial waves?

Let me not give up on it; let me see what an ellipse — even a thin one — does to a *positive* Legendre series. A series Σ c_l P_l(z) with positive c_l, convergent inside an ellipse of real semi-axis z_0, behaves like a power series with positive coefficients: its radius of convergence is governed by the boundary, and the coefficients must decay at least as fast as the boundary growth of P_l. And P_l does grow for z > 1: from the Laplace integral
  P_l(x) = (1/π) ∫_0^π (x + cos φ √(x²−1))^l dφ,  x ≥ 1,
the integrand's largest value is (x + √(x²−1))^l at φ = 0, so P_l(x) is of order (x + √(x²−1))^l / √l for large l. So at the ellipse boundary z_0 the term (2l+1) Im a_l P_l(z_0) carries a factor (z_0 + √(z_0²−1))^l. For the positive series to even converge at the boundary — and it does converge inside, A being analytic there and bounded — the coefficients must beat this:
  Im a_l(s) ≲ (bound on A at z_0) × (z_0 + √(z_0²−1))^{−l}.

Let me make the exponent concrete. Put z_0 = 1 + ε with ε = t_0/2k² small. Then √(z_0²−1) = √((1+ε)²−1) = √(2ε + ε²) ≈ √(2ε), and z_0 + √(z_0²−1) ≈ 1 + √(2ε). So
  (z_0 + √(z_0²−1))^{−l} ≈ (1 + √(2ε))^{−l} ≈ exp(−l √(2ε)).
And √(2ε) = √(t_0/k²) = √(t_0)/k. So the partial waves are damped like
  Im a_l(s) ≲ P(s) · exp(−l √(t_0)/k),
where P(s) is whatever polynomial bounds A on the contour — this is where the "finite number of subtractions" enters, P(s) ∼ s^N up to slowly varying factors. So the collapsing ellipse did *not* kill the constraint. It thinned it, replacing a hard cutoff by an exponential decay, and the decay rate is √(t_0)/k. The pion mass scale is still there, sitting in t_0.

Now I can see the shape of the answer forming. Each Im a_l is bounded two ways: by unitarity it is ≤ 1, and by analyticity it is ≲ s^N exp(−l√(t_0)/k). For small l the exponential factor is near one, so unitarity is the binding constraint and Im a_l can be O(1). For large l the exponential crushes it far below one, so analyticity is binding and the wave is negligible. The crossover is where s^N exp(−l√(t_0)/k) drops to order one:
  s^N exp(−L √(t_0)/k) ≈ 1  ⇒  L √(t_0)/k ≈ N ln s  ⇒  L ≈ (k/√(t_0)) · N ln s.
With k ≈ √s/2,
  L ≈ (N/2) · (√s/√(t_0)) · ln s.
So the number of *significant* partial waves grows. It is not constant. There is no fixed cutoff; the angular content of the amplitude keeps fattening with energy. But — and this is the crucial restraint — it fattens only logarithmically per unit of momentum, because the decay rate √(t_0)/k weakens like 1/√s while the polynomial only contributes a ln s. Let me hold that picture: O(1) waves up to l ≈ L, exponentially small beyond. Drop the tail; it is exponentially suppressed and cannot compete with what the retained block gives.

Now feed it through the optical theorem. Forward, P_l(1) = 1, each retained Im a_l ≤ 1 by unitarity:
  σ_tot(s) = (4π/k²) Σ_{l≥0} (2l+1) Im a_l(s) ≲ (4π/k²) Σ_{l=0}^{L} (2l+1) · 1.
And Σ_{l=0}^{L} (2l+1) = (L+1)² ≈ L². So
  σ_tot(s) ≲ (4π/k²) L².
Substitute L ≈ (N/2)(√s/√(t_0)) ln s and k² ≈ s/4:
  σ_tot(s) ≲ (4π/(s/4)) · (N²/4)(s/t_0) ln²s = (16π/s) · (N²/4)(s/t_0) ln²s = (4π N²/t_0) ln²s.
The s cancels. The √s in L squares to s and cancels the 1/k² ∼ 1/s, and what survives is the logarithm, squared. So
  σ_tot(s) ≤ (4π N² / t_0) · ln² s,
or more honestly with a scale to make the argument dimensionless, ≤ (4π N²/t_0) ln²(s/s_0). The growth is at most the *square of the logarithm* of the energy. That is the ceiling. Let me breathe and check I believe each step.

Hold on — let me re-examine whether the tail really is negligible, because I waved at it. The retained block gives ∼ (4π/k²)L². The tail is Σ_{l>L} (2l+1) Im a_l with Im a_l ≲ s^N exp(−l√(t_0)/k). For l just above L the bound is ≈ 1 and decreasing geometrically with ratio exp(−√(t_0)/k); the geometric sum of (2l+1) times a decaying exponential, starting from l ≈ L, is of order L/(√(t_0)/k) — i.e. L·k/√(t_0). Now L = (k/√(t_0))N ln s, so L·k/√(t_0) = L²/(N ln s): the tail is smaller than the block L² by a whole factor of the logarithm. So it certainly does not change the ln²s scaling — it is parametrically subleading. Good. The square-log law is robust. The block dominates and the answer is ln²s. I won't pretend I've nailed the optimal constant by this crude argument — the precise coefficient wants a sharper treatment of the Legendre inequality and of where exactly to truncate — but the *law* is forced.

Let me also double-check the dependence on N, the subtraction count, because if the answer blew up with N that would be a problem — I left N unspecified on purpose. It enters as N² in the coefficient, not in the exponent of the logarithm. So the *shape* ln²s is completely independent of how many subtractions the representation needs; only the prefactor remembers N. That is exactly the kind of robustness I wanted: the ceiling's functional form follows from analyticity-plus-unitarity alone, and the messy field-theoretic question of the exact polynomial degree only rescales it. Good. (And one can pin N down separately — fixed-t dispersion relations with the absorptive part bounded as derived turn out to close consistently with two subtractions, N = 2, by a Phragmén–Lindelöf argument; that would give the coefficient with N − 1 in place of N in a sharper accounting, but I don't need that to establish the law.)

Now the coefficient, with t_0 nailed down. The nearest t-channel singularity for pion processes is two-pion exchange, t_0 = 4m_π². So
  4π/t_0 = 4π/(4m_π²) = π/m_π².
Up to the subtraction-count constant, the ceiling is
  σ_tot(s) ≤ (4π/t_0) ln²(s/s_0) = (π/m_π²) ln²(s/s_0).
The pion mass is the whole physical content of the coefficient. That makes complete sense: m_π is the lightest hadron, it sets the *longest range* of the force, and a longer range means a bigger disk and a larger ceiling. If there were a lighter exchanged particle, t_0 would be smaller and the ceiling larger. The lightness of the pion is what keeps the cross section from running away faster.

I want to stress-test the result against a wall I half-fear: did I cheat by using a *positive* series? What if I'd used the full amplitude f with complex a_l? Then the analyticity bound on |a_l| still gives exponential decay in l, but I would have lost the optical-theorem positivity — at the forward point the sum Σ(2l+1) Im a_l could in principle have had cancellations if Im a_l weren't sign-definite, and a single large term could have been masked. Unitarity rescues exactly that: Im a_l ≥ 0. So the choice of the absorptive part, made early because it had positive coefficients, is what made the optical-theorem step clean. The two roles of unitarity — positivity of Im a_l, and the cap Im a_l ≤ 1 — are both indispensable, and they are different uses of the same conservation law. That's reassuring; the argument isn't using anything I haven't earned.

Let me also make sure I am not secretly assuming what I want to prove. Did I assume σ_tot is bounded anywhere? No. I assumed |A(s,t)| ≤ C(t)s^N at fixed t in a neighbourhood of zero — polynomial boundedness, the finite-subtraction hypothesis — which is *weaker* than bounding σ_tot, and is the natural field-theoretic input (matrix elements are polynomially bounded). The ceiling on σ_tot came *out*, it did not go *in*. Good.

The partial-wave manipulation is correct but it hides the physics, and I don't yet *feel* why it has to be ln²s. Each partial wave l corresponds, semiclassically, to an impact parameter b = l/k. The retained waves go up to l ≈ L = (k/√(t_0)) N ln s, so the √s in k cancels and the reach is an impact parameter
  b_max = L/k ≈ (N/√(t_0)) ln s,
energy-independent except for the logarithm. The interaction presents a disk of radius b_max that grows — but only logarithmically. A black disk of radius b_max has cross section of order π b_max² ∼ (π N²/t_0) ln²s — the same functional form and the same 1/t_0 scale as the rigorous ceiling, agreeing with it up to an O(1) geometric factor. So the square-log is just the *area of a logarithmically growing disk*. And why does the disk grow only logarithmically? Picture the crudest possible model: two particles interacting through an absorptive Yukawa exchange of mass κ, so the interaction strength at impact parameter a behaves like g e^{−κa} for some coupling g. Where g e^{−κa} ≳ 1 the partner is essentially fully absorbed; where it is ≪ 1 there is practically no scattering. The edge of the interaction disk is the impact parameter where g e^{−κa} ≈ 1, i.e. a_max = (1/κ) ln|g|. Then the cross section is about the area of that disk,
  σ ≈ π a_max² = (π/κ²) ln²|g|.
Now suppose the effective coupling grows with energy — say like a power of s. Then ln|g| ∝ ln s, and
  σ ∝ (π/κ²) ln²s.
The range scale κ is the mass of the lightest exchanged quantum, the pion, so κ ↔ m_π and π/κ² ↔ π/m_π², matching the coefficient I derived. The exponential fall-off of a finite-range force is exactly what turns a power-law growth of the coupling into a logarithmic growth of the radius; and area being radius squared gives the square. The whole rigorous edifice — Lehmann ellipse, positive Legendre series, optical theorem — is the field-theoretic incarnation of "a finite-range interaction can only grow a logarithmic disk."

Let me trace the causal chain once, cleanly, to be sure nothing is dangling. The Mandelstam representation with a finite number of subtractions means the absorptive part is polynomially bounded at fixed t. Analyticity in the angle (Lehmann's ellipse, real semi-axis z_0 = 1 + t_0/2k², t_0 = 4m_π² the two-pion threshold) means the positive Legendre series for A converges inside that ellipse, which forces the partial waves to decay like exp(−l√(t_0)/k). Crossed with the unitarity cap Im a_l ≤ 1, the waves are O(1) only up to l ≈ L ≈ (k/√(t_0))·N ln s ≈ (N/2)(√s/√(t_0)) ln s; beyond L they are exponentially negligible. The optical theorem, using the *positivity* of Im a_l, sums the retained block: σ_tot ≲ (4π/k²)·(L+1)² ≈ (4π N²/t_0) ln²s. With t_0 = 4m_π² and the subtraction count, σ_tot(s) ≤ (4π/t_0) ln²(s/s_0) = (π/m_π²) ln²(s/s_0). Semiclassically: a logarithmically growing interaction disk of radius b_max = L/k ∼ (N/√(t_0)) ln s, area π b_max² ∼ (π N²/t_0) ln²s — the same form and scale as the rigorous ceiling up to an O(1) geometric factor; finite range (pion mass) forbids anything faster.

Let me put down a compact numerical illustration so the law and its angular-momentum content are concrete: evaluate the number of significant partial waves and the resulting ceiling as functions of s, and confirm that the forward sum, capped by unitarity, tracks ln²s.

```python
import numpy as np

# pion-mass units; equal-mass elastic scattering
m_pi = 1.0
t0   = 4 * m_pi**2          # two-pion threshold = nearest t-channel singularity
def k2(s):                  # c.m. momentum squared
    return (s - 4*m_pi**2) / 4.0

# angular-analyticity ellipse (Lehmann): real semi-axis and the decay rate it forces
def decay_rate(s):
    # Im a_l <~ poly(s) * exp(-l * rate);  rate = sqrt(t0)/k  from z0 = 1 + t0/(2 k^2)
    return np.sqrt(t0) / np.sqrt(k2(s))

# number of significant partial waves: where poly(s) ~ s^N meets the unitarity cap (=1)
N_sub = 2                    # finite number of subtractions
def num_significant_partial_waves(s):
    return N_sub * np.log(s) / decay_rate(s)        # L ~ (k/sqrt(t0)) N ln s

# optical theorem with each retained wave capped by unitarity (Im a_l <= 1)
def sigma_tot_ceiling_from_waves(s):
    L = int(num_significant_partial_waves(s))
    l = np.arange(L + 1)
    return (4*np.pi / k2(s)) * np.sum(2*l + 1)       # = (4 pi / k^2)(L+1)^2

# closed-form ceiling the derivation lands on
def cross_section_ceiling(s, s0=1.0):
    return (4*np.pi / t0) * (N_sub**2) * np.log(s/s0)**2   # = (pi/m_pi^2) N^2 ln^2 s  (t0=4)

for s in [1e2, 1e4, 1e6, 1e8]:
    summed  = sigma_tot_ceiling_from_waves(s)
    closed  = cross_section_ceiling(s)
    b_max   = num_significant_partial_waves(s) / np.sqrt(k2(s))   # interaction radius L/k
    print(f"s={s:.0e}  L~{num_significant_partial_waves(s):8.1f}  "
          f"sum(2l+1)/k^2 ->{summed:10.2f}  (4pi/t0)N^2 ln^2 ->{closed:10.2f}  "
          f"b_max={b_max:6.3f}~(N/sqrt(t0))ln s")
# both columns grow like ln^2 s and agree up to the O(1) truncation constant;
# b_max = L/k grows only logarithmically -> pi b_max^2 ~ (pi N^2/t0) ln^2 s,
# the same form/scale as the rigorous (4pi N^2/t0) ln^2 s up to an O(1) geometric factor.
```
