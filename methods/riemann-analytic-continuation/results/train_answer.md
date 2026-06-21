I would call the method the Riemann analytic continuation of the zeta function, because the whole achievement is to take Euler's product, which only makes sense in a half-plane, and extend it into a genuine complex function whose zeros can be studied and used.

The starting point is the stubborn fact that the primes themselves look irregular, while their count does not. Gauss noticed from tables that the number of primes below x, written π(x), stays remarkably close to the logarithmic integral Li(x) = ∫₀ˣ dt / log t, but the two are not exactly equal. There is a small, oscillating discrepancy that grows slowly and changes sign in a way no one could explain. Empirical fits like Legendre's x/(log x − A) capture the trend a little better in places, but they are still fits, not laws, and they say nothing about the wobble. What is needed is a single analytic object that contains all primes at once and is smooth enough that complex analysis can be applied to it.

That object is Euler's product. Unique factorization gives every integer exactly once as a product of prime powers, and the geometric series for each prime gives 1/(1 − p⁻ˢ) = 1 + p⁻ˢ + p⁻²ˢ + ⋯. Multiplying over all primes and collecting terms packages every prime into one expression:

ζ(s) = ∏ₚ 1/(1 − p⁻ˢ) = ∑_{n≥1} 1/nˢ.

This is the bridge between the primes and analysis. The problem is that both sides converge only when Re(s) > 1. At s = 1 the series becomes the harmonic series and diverges, and for Re(s) ≤ 1 the formula is meaningless as written. So the prime information is trapped in a half-plane where nothing interesting can happen: ζ has no zeros there because every factor in the product is nonzero. To use contour integration, residues, and the location of zeros, ζ must be continued to the whole complex plane.

The continuation begins with Euler's factorial integral, the Gamma function Π(s−1) = Γ(s) = ∫₀^∞ e⁻ˣ xˢ⁻¹ dx. Substituting x → nx gives ∫₀^∞ e⁻ⁿˣ xˢ⁻¹ dx = Π(s−1)/nˢ, which turns a single term of the Dirichlet series into an integral. Summing over n and exchanging sum and integral yields

Π(s−1) ζ(s) = ∫₀^∞ xˢ⁻¹/(eˣ − 1) dx,   Re(s) > 1.

The integrand 1/(eˣ − 1) is explicit and has poles at x = 2πi n for every integer n, all marching up the imaginary axis. That is exactly what was missing: a function with known poles that a contour can be pushed past. Replace the real integral by a Hankel keyhole contour that comes in from +∞, loops around the origin, and returns to +∞, and write the numerator as (−x)ˢ⁻¹ with the logarithm fixed so it is real when x is negative. The two straight runs above and below the positive real axis differ by a phase e^{±πi(s−1)}, and their contribution reproduces the original integral multiplied by −2i sin(πs). Tidying gives

2 sin(πs) Π(s−1) ζ(s) = i ∫_{∞}^{∞} (−x)ˢ⁻¹/(eˣ − 1) dx.

The right-hand side is a contour integral whose integrand is analytic everywhere on the contour, so it defines an entire function of s. Dividing by 2 sin(πs) Π(s−1) therefore continues ζ(s) to all of ℂ, with only a simple pole at s = 1 coming from the harmonic divergence. At the negative even integers sin(πs) vanishes while the contour integral stays finite, forcing ζ(s) to vanish there. Those are the trivial zeros, s = −2, −4, −6, ….

Pushing the same contour the other way, outward to enclose the poles at x = 2πi n with n ≠ 0, expresses the integral as a sum of residues. That residue calculation relates ζ(s) to ζ(1−s), giving the functional equation. A cleaner way to see the symmetry is to feed the Gamma integral e^{−n²πx} instead of e^{−nx}. This produces the symmetric package

Π(s/2 − 1) π^{−s/2} ζ(s) = ∫₀^∞ ψ(x) x^{s/2−1} dx,   Re(s) > 1,

where ψ(x) = ∑_{n≥1} e^{−n²πx}. The function ψ is a theta series, and Jacobi's theta transformation says 2ψ(x) + 1 = x^{−1/2}(2ψ(1/x) + 1), a self-duality under x ↔ 1/x that comes from the Fourier self-reciprocity of the Gaussian. The Mellin integral turns that x ↔ 1/x symmetry into s ↔ 1−s symmetry. Splitting the integral at 1 and applying Jacobi's identity gives the manifestly symmetric continuation

Π(s/2 − 1) π^{−s/2} ζ(s) = 1/(s(s−1)) + ∫₁^∞ ψ(x) (x^{s/2−1} + x^{(1−s)/2−1}) dx.

The right-hand side is unchanged when s is replaced by 1−s. The term 1/(s(s−1)) has simple poles at s = 0 and s = 1; the pole at s = 1 is ζ's true pole, while the pole at s = 0 comes from Π(s/2−1) = Γ(s/2). Multiplying by ½ s(s−1) kills both poles without breaking the symmetry, producing the entire function

ξ(s) = ½ s(s−1) π^{−s/2} Π(s/2 − 1) ζ(s),   with ξ(s) = ξ(1−s).

This ξ is entire, real on the real axis, and its zeros are exactly the nontrivial zeros of ζ, all lying in the critical strip 0 < Re(s) < 1. Because ξ is real on the real axis and symmetric under s ↔ 1−s, the zeros come in quadruples ρ, 1−ρ, ρ̄, 1−ρ̄, reflected across Re(s) = ½ and across the real axis. The argument principle applied to a box in the critical strip shows that the number of zeros with imaginary part between 0 and T is approximately (T/2π) log(T/2π) − T/2π, so they thin out only logarithmically.

Since ξ is entire, it can be written as a product over its zeros in the same spirit as Euler's product for sin:

ξ(s) = ξ(0) ∏_ρ (1 − s/ρ),

taken with the zeros grouped in reflected pairs so the product converges. Taking logarithms expresses log ζ(s) in terms of the zeros, the pole at s = 1, and Gamma and pi factors. Now the primes can be recovered. The natural counting function is not π(x) directly but F(x), which jumps by 1/m at each prime power pᵐ. Its Mellin transform is

log ζ(s)/s = ∫₁^∞ F(x) x^{−s−1} dx,   Re(s) > 1.

Fourier inversion of this transform gives F(x) as a contour integral of log ζ(s)/s. Substituting the zero product and evaluating the contour term by term yields Riemann's explicit formula:

F(x) = Li(x) − ∑_{Im ρ > 0} (Li(x^ρ) + Li(x^{1−ρ})) + ∫_x^∞ dt/(t(t²−1) log t) + log ξ(0),

with log ξ(0) = −log 2. The sum runs over the nontrivial zeros in symmetric order. Each conjugate pair ρ = β + iγ, ρ̄ contributes an oscillation proportional to x^β cos(γ log x + φ)/log x, so β controls the amplitude and γ controls the frequency in log x. The fluctuations of π(x) about Li(x) are the superposition of these zero-indexed waves. Finally, Möbius inversion of F(x) = ∑_n (1/n) π(x^{1/n}) gives π(x) itself as a similar combination of Li terms corrected by the zero sum.

The Riemann Hypothesis is the statement that every nontrivial zero has real part exactly ½, or equivalently that all roots of Ξ(t) = ξ(½ + it) are real. It remains unproven, but its meaning is now sharp: it says the prime-counting oscillations are as small as the explicit formula permits, with each wave decaying like √x up to logarithmic factors.

To make the continuation concrete, the following Python script evaluates the symmetric integral representation of Π(s/2 − 1) π^{−s/2} ζ(s) on a short segment of the critical line and checks the functional equation ξ(s) = ξ(1−s) numerically. It also plots the modulus of ζ on the critical line to show the first few nontrivial zeros as dips.

```python
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

mp.mp.dps = 30  # high precision for complex zeta


def xi(s):
    """Riemann's entire xi(s) = 0.5*s*(s-1)*pi**(-s/2)*Gamma(s/2)*zeta(s)."""
    return mp.mpf('0.5') * s * (s - 1) * mp.pi**(-s/2) * mp.gamma(s/2) * mp.zeta(s)


def theta_psi(x):
    """Jacobi theta tail psi(x) = sum_{n>=1} exp(-pi n^2 x)."""
    return mp.nsum(lambda n: mp.e**(-mp.pi * n**2 * x), [1, mp.inf])


def zeta_from_symmetric(s, x_max=25):
    """Compute zeta(s) from Riemann's symmetric integral representation.

    Pi(s/2 - 1) * pi**(-s/2) * zeta(s)
        = 1/(s(s-1)) + integral_1^inf psi(x)(x^{s/2-1}+x^{(1-s)/2-1}) dx.
    """
    def integrand(x):
        p = theta_psi(x)
        return p * (x**(s/2 - 1) + x**((1 - s)/2 - 1))

    val = mp.quad(integrand, [1, x_max])
    sym = 1/(s*(s-1)) + val
    return sym / (mp.gamma(s/2) * mp.pi**(-s/2))


# Check the functional equation xi(s) == xi(1-s) at sample points.
test_points = [2 + 3j, -1 + 4j, 0.25 + 7j, 0.5 + 14j, -3 - 2j]
max_rel = 0.0
for s in test_points:
    lhs, rhs = xi(s), xi(1 - s)
    rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), mp.mpf('1e-30'))
    max_rel = max(max_rel, rel)
    print(f"s={s}: |xi(s)-xi(1-s)|/|xi(s)| = {float(rel):.2e}")
print(f"\nmax functional-equation relative error: {float(max_rel):.2e}")

# Compare the symmetric-integral value with mpmath's analytic continuation.
s_test = mp.mpc('0.5', '14.0')
print("\nAt s =", s_test)
print("mpmath zeta:       ", mp.zeta(s_test))
print("symmetric formula: ", zeta_from_symmetric(s_test))

# Plot |zeta(1/2 + i t)| to visualize the first few nontrivial zeros as dips.
t = np.linspace(0.1, 50, 2000)
z = np.array([abs(mp.zeta(mp.mpc('0.5'), mp.mpc(ti))) for ti in t])
plt.figure(figsize=(10, 4))
plt.plot(t, z)
plt.xlabel("t")
plt.ylabel("|zeta(1/2 + i t)|")
plt.title("Modulus of Riemann zeta on the critical line")
plt.grid(True)
plt.tight_layout()
plt.show()
```
