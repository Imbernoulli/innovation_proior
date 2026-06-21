# Planck's radiation law and the quantization of energy

## The problem

Find the universal spectral energy density $u(\nu,T)$ of black-body (cavity) radiation. By Kirchhoff
it depends only on frequency and temperature. It must reduce to Wien's exponential law at short
wavelengths, to the linear-in-$T$ behaviour seen in the long-wavelength infrared, and integrate to a
finite Stefan–Boltzmann $T^4$ total. No classical law does all three: equipartition over the cavity's
electromagnetic modes gives $u=8\pi\nu^2 kT/c^3$, which diverges as $\nu\to\infty$.

## The key idea

The energy of a cavity resonator of frequency $\nu$ cannot take continuous values; it is restricted
to **integer multiples of a finite energy element $\varepsilon=h\nu$**. This is forced, not assumed:
the resonator entropy can only be computed via Boltzmann's $S=k\log W$, and the number of microstates
$W$ is only a finite, countable quantity if the total energy is divided into discrete elements. Wien's
displacement law (entropy depends on $U/\nu$ only) then fixes the element to be proportional to
frequency, $\varepsilon=h\nu$, with $h$ a new universal constant — the quantum of action.

## The derivation

**Reduce to one resonator.** Model the cavity wall as Hertzian resonators. A material-independent
electromagnetic relation links a resonator's mean energy $U$ to the surrounding spectral density:
$$ u(\nu,T) = \frac{8\pi\nu^2}{c^3}\,U(\nu,T). $$
With $1/T = dS/dU$, the unknown is the resonator entropy $S(U)$.

**Count the microstates (Boltzmann).** Take $N$ resonators of frequency $\nu$ with total energy
$U_N=NU$ and total entropy $S_N=NS$. To count distributions of energy, the total must be discrete:
$U_N = P\varepsilon$, $P$ a large integer. The number of ways to place $P$ indistinguishable energy
elements into $N$ resonators ("complexes") is
$$ W = \frac{(N+P-1)!}{(N-1)!\,P!}. $$

**Stirling and the entropy.** With $\log m!\approx m\log m - m$ for large $N,P$,
$$ S_N = k\log W = k\big[(N+P)\log(N+P) - N\log N - P\log P\big]. $$
Per resonator, with $P/N = U/\varepsilon$,
$$ S = k\Big[\Big(1+\tfrac{U}{\varepsilon}\Big)\log\Big(1+\tfrac{U}{\varepsilon}\Big)
        - \tfrac{U}{\varepsilon}\log\tfrac{U}{\varepsilon}\Big]. $$

**Fix the element by Wien's displacement law.** The displacement law and the resonator bridge give
$U=\nu\phi(\nu/T)$. Inverting this as $\nu/T=\psi(U/\nu)$ makes
$dS/dU=1/T=(1/\nu)\psi(U/\nu)$, so integration at fixed $\nu$ gives $S=F(U/\nu)$. Since the counted
$S$ depends on $U/\varepsilon$, consistency requires
$$ \boxed{\;\varepsilon = h\nu\;} $$
($h$ a universal constant, units of energy$\times$time).

**Invert for the mean energy.** With $\varepsilon=h\nu$, $\;dS/dU = \dfrac{k}{h\nu}\log\!\Big(1+\dfrac{h\nu}{U}\Big) = \dfrac1T$, so
$$ \boxed{\;U(\nu,T) = \frac{h\nu}{e^{h\nu/kT}-1}\;}. $$

**Assemble the radiation law.** Through the mode bridge,
$$ \boxed{\;u(\nu,T) = \frac{8\pi h\nu^3}{c^3}\,\frac{1}{e^{h\nu/kT}-1}\;}. $$

## Limits (the law unifies the two regimes)

- **Long wavelength / low frequency** $h\nu\ll kT$: $e^{h\nu/kT}-1\approx h\nu/kT$, so $U\to kT$ and
  $u\to 8\pi\nu^2 kT/c^3$ — the classical equipartition (Rayleigh–Jeans) law. The element $h\nu$ is
  negligible against $kT$, so the energy looks continuous and classical physics is recovered.
- **Short wavelength / high frequency** $h\nu\gg kT$: $e^{h\nu/kT}-1\approx e^{h\nu/kT}$, so
  $u\to (8\pi h\nu^3/c^3)\,e^{-h\nu/kT}$ — Wien's distribution.
- **Finite total / no high-frequency divergence**: because adding energy to a mode of frequency $\nu$
  costs a whole element $h\nu$, modes with $h\nu\gg kT$ are exponentially suppressed; $\int u\,d\nu$
  is finite and scales as $T^4$ (Stefan–Boltzmann), unlike the classical equipartition spectrum which
  diverges.

## What it costs and yields

The derivation introduces two universal constants: $k$ (Boltzmann's constant, from $S=k\log W$,
which connects the entropy scale to molecular masses) and $h$ (the elementary quantum of action,
with units of energy times time). A finite $h$ means energy exchange is not continuous —
classical physics is the $h\to 0$ limit. The radiation law and the hypothesis $\varepsilon=h\nu$ are
the result; energy is counted, not poured.


