# Context — the black-body spectrum problem

## Research question

A cavity with walls held at temperature $T$, pierced by a tiny hole, glows. The radiation that
leaks out has a spectral energy density $u(\nu,T)$ — energy per unit volume per unit frequency —
and Kirchhoff proved the decisive fact: this $u(\nu,T)$ is a **universal** function of
frequency and temperature alone. It does not depend on the material of the walls, their shape, or
their size; two black bodies in equilibrium at the same temperature have identical radiation fields.
That universality is a promise: somewhere there is a single law of nature, $u(\nu,T)$, waiting to be
written down.

The question is to find it. A solution must do three things at once. It must reduce, at high frequency
(short wavelength), to the exponential fall-off that experiment confirms there. It must reduce, at low
frequency (long wavelength, high temperature), to the linear-in-$T$ behaviour that newer infrared
measurements reveal. And it must be finite when integrated over all frequencies, because the total
radiated energy is finite and obeys the Stefan–Boltzmann $T^4$ law. No formula on the table
does all three. Finding the one that does — and understanding *why* it must have
the form it has — is the problem.

## Background

**Kirchhoff's universality and the Stefan–Boltzmann law.** Kirchhoff's theorem reduces the whole
problem to one universal function. Josef Stefan, reading Tyndall's platinum-filament data,
found empirically that the total emitted power scales as $T^4$. Ludwig Boltzmann derived this
thermodynamically: treating radiation as a gas exerting pressure $P=\tfrac13\rho_T$ (with $\rho_T$ the
total energy density) and using $dU=T\,dS-P\,dV$ with $U=\rho_T V$, the integrability of $dS$ forces
$d\rho_T/dT = 4\rho_T/T$, hence $\rho_T = aT^4$. The total energy is finite — any spectral law must
integrate to this.

**Wien's displacement (scaling) law.** Wilhelm Wien, combining thermodynamics with the Doppler
shift of radiation reflected off a slowly moving piston, showed the universal function is not free in
two variables but constrained to the form $u(\nu,T) = \nu^3\,f(\nu/T)$ — equivalently
$\rho(\lambda,T)=\lambda^{-5}\,\varphi(\lambda T)$, and $\lambda_{\max}T=\text{const}$. Only the single
combination $\nu/T$ (or $\lambda T$) can appear. This is a powerful constraint: it says any candidate
spectral law, and any expression for the mean energy or entropy of a single mode, must respect this
scaling.

**Wien's distribution law.** Wien went further and proposed an explicit form,
$\rho(\lambda,T)=C\,\lambda^{-5}\,e^{-c_2/\lambda T}$, equivalently $u\propto \nu^3 e^{-a\nu/T}$,
modelled on a Maxwell–Boltzmann velocity-distribution analogy. It satisfies the displacement law and
integrates to a $T^4$ total. Friedrich Paschen's short-wavelength infrared data fit it beautifully.
It looked like the answer.

**The diagnostic failure at long wavelengths.** New measurements broke it. Otto Lummer and Ernst
Pringsheim, and decisively Heinrich Rubens and Ferdinand Kurlbaum (working with the long-wavelength
"residual rays" of fluorite and rock salt, reaching far into the infrared), found systematic
deviations from Wien's law at large $\lambda$, high $T$. Where Wien predicts an exponential cutoff,
the measured intensity instead grows linearly with $T$. Re-expressed through the quantity
$R \equiv -1/(d^2S/dU^2)$ for a single resonator (the positive reciprocal of the negative second
derivative of its entropy with respect to its energy), the
data say: $R\propto U$ for small energy (the Wien regime), but $R\propto U^2$ for large energy (the
new long-wave regime). Two clean empirical limits, in conflict with any single exponential.

**The classical mode-counting law and its divergence.** Independently, treating cavity radiation as
electromagnetic standing waves, one counts modes: the number per unit volume with frequency in
$[\nu,\nu+d\nu]$ is $8\pi\nu^2/c^3\,d\nu$ (two polarizations, mode-number sphere). Classical
equipartition assigns each mode a mean energy $kT$. The product gives $u = 8\pi\nu^2 kT/c^3$, linear
in $T$ and matching the long-wave data — but it grows without bound as $\nu\to\infty$, so
$\int_0^\infty u\,d\nu$ **diverges**. A purely classical equipartition spectrum cannot be the answer:
it has no finite total energy and contradicts Stefan–Boltzmann.

**Boltzmann's bridge between entropy and counting.** The other load-bearing idea is statistical: for a
system whose macrostate can be realized in $W$ microscopic ways, the entropy is $S = k\log W$ (within
an additive constant). Entropy is a count of arrangements. This is the tool that, applied to the
resonators of the cavity wall, could in principle give a *physical* entropy rather than a guessed one.

**The resonator bridge.** The cavity walls can be modelled, following Hertz, as a collection of charged
linear oscillators ("resonators") of every frequency, exchanging energy with the field. A
material-independent electromagnetic result connects a resonator's mean energy $U$ at frequency $\nu$
to the surrounding spectral density: $u = (8\pi\nu^2/c^3)\,U$. So finding $u(\nu,T)$ reduces to finding
the mean energy $U$ of one resonator, which (through $1/T = dS/dU$) reduces to finding its entropy
$S(U)$ as a function of its energy.

## Baselines

**Wien's distribution law.** $u(\nu,T)=C\nu^3 e^{-a\nu/T}$, equivalently
$\rho=C'\lambda^{-5}e^{-c_2/\lambda T}$. Core idea: a Maxwell–Boltzmann
analogy for the radiation, consistent with Wien's displacement law. *Gap:* fits short waves, fails at
long waves — the Rubens–Kurlbaum deviations. In resonator terms it corresponds to $R\propto U$, i.e.
$d^2S/dU^2 = -\alpha/U$, giving an entropy that is wrong for large $U$.

**The classical equipartition / standing-wave law.** Mode density
$8\pi\nu^2/c^3$ times equipartition energy $kT$ gives $u\propto T\lambda^{-4}$. Core idea: count
electromagnetic standing-wave modes in a box, give each the classical thermal energy $kT$. *Gap:*
matches the long-wave data but diverges at high frequency — no finite total energy, violating
Stefan–Boltzmann. It corresponds to $R\propto U^2$ alone, the opposite extreme from Wien.

**Thermodynamic interpolation via $R(U)$.** Given the two limits $R\propto U$ (Wien) and $R\propto U^2$
(long-wave), one can write the simplest law containing both, $R=U(U+\beta)/\alpha$, i.e.
$d^2S/dU^2 = -\alpha/[U(U+\beta)]$, integrate, and impose Wien scaling. This produces a spectral
formula of the right shape. *Gap:* it is an interpolation — it matches the limiting regimes but carries no
physical account of *why* entropy should take that form. A formula without a mechanism.

## Evaluation settings

The yardsticks are the spectral measurements of cavity ("black-body") radiation available at the time:
Lummer and Pringsheim, and Rubens and Kurlbaum, mapping $u(\nu,T)$ across the visible and well into the
infrared (the long-wavelength residual rays of fluorite and rock salt), over a range of cavity
temperatures (roughly $10^2$ to $10^3$ K). The relevant observable is the spectral energy density as a
function of frequency/wavelength at fixed temperature, together with the integrated total (the
Stefan–Boltzmann $T^4$ constraint) and the displacement of the spectral peak ($\lambda_{\max}T=$ const).
A candidate law is judged by whether a single functional form fits the short-wave exponential fall-off,
the long-wave linear-in-$T$ rise, and the finite $T^4$ total simultaneously, with frequency-independent
universal constants.


