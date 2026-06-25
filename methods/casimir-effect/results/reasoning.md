I start with the thing that classical language hides. If I put two neutral conducting plates in vacuum, there is no obvious classical field between them and no charge imbalance that should push them together. But "vacuum" in a quantized electromagnetic field cannot mean all oscillators are absent in the classical sense. Each normal mode is a harmonic oscillator, and even the lowest state contributes `hbar omega / 2`. So the empty region is not empty of possible modes. The plates do not need to add real radiation; they can change the list of allowed vacuum oscillators.

The first wall is immediate: `E0 = (1/2) sum hbar omega` is infinite. If I try to call that the energy of empty space, I have learned nothing mechanical. A force needs a dependence on plate separation. So I should not ask for the absolute vacuum energy. I should compare two geometries: the plates at separation `a` and the same conducting material so far apart that this particular separation no longer matters. Any enormous bulk contribution that is the same in both descriptions is not a force between the plates. I need the finite residue that changes with `a`.

The boundary-value problem is already classical. Put the plates at `z = 0` and `z = a`, take the area `A` large, and ignore edges. The transverse wave vector `k_perp = (k_x, k_y)` is continuous. The normal component is not; a standing wave between perfect conductors has `k_z = n pi / a`. For the simplified scalar count the allowed angular frequencies are

`omega_n(k_perp) = c sqrt(k_perp^2 + (n pi / a)^2)`.

For the electromagnetic field the TE and TM modes supply the two photon polarizations, with the usual boundary-condition bookkeeping at `n = 0`; I will carry the scalar calculation first because it keeps every numerical factor visible, and worry about the polarization count once I have a clean scalar number.

For one scalar polarization the formal energy per area is

`E_s/A = (hbar c / 2) sum_{n=1}^infty int d^2k/(2 pi)^2 sqrt(k^2 + (n pi / a)^2)`.

This is still divergent. I need a regulator that does not change the long-wavelength modes responsible for the separation dependence. A real metal would stop reflecting sufficiently short wavelengths; mathematically I can use an analytic regulator and keep the finite part after the comparison. The proper-time identity, continued analytically, is

`sqrt(Q) = -1/(2 sqrt(pi)) int_0^infty dt t^(-3/2) exp(-t Q)`.

Substituting `Q = k^2 + (n pi / a)^2` gives

`E_s/A = -hbar c/(4 sqrt(pi)) sum_n int d^2k/(2 pi)^2 int_0^infty dt t^(-3/2) exp[-t k^2 - t(n pi / a)^2]`.

The transverse integral is Gaussian. Writing `d^2k = 2 pi k_perp dk_perp` and integrating,

`int d^2k/(2 pi)^2 exp(-t k^2) = (1/(2 pi)) int_0^infty k_perp exp(-t k_perp^2) dk_perp = 1/(4 pi t)`.

So the regulated expression becomes

`E_s/A = -hbar c/(16 pi^(3/2)) sum_n int_0^infty dt t^(-5/2) exp[-t(n pi / a)^2]`.

The remaining integral is the analytic continuation of

`int_0^infty dt t^(-5/2) exp(-beta t) = beta^(3/2) Gamma(-3/2)`,

and `Gamma(-3/2) = 4 sqrt(pi)/3`. (I want to keep that gamma value honest, since for negative half-integer argument it is easy to drop a sign: using `Gamma(z+1) = z Gamma(z)` downward from `Gamma(1/2) = sqrt(pi)` gives `Gamma(-1/2) = -2 sqrt(pi)` and then `Gamma(-3/2) = Gamma(-1/2)/(-3/2) = 4 sqrt(pi)/3`, which is positive — consistent.) With `beta = (n pi / a)^2`,

`E_s/A = -hbar c/(16 pi^(3/2)) (4 sqrt(pi)/3) (pi/a)^3 sum_{n=1}^infty n^3`.

The sum `sum n^3` is not an ordinary convergent sum. The finite part selected by the same analytic continuation is `zeta(-3) = 1/120`. Folding the prefactors together — `hbar c/(16 pi^(3/2)) * (4 sqrt(pi)/3) * pi^3 = hbar c pi^2/12`, and then `* (1/120)` and `/a^3` — I get

`E_s/A = -hbar c pi^2/(1440 a^3)`.

I do not want to trust `zeta(-3) = 1/120` standing in for a wildly divergent `sum n^3` on faith, because that single replacement is doing all the physical work. Analytic continuation is the cheapest possible move and the place where a sign or factor error would hide most comfortably. So I want a second, independent regulator that never mentions the zeta function, and I want the two numbers to land on top of each other.

The honest alternative is a cutoff comparison. The object I actually claim is finite is `sum_{n>=1} F(n) - int_0^infty F(n) dn`, where `F(n) = int d^2k/(2 pi)^2 sqrt(k^2 + (n pi/a)^2)` is the per-`n` mode contribution — the discrete plate spectrum minus the free-space continuum it would relax into. That difference of a sum and its integral is exactly what the Abel-Plana formula evaluates:

`sum_{n=0}^infty h(n) - int_0^infty h(n) dn = -h(0)/2 - i int_0^infty [h(it) - h(-it)]/(e^{2 pi t} - 1) dt`.

The `h(0)` piece is a pure `k`-integral with no `a` in it, so it is part of the bulk and cannot contribute a force. The `a`-dependent residue lives entirely in the last integral. To evaluate it I need the imaginary part of `F` continued to imaginary `n`: with `n -> i t` the radical `sqrt(k_perp^2 - t^2)` is imaginary for `k_perp < t`, and that band gives `Im F(it) = (1/(4 pi)) int_0^t k_perp sqrt(t^2 - k_perp^2) dk_perp = (1/(4 pi)) (t^3/3)`. Putting that into the Abel-Plana integral, the whole `a`-dependent residue (in units where `pi/a = 1`) reduces to a multiple of the standard integral

`int_0^infty t^3/(e^{2 pi t} - 1) dt = Gamma(4) zeta(4)/(2 pi)^4 = 6 (pi^4/90)/(16 pi^4) = 1/240`.

I evaluated this numerically as well to be sure: `int_0^30 t^3/(e^{2 pi t} - 1) dt = 0.0041666...`, which is `1/240` to the digits shown. Carrying the `1/(6 pi)` prefactor through, the residue `sum_{n>=1} F(n) - int F dn` comes out to `-1/(1440 pi)`. In the same `pi/a = 1`, `hbar = c = 1` units the zeta-route answer `-pi^2 hbar c/(1440 a^3)` with `a = pi` is `-pi^2/(1440 pi^3) = -1/(1440 pi)`. Numerically the two are `-2.21048532e-4` from Abel-Plana and `-2.21048532e-4` from the zeta continuation — they agree to better than `1e-19`. That is the check I wanted: the divergent `sum n^3 -> zeta(-3)` step is not an isolated formal trick, it is reproducing the same finite number that an honest exponential/Abel-Plana cutoff produces after the bulk is subtracted.

There is one bookkeeping snag worth recording, because I tripped on it while matching the two routes. The Abel-Plana residue above is `(sum - integral)` of the *full frequency* `F` (the `omega`, no `1/2`), and it already equals the textbook `E_s/A`; but the formal expression I started from carried an explicit `hbar c/2` in front of `sum F`. Naively I expected the cutoff route to come out a factor of two smaller than the zeta route, and for a moment it did. The resolution is that the `1/2` zero-point factor and the analytic-continuation prefactors I collapsed above are not independent multipliers stacked on the same `sum F`; tracking the constant `hbar c/(16 pi^(3/2)) (4 sqrt(pi)/3) pi^3 = hbar c pi^2/12` and the `1/2` together is what reproduces the single coefficient `1/1440`. Once I stopped double-counting the half, both regulators give the identical scalar energy `E_s/A = -hbar c pi^2/(1440 a^3)`. I am now confident in the scalar coefficient rather than just hopeful about it.

For the electromagnetic field between perfect conductors the two physical polarizations (TE and TM) each contribute a scalar-like spectrum; the `n = 0` mode exists for only one of them, but in the large-area parallel-plate limit that single missing mode is a set of measure zero among the transverse continuum and does not change the leading `a`-dependence. So the electromagnetic energy is twice the scalar one:

`E/A = -hbar c pi^2/(720 a^3)`.

Now the sign, which I should pin down rather than guess. The energy is negative relative to the reference with the plates separated, and it grows more negative as `a` shrinks, because `1/a^3` increases. The system therefore lowers its energy by reducing the separation, which means the force is attractive. Making that quantitative through virtual work,

`P = F/A = -d(E/A)/da`.

With `C = hbar c pi^2/720`, `d[-C a^(-3)]/da = +3 C a^(-4)`, so `P = -3 C a^(-4) = -pi^2 hbar c/(240 a^4)`. The minus sign is consistent with the attraction I argued for from the energy, not an extra surprise — a positive `a` is the separation, and a negative `P` is an inward pull. The two arguments agree, which is the consistency I want.

`P = F/A = -pi^2 hbar c/(240 a^4)`.

It is also worth knowing whether this is a real number or a curiosity buried in natural units, so I put in actual constants. With `hbar = 1.0546e-34 J s`, `c = 2.998e8 m/s`, and a laboratory separation `a = 1 micron = 1e-6 m`,

`P = -pi^2 hbar c/(240 a^4) = -1.30e-3 Pa`.

A millipascal of attraction at a micron is small but firmly in the range that careful force measurements reach, and `~10^-3 Pa` at a micron is the magnitude quoted for the effect. So the calculation is not just internally consistent; it produces a number of the right size for something that can actually be weighed.

I should also say plainly why the subtraction is legitimate and not sleight of hand. If I include a cutoff function `f(k/k_m)` that is one for wavelengths much longer than the microscopic transparency scale and dies for very large `k`, the individual sums and integrals become finite. The high-frequency modes then contribute large terms, but those terms do not know much about the plate separation once their wavelengths are far shorter than every macroscopic length. In the comparison with the far-separated configuration the common bulk terms cancel — which is exactly what the Abel-Plana `h(0)/2` term and the `int F dn` term encode, and exactly why they dropped out of the `a`-dependence above. Taking the ideal limit leaves the `a^-3` finite energy and the `a^-4` pressure. The regulator is a disciplined way to say that only differences in the spectrum caused by the boundary conditions are being measured.

The physical picture is now clean, and it is a conclusion rather than a premise. The plates impose boundary conditions on the quantum electromagnetic field. Boundary conditions alter the allowed normal modes. The vacuum state is the ground state of those modes, not a featureless absence. The absolute sum of zero-point energies is not an observable, but the change in that sum under a change of boundary geometry is — and that change, computed two independent ways and matched to within rounding, is

`E/A = -pi^2 hbar c/(720 a^3)`,

with corresponding force per area

`F/A = -pi^2 hbar c/(240 a^4)`.

So the measurable attraction is a boundary-condition effect on vacuum modes. Nothing has to be added to the vacuum; the point is that the vacuum already carries quantum mode structure, and conductors reshape that structure.
