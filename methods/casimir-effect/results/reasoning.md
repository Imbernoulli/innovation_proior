I start with the thing that classical language hides. If I put two neutral conducting plates in vacuum, there is no obvious classical field between them and no charge imbalance that should push them together. But "vacuum" in a quantized electromagnetic field cannot mean all oscillators are absent in the classical sense. Each normal mode is a harmonic oscillator, and even the lowest state contributes `hbar omega / 2`. So the empty region is not empty of possible modes. The plates do not need to add real radiation; they can change the list of allowed vacuum oscillators.

The first wall is immediate: `E0 = (1/2) sum hbar omega` is infinite. If I try to call that the energy of empty space, I have learned nothing mechanical. A force needs a dependence on plate separation. So I should not ask for the absolute vacuum energy. I should compare two geometries: the plates at separation `a` and the same conducting material so far apart that this particular separation no longer matters. Any enormous bulk contribution that is the same in both descriptions is not a force between the plates. I need the finite residue that changes with `a`.

The boundary-value problem is already classical. Put the plates at `z = 0` and `z = a`, take the area `A` large, and ignore edges. The transverse wave vector `k_perp = (k_x, k_y)` is continuous. The normal component is not; a standing wave between perfect conductors has `k_z = n pi / a`. For the simplified scalar count the allowed angular frequencies are

`omega_n(k_perp) = c sqrt(k_perp^2 + (n pi / a)^2)`.

For the electromagnetic field the TE and TM modes supply the two photon polarizations, with the usual boundary-condition bookkeeping at `n = 0`; for this parallel-plate idealization the final electromagnetic pressure is twice the scalar result. I can do the scalar calculation first because it keeps the factor check visible.

For one scalar polarization the formal energy per area is

`E_s/A = (hbar c / 2) sum_{n=1}^infty int d^2k/(2 pi)^2 sqrt(k^2 + (n pi / a)^2)`.

This is still divergent. I need a regulator that does not change the long-wavelength modes responsible for the separation dependence. A real metal would stop reflecting sufficiently short wavelengths; mathematically I can use an analytic regulator and keep the finite part after the comparison. The proper-time identity, continued analytically, is

`sqrt(Q) = -1/(2 sqrt(pi)) int_0^infty dt t^(-3/2) exp(-t Q)`.

Substituting `Q = k^2 + (n pi / a)^2` gives

`E_s/A = -hbar c/(4 sqrt(pi)) sum_n int d^2k/(2 pi)^2 int_0^infty dt t^(-3/2) exp[-t k^2 - t(n pi / a)^2]`.

The transverse integral is Gaussian:

`int d^2k/(2 pi)^2 exp(-t k^2) = 1/(4 pi t)`.

Now the regulated expression becomes

`E_s/A = -hbar c/(16 pi^(3/2)) sum_n int_0^infty dt t^(-5/2) exp[-t(n pi / a)^2]`.

The remaining integral is the analytic continuation of

`int_0^infty dt t^(-5/2) exp(-beta t) = beta^(3/2) Gamma(-3/2)`,

and `Gamma(-3/2) = 4 sqrt(pi)/3`. With `beta = (n pi / a)^2`,

`E_s/A = -hbar c/(16 pi^(3/2)) (4 sqrt(pi)/3) (pi/a)^3 sum_{n=1}^infty n^3`.

The sum is not an ordinary convergent sum. The finite part selected by the same analytic continuation is `zeta(-3) = 1/120`, so

`E_s/A = -hbar c pi^2/(1440 a^3)`.

This is the scalar answer. For the electromagnetic field between perfect conductors, the two polarizations double it:

`E/A = -hbar c pi^2/(720 a^3)`.

I need to check the sign before trusting the result. The energy is negative relative to the reference with the plates separated. If `a` decreases, `1/a^3` grows and the energy becomes more negative. The system lowers its energy by reducing the separation, so the force must be attractive. Virtual work gives the pressure

`P = F/A = -d(E/A)/da`.

Since `d[-C a^(-3)]/da = 3C a^(-4)` with `C = hbar c pi^2/720`, the pressure is

`P = -pi^2 hbar c/(240 a^4)`.

The negative sign is not a mysterious negative energy pushing from nowhere. It says the force on a plate is inward when positive `a` is taken as separation. The finite part of the zero-point energy changes with geometry, and the derivative of that finite part is a mechanical pressure.

I should also make the subtraction less like a formal trick. If I include a cutoff function `f(k/k_m)` that is one for wavelengths much longer than the microscopic transparency scale and dies for very large `k`, the individual sums and integrals become finite. The high-frequency modes then contribute large terms, but those terms do not know much about the plate separation once their wavelengths are far shorter than every macroscopic length. In the comparison with the outside or far-separated configuration, the common volume terms cancel. Taking the ideal limit leaves exactly the `a^-3` finite energy and the `a^-4` pressure. The regulator is a disciplined way to say that only differences in the spectrum caused by the boundary conditions are being measured.

The physical picture is now clean. The plates impose boundary conditions on the quantum electromagnetic field. Boundary conditions alter the allowed normal modes. The vacuum state is the ground state of those modes, not a featureless absence. The absolute sum of zero-point energies is not an observable, but the change in that sum under a change of boundary geometry is. For two perfect parallel conductors, that change is

`E/A = -pi^2 hbar c/(720 a^3)`,

and the corresponding force per area is

`F/A = -pi^2 hbar c/(240 a^4)`.

So the measurable attraction is a boundary-condition effect on vacuum modes. Nothing has to be added to the vacuum; the point is that the vacuum already carries quantum mode structure, and conductors reshape that structure.
