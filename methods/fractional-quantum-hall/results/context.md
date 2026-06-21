## Research question

What state of a two-dimensional electron gas can produce a Hall plateau at a fractional filling of the lowest Landau level? The integer effect has a natural one-electron explanation: Landau levels form, disorder localizes states in the tails, and when an integer number of extended Landau-level states is filled the Hall resistance is quantized while longitudinal resistance vanishes. At filling `nu = 1/3`, the lowest Landau level is only partially occupied, so there is no one-electron Landau gap available to set the scale of an incompressible transport state.

The setting is to describe a partially filled Landau level that nonetheless gives a Hall conductance `sigma_xy = nu e^2/h` at fractional `nu`, with attention to the nature of its low-lying charged excitations.

## Background

A high magnetic field quantizes the kinetic energy of a two-dimensional electron into Landau levels. In a sample of area `A`, the degeneracy of one Landau level is the number of flux quanta through the sample,

`N_phi = BA/Phi_0`, with `Phi_0 = h/e`,

so the filling factor is `nu = N/N_phi`. The magnetic length `ell = sqrt(hbar/(eB))` fixes the area scale of a lowest-Landau-level orbital. In the extreme quantum limit, the kinetic energy is frozen and all low-energy structure inside a partially filled lowest Landau level comes from electron-electron interaction and the antisymmetry of the many-electron wavefunction.

The integer effect uses the Pauli principle and Landau-level gaps. At `nu = 1`, there is one flux quantum per electron, and filling the level leaves no low-energy one-electron rearrangement. Disorder broadens the levels and localizes tail states, letting the plateau persist over a finite interval of magnetic field.

At fractional filling, there is room inside the same Landau level. Electrons can lower Coulomb repulsion by arranging their relative motion so that other electrons are kept away. Störmer's Nobel account describes this as an electron-correlation effect: unlike the integer effect, it is not understood from independent electrons in a magnetic field. Laughlin's Nobel account makes the same point in phase language: a fractional plateau is not adiabatically deformable to a noninteracting electron state, because that would force integer Hall conductance.

## Baselines

The classical Hall effect gives `R_xy = B/(ne)`. It measures carrier density and charge.

The integer quantum Hall picture fills whole Landau levels. Its conductance is pinned by flux insertion: adiabatically inserting one flux quantum transfers one electron per filled Landau level across the sample. This explains `sigma_xy = i e^2/h`, with integer `i`, and disorder-localized states explain plateau width. The state is adiabatically connected to noninteracting electrons, so the pumped object has electron charge and the result is integral.

A Wigner crystal is the strong-repulsion guess at low density: localize electrons in a static lattice to minimize Coulomb energy.

Mean-field or independent-particle filling of a partially occupied Landau level treats the lowest-Landau-level degeneracy as independent orbital occupancy: each electron occupies one of the available single-particle orbitals.

## Evaluation settings

The natural setting is a high-mobility GaAs/AlGaAs two-dimensional electron system at very low temperature and high magnetic field, with longitudinal and Hall resistances measured as functions of field. The decisive regime is the extreme quantum limit, where the lowest Landau level is partially filled. The clean diagnostic is a plateau in `R_xy` accompanied by a minimum or vanishing of longitudinal resistance at a fractional filling such as `nu = 1/3`.

The theoretical setting is fixed too. A candidate state lives in the lowest Landau level, is antisymmetric for electrons, and has uniform density at the target filling. Its energetics are judged by the Coulomb interaction, and its transport by the flux-insertion argument for the Hall conductance.

## Code framework

The field-appropriate scaffold is a many-body wavefunction and mechanism, not software.

Start with complex coordinates for electrons in the plane:

`z_j = x_j + i y_j`.

Use lowest-Landau-level analytic structure: a many-electron state has an analytic polynomial in the `z_j` times the Gaussian orbital factor,

`Psi(z_1,...,z_N) = P(z_1,...,z_N) exp[-sum_j |z_j|^2/(4 ell^2)]`.

The open slot is the polynomial `P`, which sets the filling and the correlations of the state.

```text
Given:
  N electrons in the lowest Landau level
  N_phi magnetic flux quanta
  filling nu = N / N_phi
  magnetic length ell

Find:
  analytic antisymmetric polynomial P(z_1,...,z_N)
  for Psi = P * Gaussian at fractional filling

Check:
  density fixes nu
  Coulomb energy of the state
  charge of a localized excitation under flux insertion
```
