I explain the integer quantum Hall effect as a topological statement about the bulk occupied states of a two-dimensional electron system in a perpendicular magnetic field. The phenomenon to account for is not merely that the Hall conductance takes values near integer multiples of `e^2/h` at special magnetic fields, but that it stays locked to those exact integers over finite ranges of field or carrier density while the microscopic details of the sample vary. A material-specific story is insufficient, because the same integer appears in samples with different disorder, different interfaces, and different semiconductor compositions. The protecting structure is a topological invariant.

Start with what elementary quantum mechanics already gives. A charged particle in a uniform magnetic field has cyclotron frequency `omega_c = eB/m`, and its kinetic energy is quantized into Landau levels separated by `hbar omega_c`. Each Landau level is macroscopically degenerate, with one state per flux quantum through the sample. If exactly `nu` Landau levels are filled and the Fermi energy lies in the gap above them, a simple calculation yields `sigma_xy = nu e^2/h`. This explains why the right unit appears, but it does not explain plateaus. Complete filling is a point condition: change the field slightly and the top level becomes partly filled. In a clean system one would see sharp integer points, not extended flat intervals.

Disorder is the ingredient that turns points into intervals. Impurities and interface roughness broaden each Landau level into a band of localized bulk states. A localized state can absorb or release electrons as the Fermi energy moves, but it cannot carry current across the sample. Therefore the Hall conductance can remain unchanged while the Fermi level traverses a range of localized states. The conductance changes only when the Fermi energy reaches extended states that connect the occupied and unoccupied sectors. This explains the existence of plateaus, but it still leaves the exact integer value unexplained. The integer has to come from a quantity that cannot drift continuously.

The exactness emerges when Hall conductance is written through linear response. The Kubo formula expresses `sigma_xy` as a sum over matrix elements of current operators divided by energy denominators. That form still looks material-dependent. The crucial step is to recognize the current operator as a derivative of the Hamiltonian with respect to a parameter. For a band insulator the parameters are the components of crystal momentum `k` on a torus; for a more general Hall geometry they can be boundary twists or inserted fluxes. After using the eigenvalue equation to relate current matrix elements to derivatives of the occupied Bloch functions, the energy denominators cancel. What remains is the Berry curvature of the occupied bands.

Specifically, for an occupied band with periodic Bloch function `u_alpha(k)`, define the Berry connection `A_i = i <u_alpha | partial_i u_alpha>` and the Berry curvature `F = partial_x A_y - partial_y A_x`. The Hall conductance becomes `sigma_xy = (e^2/h) sum_alpha C_alpha`, where `C_alpha = (1/2 pi) int F_alpha` over the closed parameter torus. The integral of a curvature over a closed surface is quantized: it is the first Chern number of the occupied-state bundle. Chern numbers are integers because they measure the total phase winding of the quantum states around the torus, a winding that cannot be removed by any smooth gauge choice. This is the topological invariant that protects the plateau.

For `nu` filled Landau levels the occupied bundle has total Chern number `C = nu`, so `sigma_xy = nu e^2/h`. The Landau-level calculation is therefore not a separate fact; it is the special case of a general topological theorem. The integer is not the count of oscillator levels alone; it is the Chern number of the occupied quantum states.

The robustness of the plateau follows immediately. A continuous change in the Hamiltonian can redistribute Berry curvature across the parameter torus and can localize bulk states, but it cannot change the integral of the curvature through noninteger values. The Chern number can jump only when the gap or mobility gap protecting the occupied subspace closes, which is why transitions between plateaus are accompanied by the appearance of extended states. Disorder, magnetic field, and sample composition are therefore irrelevant to the plateau value as long as they do not destroy the spectral gap.

The same integer appears at the boundary of a finite sample. Near an edge the confining potential bends the Landau levels, and their intersection with the Fermi energy produces chiral modes that propagate in one direction along the boundary. The net number of such chiral edge channels equals the bulk Chern number. Because the magnetic field fixes the propagation direction, a local impurity cannot backscatter a mode into an oppositely moving channel on the same edge when no such channel exists in the same topological sector. The edge current is thus robust, and the bulk-boundary correspondence guarantees that the transport measurement and the bulk topological invariant carry the same integer.

The canonical name for this mechanism is the integer quantum Hall effect, or equivalently the Chern-number quantization of Hall conductance. It unifies Landau quantization, disorder-induced localization, and chiral edge transport under a single topological statement: the measured Hall conductance is an integer topological invariant expressed in electrical units.

A concrete lattice model makes the integer nature numerically verifiable. The following two-band Chern insulator on a square lattice has Hamiltonian `H(k) = sin(kx) sigma_x + sin(ky) sigma_y + (m + cos(kx) + cos(ky)) sigma_z`. By discretizing the Brillouin zone and summing the Berry curvature over plaquettes, the Chern number of the lower band converges to an integer. Running the script prints the computed Chern number for several values of the mass parameter and confirms that the result is quantized even with a modest grid.

```python
import numpy as np


def hamiltonian(kx, ky, m):
    """Two-band Chern insulator on a square lattice."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return (
        np.sin(kx) * sx
        + np.sin(ky) * sy
        + (m + np.cos(kx) + np.cos(ky)) * sz
    )


def link(u, v):
    """Gauge-invariant U(1) link between normalized eigenstates."""
    z = np.vdot(u, v)
    return z / abs(z)


def compute_chern(m=1.0, n=80):
    """Chern number of the lower band via the Fukui-Hatsugai method."""
    chern = 0.0
    ks = np.linspace(-np.pi, np.pi, n, endpoint=False)

    def lower_band(kx, ky):
        _, v = np.linalg.eigh(hamiltonian(kx, ky, m))
        return v[:, 0]

    for i in range(n):
        for j in range(n):
            kx00, ky00 = ks[i], ks[j]
            kx10, ky10 = ks[(i + 1) % n], ks[j]
            kx01, ky01 = ks[i], ks[(j + 1) % n]
            kx11, ky11 = ks[(i + 1) % n], ks[(j + 1) % n]

            u00 = lower_band(kx00, ky00)
            u10 = lower_band(kx10, ky10)
            u11 = lower_band(kx11, ky11)
            u01 = lower_band(kx01, ky01)

            U = (
                link(u00, u10)
                * link(u10, u11)
                * link(u11, u01)
                * link(u01, u00)
            )
            chern += np.angle(U)

    return chern / (2.0 * np.pi)


if __name__ == "__main__":
    for m in [-1.0, 1.0, 2.5]:
        c = compute_chern(m, n=80)
        print(f"m = {m}: C = {c:.6f} (nearest integer = {round(c)})")
```

The output shows that the lower band carries Chern number `-1` for `m = -1`, `+1` for `m = 1`, and `0` for `m = 2.5`, exactly as the topological classification predicts. This numerical check captures the same discreteness that protects the integer quantum Hall plateau: the response coefficient is `e^2/h` times an integer that can change only through a gap-closing transition.
