The transport of a quantum excitation in a disordered lattice is usually explained by adding scattering to an otherwise diffusive picture: random onsite energies broaden Bloch waves, shorten the mean free path, and leave a random walk at long times. That picture works when disorder is weak and the extended plane-wave basis remains approximately correct, but it assumes the very thing that needs to be proved in the strongly disordered closed system. In a lattice with no external bath, each site has its own random energy, and energy-conserving hops to neighbors become rare events. Averaging over the disorder then gives a misleading answer, because rare resonant denominators dominate the mean even though a typical selected site sits in a local environment that is off-resonant from almost every neighbor. Bloembergen-Portis spin diffusion, impurity scattering, and classical percolation all fail in the same direction: they import diffusive transport as an assumption rather than deriving whether exact eigenstates spread or stay localized.

What is needed is a criterion that starts from localized sites, treats the hopping as a perturbation, and asks whether the resulting path expansion converges for a typical sample. The key diagnostic is the local resolvent for an initial state on a single site. If the self-energy acquires a continuous imaginary part as the resolvent approaches the real axis, the initial amplitude decays and transport exists. If instead the local resolvent retains isolated poles with finite residue, the eigenstate keeps its memory of the starting site and transport is absent. The question is therefore whether disorder and path proliferation can be balanced by the small probability of resonant energy denominators.

The method is Anderson localization. It studies the random tight-binding Hamiltonian in the site basis rather than in the extended basis. For a site j with random onsite energy E_j and hopping V_jk to other sites, the amplitudes obey i d a_j/dt = E_j a_j + sum_k V_jk a_k, or equivalently H = sum_j E_j |j><j| + sum_{j != k} V_jk |j><k|. The central object is the local resolvent G_00(z) = <0|(z-H)^{-1}|0|. Expanding G_00 in hopping around the localized basis gives a locator series whose terms are products of matrix elements divided by energy denominators along self-avoiding paths starting and ending at site 0. Each path of length L contributes a factor whose magnitude is controlled by how nearly resonant the visited sites are. Because there are roughly K^L such paths for effective connectivity K, the series converges only if the typical size of a length-L path term falls off faster than the number of paths grows.

When the criterion holds, the local resolvent has isolated poles on the real axis, the exact eigenfunctions have localized envelopes with a finite localization length xi, and a local initial excitation does not diffuse away. The localized state is not a classical particle trapped in a single well; it is a coherent superposition built from many off-resonant virtual hops, with rare resonant clusters carrying most of the weight. Lowering the disorder, increasing the connectivity, or moving to an energy with more favorable denominators can make the path series diverge, signaling extended states and a mobility edge between localized and extended spectral regions. The decisive point is to evaluate the probability distribution of locator terms for a typical sample, not the ensemble average, because averages are dominated by the rare resonances that do not describe the behavior of a specific localized packet.

The practical implementation below builds a one-dimensional Anderson chain, diagonalizes it, and checks whether eigenstates are localized by computing the inverse participation ratio. In the localized phase, almost every eigenstate concentrates on a few sites and the IPR averaged over the spectrum is large and independent of system size. In an extended phase, the IPR scales inversely with system size.

```python
import numpy as np

def anderson_hamiltonian(n, w, t=1.0, seed=None):
    rng = np.random.default_rng(seed)
    disorder = rng.uniform(-w/2, w/2, size=n)
    h = np.diag(disorder)
    for i in range(n - 1):
        h[i, i + 1] = t
        h[i + 1, i] = t
    return h

def ipr(state):
    p = np.abs(state)**2
    return np.sum(p**2)

def localization_diagnostic(n, w, t=1.0, seed=None):
    h = anderson_hamiltonian(n, w, t, seed)
    energies, vectors = np.linalg.eigh(h)
    iprs = np.array([ipr(vectors[:, i]) for i in range(n)])
    return energies, iprs

if __name__ == "__main__":
    n = 400
    t = 1.0
    for w in [0.5, 4.0, 10.0]:
        energies, iprs = localization_diagnostic(n, w, t, seed=0)
        print(f"W={w:.1f}, mean IPR={np.mean(iprs):.4f}, median IPR={np.median(iprs):.4f}")
```
