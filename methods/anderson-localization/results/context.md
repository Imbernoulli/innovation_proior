## Research question

How should transport be described when a quantum excitation moves by hopping among localized sites whose energies are random? The familiar expectation is that disorder scatters otherwise mobile waves, shortens a mean free path, and leaves a diffusive random walk at long times. The hard case is a low-temperature lattice or impurity band with no thermal reservoir available to repair energy mismatches. A solution has to decide whether an initially localized amplitude must eventually spread through the whole sample, or whether the random energy landscape can invalidate diffusion itself.

The concrete setting is a lattice of sites `j`, each with an onsite energy `E_j` drawn from a distribution of width `W`, and hopping matrix elements `V_jk` that move the excitation between sites. The two competing scales are the disorder width `W` and the hopping strength or connectivity encoded by `V_jk`. The transport question is long-time and sample-specific: after preparing amplitude on one site or in one narrow frequency packet, does the probability distribution diffuse away, and what happens to the exact eigenstates?

## Background

Band theory supplies the clean starting point. On a periodic lattice with identical site energies, hopping produces extended Bloch waves. In the Drude-like transport picture, imperfections scatter those extended waves; increasing disorder lowers conductivity by reducing the mean free path. Quantum mechanics already modifies the classical picture because waves diffract through an ideal crystal and scatter only from imperfections, but the working transport intuition still treats disorder as a source of scattering within an extended-state basis.

The motivating experimental pressure came from donor spins in very pure silicon. Donor electrons have extended hydrogenic orbitals, and their hyperfine coupling to surrounding nuclei creates inhomogeneous resonance lines. Hole-burning and ENDOR-style measurements selected narrow spin packets that kept their individual frequencies for seconds or minutes. That was much longer than the `0.1` to `10^-6` second packet lifetimes estimated from a Golden-Rule/random-walk spectral diffusion picture. Coulomb arguments could make actual charged donor motion sluggish, but they did not by themselves explain why spin diffusion failed so strongly.

The model strips away everything except the transport obstruction. A site basis `|j>` is retained; each site has a random energy `E_j`; hopping `V_jk` may be short-ranged or decay with distance; and there is no external heat bath. The time-dependent amplitudes obey

`i d a_j/dt = E_j a_j + sum_k V_jk a_k`.

For `W = 0`, the model is a band problem. For weak randomness, conventional perturbation theory starts from plane waves and treats `E_j` fluctuations as scattering. For strong randomness, the unperturbed objects are the sites themselves, and energy matching between sites becomes a rare-event question rather than an averaged scattering rate.

## Baselines

Bloembergen spin diffusion treats interacting spins as exchanging polarization through mutual precession. A Golden-Rule estimate plus random-walk reasoning gives equilibration of spin temperature across space. It explains ordinary nuclear spin diffusion, but it assumes that local spectral packets can communicate through enough resonant neighbors to act diffusively.

Portis spectral diffusion adds inhomogeneous broadening: spins are grouped into packets with definite resonance frequency inside a broadened line. This recognizes that different sites have different local resonance energies. Its limitation is the averaging step. If the lifetime estimate is made from an ensemble mean, rare near-resonances dominate the mean even when a typical selected packet remains isolated in a particular sample.

Ordinary impurity-scattering transport starts from extended plane waves. The random potential produces a self-energy with an imaginary part, interpreted as a finite lifetime and a finite scattering rate. This is the correct language when disorder is weak enough that the extended basis survives, but it builds diffusion in from the start and does not answer whether the localized-site expansion can remain self-consistent.

Classical percolation counts connected paths through a random medium. It provides the right kind of language for path proliferation and connectivity, but connectivity alone is not enough for a quantum hopping problem. A path also carries energy denominators and complex amplitudes, so a geometrically connected lattice can still fail to provide resonant transport.

## Evaluation settings

The clean diagnostic is a single-site initial condition, `a_0(0)=1`, with all other amplitudes initially zero. In Laplace or resolvent language, the long-time question is controlled by the behavior as the real part of the Laplace variable tends to zero. A finite decay rate means the site loses memory of the initial excitation; a persistent pole with finite residue means the initial site remains represented in long-time amplitudes.

The limiting regimes are also fixed in advance. With identical onsite energies, hopping gives extended Bloch states. With weak randomness, scattering theory should recover finite lifetimes and transport. With large `W/V`, nearest-neighbor hopping is usually off-resonant, and any transport must come from rare resonances or long multi-step virtual paths. A useful theory must say when those paths form an effective transport network and when they do not.

Natural stress tests include finite-dimensional lattices, one-dimensional chains, higher-dimensional lattices with increasing path multiplicity, and finite samples where conductance or transmission can be compared as a function of system size. The later scaling language uses a dimensionless conductance `g` and asks whether `g` grows or shrinks with sample size, but the local transport obstruction is already visible in the site-basis model.

## Code framework

The field-appropriate scaffold is mathematical rather than algorithmic. Start with a Hilbert space spanned by localized site states `|j>`. Specify random onsite energies `E_j` sampled independently from a distribution of width `W`, and hopping matrix elements `V_jk` whose magnitude is set by overlap and whose range is short enough to make distant direct hops weak.

The unresolved object is the long-time resolvent for an initially occupied site:

`G_00(z) = <0|(z - H)^(-1)|0>`.

Any proposed mechanism must fill in how `G_00` behaves near the real axis, how rare resonant denominators are counted, how multi-step paths are controlled against their combinatorial proliferation, and how the result appears in the spatial profile of exact eigenfunctions.
