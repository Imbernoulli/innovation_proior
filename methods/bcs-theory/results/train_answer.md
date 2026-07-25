The problem is to explain why a metal below a critical temperature loses all electrical resistance, expels magnetic fields, and develops a gap in its low-energy electronic excitations. The normal-state picture of independent Bloch electrons cannot account for any of this: it has gapless single-particle states arbitrarily close to the Fermi surface, and merely reducing scattering would improve conductivity but would not produce a sharp thermodynamic transition or perfect diamagnetism. A first hint comes from Cooper's two-electron calculation, which shows that two electrons above an inert Fermi sea can form a bound state if their interaction is weakly attractive. That bound state, however, is enormous in real space, so in any real superconductor an enormous number of pairs overlap. A simple dilute-gas picture of independent molecules is therefore the wrong next step.

What is needed is a many-electron state that can exploit the attractive matrix elements coherently without violating Fermi statistics. The key obstacle is that random occupation configurations make the off-diagonal pair-scattering terms cancel. The way out is to restrict the allowed configurations: states are occupied in time-reversed pairs, so that if the single-particle state with momentum k and spin up is occupied, the state with momentum -k and spin down is also occupied. With this pairing constraint, the scattering terms connect different configurations with a common sign and the interaction energy can be lowered collectively.

The method is Bardeen-Cooper-Schrieffer (BCS) theory. It constructs a variational ground state in which every near-Fermi pair state is a superposition of empty and doubly occupied amplitudes, u_k and v_k. The trial state is the product over one representative of each time-reversed pair,

|Ψ⟩ = ∏_k (u_k + v_k c†_{k↑} c†_{-k↓}) |0⟩,

with u_k² + v_k² = 1. The reduced Hamiltonian keeps only the kinetic energy and the attractive pair-scattering channel,

H_red = ∑_{k,σ} ε_k c†_{kσ} c_{kσ} − ∑_{k,k′} V_{kk′} c†_{k↑} c†_{-k↓} c_{-k′↓} c_{k′↑}.

Minimizing the expectation value of this Hamiltonian gives the self-consistent gap equation. For a constant attraction V inside a shell |ε| < ℏω around the Fermi surface, the quasiparticle spectrum becomes E_k = √(ε_k² + Δ²), and the gap Δ is determined by

1 = N(0)V ∫_0^{ℏω} tanh(√(ε² + Δ²)/(2 k_B T)) / √(ε² + Δ²) dε.

The same variational parameter Δ that mixes the pair occupancy also serves as the order parameter. In the weak-coupling limit the transition temperature and zero-temperature gap satisfy

k_B T_c ≈ 1.14 ℏω exp(−1/(N(0)V)),
2Δ(0)/(k_B T_c) ≈ 3.50.

Because the lowest quasiparticle energy is Δ, there is a minimum energy cost to create an excitation, which explains the observed gap in single-particle and pair-breaking probes. Finally, the paired state is rigid against long-wavelength electromagnetic perturbations: the paramagnetic current no longer cancels the diamagnetic term, giving the London rigidity at short distances and the nonlocal Pippard response at finite coherence lengths. This closes the loop from a weak microscopic attraction to the macroscopic electrodynamics of superconductivity.

Collected into one statement, this is the deliverable: for a constant attractive matrix element $V$ acting within a shell $|\varepsilon_k| < \hbar\omega$ around the Fermi surface, with $N(0)$ the single-spin density of states at the Fermi level, the paired ground state

$$|\Psi\rangle = \prod_k \left(u_k + v_k\, c^\dagger_{k\uparrow} c^\dagger_{-k\downarrow}\right)|0\rangle, \qquad u_k^2 + v_k^2 = 1,$$

minimizes the reduced Hamiltonian

$$H_{\mathrm{red}} = \sum_{k,\sigma} \varepsilon_k\, c^\dagger_{k\sigma} c_{k\sigma} \;-\; \sum_{k,k'} V_{kk'}\, c^\dagger_{k\uparrow} c^\dagger_{-k\downarrow}\, c_{-k'\downarrow} c_{k'\uparrow}$$

when the occupation amplitudes and quasiparticle energies satisfy

$$u_k^2 = \frac{1}{2}\left(1 + \frac{\varepsilon_k}{E_k}\right), \qquad v_k^2 = \frac{1}{2}\left(1 - \frac{\varepsilon_k}{E_k}\right), \qquad E_k = \sqrt{\varepsilon_k^2 + \Delta^2},$$

with the gap $\Delta$ fixed self-consistently by

$$1 = N(0)\,V \int_0^{\hbar\omega} \frac{\tanh\!\left(\sqrt{\varepsilon^2+\Delta^2}\,/\,2k_BT\right)}{\sqrt{\varepsilon^2+\Delta^2}}\, d\varepsilon.$$

In the weak-coupling limit this reduces to two parameter-free numbers that make the theory falsifiable:

$$k_BT_c \approx 1.14\, \hbar\omega\, e^{-1/N(0)V}, \qquad \frac{2\Delta(0)}{k_BT_c} \approx 3.50.$$

This self-consistent gap equation, together with the universal ratio it forces regardless of the coupling strength, is the theory's final content: a microscopic pairing instability that produces a temperature-dependent excitation gap and reduces, in the appropriate limits, to the observed thermodynamics and electrodynamics of the superconducting state.
