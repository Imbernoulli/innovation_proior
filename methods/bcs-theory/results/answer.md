The final construction is a variational many-electron state built from coherent occupation of time-reversed pair states. With `V_{k k'} > 0` denoting the net attractive matrix element, keep only the near-Fermi pair-scattering channel

```text
H_red = sum_{k,sigma} epsilon_k c^dagger_{k sigma} c_{k sigma}
        - sum_{k,k'} V_{k k'} c^dagger_{k up} c^dagger_{-k down}
          c_{-k' down} c_{k' up}.
```

Use the paired trial state, where the product takes one representative from each time-reversed pair:

```text
|Psi> = product_k (u_k + v_k c^dagger_{k up} c^dagger_{-k down}) |0>,
u_k^2 + v_k^2 = 1.
```

The self-consistent pair field is

```text
Delta_k = sum_{k'} V_{k k'} u_{k'} v_{k'} (1 - 2 f_{k'}).
```

For a constant attractive interaction `V` in a shell `|epsilon| < hbar omega`, using `N(0)` as the single-spin density of states at the Fermi surface, minimization gives

```text
E_k = sqrt(epsilon_k^2 + Delta^2),
u_k^2 = (1 + epsilon_k / E_k) / 2,
v_k^2 = (1 - epsilon_k / E_k) / 2,
1 = N(0) V int_0^{hbar omega}
    tanh(sqrt(epsilon^2 + Delta^2) / 2 k_B T)
    / sqrt(epsilon^2 + Delta^2) d epsilon.
```

The nonzero solution is the superconducting state. The quasiparticle spectrum has minimum energy `Delta`, while single-particle pair-breaking observables see the gap width `2 Delta`; in the weak-coupling limit,

```text
k_B T_c = 1.14 hbar omega exp[-1 / (N(0)V)],
2 Delta(0) / k_B T_c = 3.50.
```

Electrons do not merely stop scattering. A weak net attraction makes the Fermi surface unstable, but the actual state is a coherent superposition of overlapping Cooper-pair occupations. That state lowers the energy collectively, produces gapped quasiparticles, and gives a rigid electromagnetic response that reduces to London/Pippard behavior in the appropriate limits.
