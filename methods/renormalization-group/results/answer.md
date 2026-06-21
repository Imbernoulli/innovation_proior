# Renormalization Group

The renormalization group's distinctive insight is that a physical theory should be treated as an
effective description that changes with scale. A microscopic Hamiltonian is not simply reorganized
into a nicer microscopic Hamiltonian. Instead, one integrates out short-distance degrees of freedom,
rescales the remaining variables, and reads off a new effective theory. Repeating this operation makes
"change of scale" into a dynamical system on the space of couplings.

Wilson's version is powerful because it does not require the effective theory to keep the original
few-parameter form. Coarse-graining generally generates many new local interactions. Rather than
discarding them by assumption, the Wilsonian view allows the full coupling space and uses locality,
symmetry, and dimensional scaling to decide which terms matter most.

Critical phenomena are then explained by the geometry of the flow:

- A fixed point is a scale-invariant effective theory.
- A critical fixed point has infinite correlation length.
- Relevant directions grow under coarse-graining and must be tuned, such as temperature or field.
- Irrelevant directions shrink under coarse-graining, so microscopic details are forgotten.
- Critical exponents are eigenvalues of the linearized flow near the fixed point.

This explains universality. A magnet, fluid, or alloy can have different microscopic constituents,
but if their long-distance flows approach the same critical fixed point and differ only in irrelevant
directions, they share the same scaling laws and critical exponents.

In one dimension the Ising decimation step makes the idea explicit:

```text
K' = (1/2) log cosh(2K)
tanh K' = tanh^2 K
```

Every finite coupling flows toward the disordered fixed point K = 0, while K = infinity is the
zero-temperature unstable fixed point. The flow therefore has no finite-temperature critical fixed
point, matching the known absence of a finite-temperature phase transition in the one-dimensional
Ising chain.

In higher-dimensional phi^4 theory, the same logic becomes quantitative through the epsilon = 4 - d
expansion. At four dimensions the interaction is marginal; just below four dimensions a nontrivial
Wilson-Fisher fixed point appears close enough to the Gaussian fixed point to compute its eigenvalues
perturbatively. That is how Wilson's method turns scaling and universality from phenomenology into a
calculational framework.
