# Toda's Theorem

Toda's theorem states that `PH subseteq P^#P`; equivalently, the polynomial hierarchy is decidable in deterministic polynomial time with a counting oracle. The retrieved 1991 SIAM metadata states the journal theorem in PP form: every set in `PH` is polynomial-time Turing reducible to a set in `PP`, `PH` is included in `BP . oplusP`, and the stronger `PP(PH)` reduction to `PP` is also shown. The 1989 FOCS metadata records the precursor result as a comparison of `PP` and `(+)P` with `PH`, including randomized reducibility to `(+)P` and collapse consequences.

The proof route reconstructed from the local evidence is:

1. Valiant-Vazirani isolation gives a randomized reduction from satisfiability to instances with a unique satisfying assignment.
2. A unique satisfying assignment is detectable by parity, so `NP subseteq BPP^oplusP`; Fortnow attributes this consequence to Valiant-Vazirani and Toda.
3. Relativizing that inclusion, using `oplusP^oplusP = oplusP` from Papadimitriou-Zachos, and applying Zachos's theorem that `NP subseteq BPP` implies `PH subseteq BPP`, gives `PH subseteq BPP^oplusP`.
4. Since bounded-error randomized computation is contained in PP relative to an oracle, it remains to place `PP^oplusP` inside `P^#P`.
5. Fortnow's GapP proof does this by representing the oplusP predicate through parity of a GapP function, amplifying parity with `g(m)=3m^2-2m^3`, summing over all random strings, and reading the majority comparison from a residue that a #P oracle can compute.

The distinctive move is therefore not the slogan that #P is powerful. It is the conversion pipeline: alternation is routed through randomized isolation, parity, threshold comparison, and modular GapP arithmetic until the whole hierarchy becomes a deterministic polynomial-time computation with exact counting access.
