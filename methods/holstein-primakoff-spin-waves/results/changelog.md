# changelog

## 2026-08-18 — svfix(W3_primary_plus_ancestors)
Fixed a sign inconsistency in the Zeeman/field term, present identically in
reasoning.md, answer.md, train_answer.md, and context.md: the trace derived
+gμ_B H Σ_i S_i^z from a bare −μ·H bookkeeping, then hit a negative magnon
gap when expanding it in the boson basis (which would mean the field
depopulates the S^z=+S ground state — backwards) and hand-waved past the
contradiction ("or, depending on the sign convention ... Either way, the
field enters trivially") instead of resolving it. Corrected to
−gμ_B H Σ_i S_i^z, with the sign now pinned by the trace's own stated setup
(field applied along the magnetization, i.e. reinforcing the exchange
ground state), matching refs/sissa_spinwaves.pdf §2.7 ("we have chosen the
sign of B0 so that the lowest energy configuration occurs when all spins
are 'up'"). The final landing (ε_k(H) = qJS(1−γ_k) + gμ_B H,
M(T,H) = gμ_B(NS − Σ_k n_B(ε_k(H)))) is unchanged — it was already
consistent with the corrected sign, only the derivation leading to it was
broken. The decisive step (spin-deviation → boson-number-operator mapping,
S^± fixed by algebra, verified exactly and numerically) was reviewed and
left untouched: it is genuinely self-derived on the page and needs no
external source (see notes/sources.md).
