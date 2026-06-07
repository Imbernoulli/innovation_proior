# Antecedents — load-bearing lineage (with gaps)

## Erdős–Lovász 1975 (the lemma itself) — refs/zhao-pm6-lll.pdf, primary bib [EL75]
- Symmetric form: events A_i with Pr[A_i] ≤ p, each independent of all but ≤ d others; if e·p·(d+1) ≤ 1 then Pr[⋀ Āᵢ] > 0.
- General form: reals x(A) ∈ (0,1) with Pr[A] ≤ x(A)∏_{B∈Γ(A)}(1-x(B)); then Pr[no event] ≥ ∏(1-x(A)) > 0.
- Proof: induction on |S| proving Pr[A_i | ⋀_{j∈S} Ā_j] ≤ x_i, split S = S₁ (neighbors) ∪ S₂; numerator uses independence of A_i from S₂ (Pr[A_i|⋀S₂]=Pr[A_i]≤x_i∏(1-x_j)); denominator ≥ ∏(1-x_j) by induction. Then telescope Pr[⋀Āᵢ]=∏Pr[Āᵢ|prev] ≥ ∏(1-x_i).
- LIMITATION (the pain): purely existential. The conditioning Pr[A_i|⋀S Ā_j] is over an event of exponentially small probability; the proof gives no procedure to *find* the good point, and the good set may have measure ∏(1-x_i) ≈ exponentially small ⇒ rejection sampling is exponential.
- SAT incarnation (Moser STOC Thm): k-CNF with |Γ(C)| ≤ 2^{k-2} is satisfiable (each clause violated w.p. 2^{-k}, p=2^{-k}, d=2^{k-2}, e·2^{-k}·(2^{k-2}+1) ≤ 1).

## Beck 1991 [Bec91] — first constructive (web-researched; not downloaded — GAP)
- Hypergraph 2-coloring; poly-time if each edge shares vertices with ≤ ~2^{k/48} others. Two-phase: identify a "dangerous" core via the dependency structure, 2-color the easy part greedily/randomly, brute-force the small frozen components.
- LIMITATION: exponential gap — `2^{k/48}` vs existential `~2^{k}/e`. The freezing throws away almost all the slack.

## Alon 1991 [Alo91] — randomized simplification, `2^{k/8}`. Same two-phase shape. (web-researched; GAP — not downloaded)
## Srinivasan 2008 [Sri08] — `2^{k/4}`. (web-researched; GAP)
## Moser 2008 [Mos08, arXiv 0807.2120] — `2^{k/2}`. (web-researched; GAP)
- All share the limitation: they never reach the existential threshold; the constant in the exponent is an artifact of the freeze-and-brute-force strategy, not of the problem.

## GAPS
- Beck/Alon/Srinivasan/Moser-2008 antecedent papers were characterized from the survey/abstract trail and the two primary papers' own related-work sections, not downloaded individually. Their precise two-phase mechanics are summarized, not re-derived. This is sufficient for the trace, which only needs "prior constructive work stalled at 2^{k/c} via freeze-and-brute-force"; the method's novelty is the single-phase resample-and-incompressibility argument, fully grounded in the two primary sources.
- No STOC 2009 talk video/slides found; talk's entropy/Kolmogorov framing reconstructed from Fortnow's contemporaneous report + Tao's write-up (both in refs/).
