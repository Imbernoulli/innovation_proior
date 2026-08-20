# Synthesis — gauge-principle-yang-mills

## Sources read in full (refs/)
- **PRIMARY**: yang-mills-1954.pdf — Yang & Mills, "Conservation of Isotopic Spin and Isotopic Gauge Invariance," Phys. Rev. 96, 191 (1954). Read pp.191–195 visually. All equations captured.
- **ANTECEDENTS**: jackson-okun-gauge-history.pdf (hep-ph/0012061, Jackson & Okun, "Historical roots of gauge invariance," Rev.Mod.Phys.) for Weyl 1918/1929 abelian gauge principle and the EM gauge transformation; oraifeartaigh-gauge-history.pdf (hep-ph/9810524) for Weyl geometry / fiber-bundle framing; weyl-higgs-gauge.pdf (weylmann.com) for the abelian covariant-derivative derivation; Heisenberg isospin SU(2) from the primary's own intro (refs Heisenberg 1932, Wigner 1937, Cassen-Condon 1936). 
- **EXPLAINER**: yangmills-redux.pdf (physics/0609084, Marateck) — commutator derivation [D_μ,D_ν]=iεF_μν, trial-and-error history, gauge-transform derivation. Plus weyl-higgs (abelian) as a second explainer.
- **SELF-ACCOUNT (key)**: Yang's own first-person account, from NSR "Conversation with Chen-Ning Yang: reminiscence and reflection," Natl Sci Rev 7(1):233 (2020), PMC8288855, doi:10.1093/nsr/nwz113 — archived at `refs/yang-nsr-interview-2019.txt` (svfix repair pass; see `notes/sources.md`). VERBATIM Yang quotes below verified present in the archived file.

## Yang's own reasoning (the backbone of reasoning.md) — verbatim
- "I thought a general principle of interactions was needed, and that principle may come from Weyl's gauge symmetry."
- "I have always liked symmetry considerations and Group Theory. Thus it occurred to me that one should generalize Weyl's gauge symmetry from a U1 symmetry to an SU2 symmetry."
- "the first steps to a non-commuting theory were mathematically easy, the next steps led to formulae that became more and more complicated, and I had to give up."
- "Between 1947 and 1954 I must have repeated this unsuccessful attempt three or four times."
- "During one of these discussions, we observed that the undesired complicated terms were quadratic and cubic. Could they be cancelled if we introduce quadratic and/or cubic terms at the beginning?"
- "a simple quadratic term introduced at the beginning did miraculously cancel all the undesired complicated terms! The cancellation was so beautiful we knew we had hit a gold mine."
- "The theory seemed to require the existence of charged massless particles, which for many reasons cannot be!"
- Pauli: "What is the mass of this field B_μ?" — Yang: "we did not know." Pauli: "That is no sufficient excuse." (mass question; NOT to be imported as dated anecdote in reasoning.md — but the *mass problem* itself is in-frame reasoning.)

The decisive in-frame insight: the SU(2) generalization produces extra quadratic/cubic terms in the field strength that break covariance; the cure is to ADD a quadratic term at the start — that quadratic term is exactly the [A,A] commutator self-interaction. This is the "must interact with itself" realization, in Yang's own reasoning order.

## Background facts (context.md) — all pre-method, sourced
- **Weyl 1918**: scale (gauge/"Eichinvarianz") invariance of GR metric g_μν → e^{λ(x)}g_μν; failed as physics (path-dependent length). 
- **London 1927 / Fock / Weyl 1929**: replace real scale by complex phase ψ → e^{iα(x)}ψ; local phase invariance. EM gauge transform A_μ → A_μ + (1/e)∂_μα (primary's own eq, Gaussian/their units). Covariant derivative ∂_μ → ∂_μ − ieA_μ so that (∂_μ−ieA_μ)ψ transforms like ψ. F_μν=∂_μA_ν−∂_νA_μ gauge-invariant; massless photon because mA_μA^μ not invariant.
- **Heisenberg 1932 isospin**: p,n as two states of nucleon, SU(2) "isotopic spin"; Wigner 1937 total isospin T; Breit-Condon-Present 1936 charge-independence of nuclear forces; Cassen-Condon 1936. By early 1950s isospin is a good GLOBAL symmetry of strong interactions; pion has isospin 1 (three charge states), Hildebrand experiment p+p→π⁺+d vs n+p→π⁰+d confirms pion isospin unity.
- **Pain point (research question)**: a global internal symmetry fixes the isospin "orientation" once and for all across spacetime — but if you've chosen "which is the proton" here, you're not free to choose differently at another spacetime point. That rigidity is unnatural for a symmetry with no physical preferred direction. Demand it locally.

## The derivation (must be COMPLETE in reasoning.md; modern convention D=∂−igA)
1. Global: ψ→Sψ, S∈SU(2) constant; Dirac kinetic ψ̄γ^μ∂_μψ invariant because S constant commutes with ∂_μ.
2. Local: S=S(x). ∂_μ(Sψ)=(∂_μS)ψ+S∂_μψ — the inhomogeneous (∂_μS) term breaks invariance. (Abelian analog: ∂_μα term.)
3. Cure: covariant derivative D_μ=∂_μ−igA_μ, A_μ=A_μ^a T^a Lie-algebra-valued (Hermitian, traceless for SU(2): T^a=σ^a/2). Demand D_μψ→S(D_μψ).
4. Solve for A_μ transform: (∂_μ−igA'_μ)Sψ = S(∂_μ−igA_μ)ψ ⇒ (∂_μS)ψ+S∂_μψ−igA'_μSψ = S∂_μψ−igSA_μψ ⇒ (∂_μS)ψ−igA'_μSψ=−igSA_μψ ⇒ A'_μS = SA_μ + (1/(ig))∂_μS ⇒ **A'_μ = SA_μS^{-1} − (i/g)(∂_μS)S^{-1}** = SA_μS^{-1} + (i/g)S∂_μS^{-1} (using ∂S^{-1}=−S^{-1}(∂S)S^{-1}). Inhomogeneous term present (abelian limit S=e^{iα}: A'_μ=A_μ+(1/g)∂_μα).
   - (Original-paper convention: ψ=Sψ', D=∂−iεB, B'_μ=S^{-1}B_μS+(i/ε)S^{-1}∂_μS — primary eq (3). Same content, S↔S^{-1}.)
5. Field strength from naive curl: try F=∂_μA_ν−∂_νA_μ. Transform: ∂_μA'_ν−∂_νA'_μ = S(∂_μA_ν−∂_νA_μ)S^{-1} + EXTRA terms (from ∂ hitting S and S^{-1}). These extras are the "more and more complicated" quadratic terms Yang hit. They do NOT vanish for non-commuting S.
6. The fix = add a quadratic term: F_μν=∂_μA_ν−∂_νA_μ−ig[A_μ,A_ν]. Cleanest derivation: commutator. [D_μ,D_ν]ψ: derivative terms cancel, leaving [D_μ,D_ν]=−ig(∂_μA_ν−∂_νA_μ−ig[A_μ,A_ν])=−igF_μν. Since D_μ→SD_μS^{-1} (because D_μψ→SD_μψ for all ψ), the commutator ⇒ **F_μν→SF_μνS^{-1}** — homogeneous, covariant. The [A_μ,A_ν] term is the self-interaction; nonzero precisely because SU(2) is non-abelian. ([A,A]=0 for U(1) ⇒ recovers Maxwell.)
   - SIGN CHECK (sympy-verified): D=∂−igA ⇒ [D_μ,D_ν]=−igF, F=∂_μA_ν−∂_νA_μ−ig[A_μ,A_ν]. Matches task-requested form exactly.
   - Original-paper convention eq(4): F_μν=∂_μB_ν−∂_νB_μ+iε(B_μB_ν−B_νB_μ)=∂B−∂B+iε[B,B]; eq(9) for b-vector: f_μν=∂_μb_ν−∂_νb_μ−2εb_μ×b_ν (the cross product = su(2) structure constants). Sign of self-term is convention-dependent (their D has +? no: their D=∂−iεB, their F has +iε[B,B] — opposite of modern because of trial-and-error normalization; both give covariant F'=S^{-1}FS).
7. Lagrangian: L=−¼ F_μν^a F^{aμν} (= −¼ f_μν·f_μν, primary) + ψ̄(γ^μD_μ − m)ψ. The F² term, expanded, contains cubic (A∂A·A) and quartic (A·A·A·A) self-coupling vertices — 3- and 4-gluon vertices. No mA_μA^μ term allowed (not gauge invariant) ⇒ massless b-quanta. That is the mass problem (Pauli's question).
8. Consequences from primary: current J_μ from spin-½ field is NOT conserved alone (∂J=−2εb×J, eq14); the conserved isospin current ℑ_μ=J_μ+2εb×f_μν includes the b-field's own isospin (eq15) — the field is its own source. Quanta: spin 1, isospin 1, charges ±e and 0 (the three b-quanta).

## In-frame discipline
- reasoning.md: one continuous first-person present-tense monologue. No headers. Insight (local symmetry pain) BEFORE method. Walls: the abandoned attempts where formulae blow up (Yang's own "give up" three or four times) lived as the quadratic/cubic terms refusing to cancel; the patch = add quadratic term at start. The mass problem lived as an open worry, NOT as the Pauli anecdote with dates.
- NO dated anecdote (no "February 1954", no Pauli-in-the-audience scene). The mass *reasoning* is in-frame; the seminar story is not.
- Name "isotopic gauge invariance"/"b field" OK; never call it "the paper".
- context.md: 5 sections, pre-method, no hindsight (no Standard Model, no QCD, no renormalizability — all posterior).
