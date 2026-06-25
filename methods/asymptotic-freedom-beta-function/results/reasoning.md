Let me start from the thing that won't leave me alone. You hit a proton hard — deep inelastic, large momentum transfer — and the inclusive cross sections come out *scaling*: structure functions that depend only on x = Q²/2Mν, not separately on Q². The constituents act free and point-like, spin-½, fractional baryon number. Quarks, really, even though no one will say it without apologizing. And yet you can't knock a quark loose to save your life. So the force is strong enough to confine permanently, but at short distance it acts as if it had turned off. A strong force that gets weak when you look closely. That is the whole puzzle, and everything I do has to bend toward explaining it.

The renormalization group is the only honest tool for "what happens at short distance." Renormalization already taught us the coupling is not a number, it's a function of scale — you measure g at some scale M, and at another scale it's something else, and the rate of change is β(g) = M dg/dM. Short distance means large momentum means M → ∞. If I want the constituents to look free up there, I need the effective coupling to *go to zero* as M → ∞. So the question is brutally simple in the end: what is the sign of β near g = 0? If β < 0, then g shrinks toward the ultraviolet — the force turns off — and I can hope to explain scaling. If β > 0, g grows, the way it does in electrodynamics, and I'm dead.

And here's the wall I keep slamming into before I even compute anything: everyone "knows" β > 0. Electrodynamics is the paradigm and it screens. Picture the vacuum as a polarizable medium of virtual e⁺e⁻ pairs. Drop a bare charge in; the pairs orient to cancel it; the dressed charge you see at distance r is *smaller* than the bare one, and gets smaller the farther out you go — equivalently, larger as r → 0. The effective coupling rises toward short distance. β > 0. Landau took this to the cliff edge: sum the screening to all orders and the physical charge measured at any finite distance vanishes for any bare charge — zero charge. And the lesson everyone drew was that this is *generic*: every theory anyone computed — scalar φ⁴, Yukawa, Abelian gauge — gave β > 0. Screening felt like a law of nature. There was even a clean physical reason for it and no obvious physical reason for the opposite, so people assumed the opposite couldn't happen.

So before computing I should ask whether I can *prove* the opposite is impossible, and get it over with. That's the efficient move — kill the hope with a general argument. The argument that works against scalars and fermions and Abelian gauge fields is roughly a positivity argument: the spectral representation of the relevant two-point function has a definite sign, intermediate physical states come with positive probability, and that pins the sign of the running. I try to push the same machinery onto non-Abelian gauge theory. And it doesn't go through. The spectral/positivity reasoning that closes the door for every other theory just... fails to close for Yang-Mills. The gauge field is not like the others. I can't prove non-Abelian gauge theory screens. I also can't prove it doesn't. The general argument is silent exactly here.

That silence is the whole opening. The one renormalizable theory the no-go can't touch is the one with a charged, self-interacting gauge field. I have to actually compute its β-function. No shortcut.

Why might it be different at all? Let me get my hands on the physics before the Feynman graphs, because the arithmetic is going to be a swamp and I want a compass. Electrodynamics screens because the only thing the vacuum does is polarize charge. But there's an identity I keep underusing: in units c = 1, the vacuum's dielectric constant and magnetic permeability obey ε μ = 1. So "the charge grows at short distance" (ε > 1) is *the same statement* as "the vacuum is diamagnetic" (μ < 1). I've been thinking electrically. Let me think magnetically. Electric screening ⟺ diamagnetism. Now — is the vacuum forced to be diamagnetic? Classically yes: induced currents oppose the applied field, Lenz's law, μ < 1 always. But quantum mechanically a medium can be *para*magnetic if it has permanent magnetic dipoles that align *with* the field. Spin does that.

So decompose the vacuum's magnetic response to a color charge into two pieces. There's the orbital/diamagnetic piece — the ordinary Lenz response of charged virtual stuff — and it contributes negatively to μ; for a particle of charge q this piece is something like −q²/3. That's the screening piece, the only one a spinless or a fermionic source really has. But a *spin-1* charged particle is a permanent magnetic dipole, and it can give a large *para*magnetic contribution that goes the other way, μ > 1, which by ε μ = 1 means ε < 1 means *anti*screening — the effective charge gets *bigger* with distance, smaller toward short distance. The gluon is exactly this: a charged spin-1 boson. In electrodynamics the photon is neutral, so there's no such contribution and you only ever get the diamagnetic screening. In a non-Abelian theory the gluons are charged, so the paramagnetic spin term is on the table, and it has the opposite sign. Two effects fighting. That's why this theory can be different. Now I need the arithmetic to tell me which one wins.

Let me set the calculation up cleanly. One coupling g, gauge group G with structure constants f^{abc}, fermions ("quarks") in some representation R with generators T^a. The coupling appears in the covariant derivative D_μ = ∂_μ − i g A_μ^a T^a and in the field strength F_{μν}^a = ∂_μ A_ν^a − ∂_ν A_μ^a + g f^{abc} A_μ^b A_ν^c — that last term is the gluon self-interaction, the thing QED doesn't have. β at one loop is fixed by the divergent (logarithmic, the −ln μ²) parts of the one-loop graphs; the finite parts are scheme junk. So I need the one-loop counterterms.

There's a subtlety that makes this harder than QED and I should face it now rather than be ambushed. In QED the photon is neutral, so charge renormalization is gauge-trivial: the Ward identity gives Z₁ = Z₂ (vertex renorm equals field renorm), and the running of e comes purely from the photon vacuum polarization Z₃, e_bare = e/√Z₃. Here the gluon is charged, the Ward identities are the weaker Slavnov–Taylor identities, and Z₁ ≠ Z₂. The coupling renormalization, read off the quark–gluon vertex, is

  g_bare = g · Z₁ / (Z₂ Z₃^{1/2}),

where Z₁ is the quark–gluon vertex, Z₂ the quark field, Z₃ the gluon field. Writing each Z = 1 + δ and taking the log derivative, the one-loop β-function is built from the residues of the 1/ε poles of the counterterms,

  β(g) = g · Res[ 2δ₁ − 2δ₂ − δ₃ ] · (loop counting),

i.e. I need three counterterms: the quark self-energy δ₂, the quark–gluon vertex δ₁, and the gluon self-energy (vacuum polarization) δ₃. The gluon piece δ₃ is where the magnetic physics will live. Let me grind through them, watching every sign — and let me keep the group invariants generic so I can see exactly which sign comes from which structure. The invariants are C(G), the adjoint Casimir (= N for SU(N)), and for the quarks C(Q) = C₂(R) and the index R_net = Σ T(r) (= n_f · ½ for n_f Dirac fermions in the SU(N) fundamental).

Take the quark self-energy first, the piece I'll call δ₂. A quark emits and reabsorbs a gluon. Strip the color and it is exactly the QED electron self-energy diagram. The Dirac/momentum part gives the familiar log divergence; the only new thing is the color factor T^a T^a summed over a, which on an irreducible multiplet is the Casimir C(Q) times the identity. So Σ(p) ∝ C(Q) × (QED self-energy), and the counterterm needed to cancel its UV pole is

  δ₂ = − (g²/16π²) · C(Q) · (1/ε),  i.e.  Res[δ₂] = − C(Q).

Clean. The quark field renormalization is just the QED one decorated by the quark Casimir.

Now the quark–gluon vertex, δ₁. Two diagrams. The first looks like the QED vertex correction — quark emits gluon a, exchanges a gluon across, reabsorbs — with color factor T^b T^a T^b summed over b. I have to reduce that. Write T^b T^a T^b = T^b (T^b T^a + [T^a, T^b]) = T^b T^b T^a + T^b (i f^{abc} T^c) = C(Q) T^a + i f^{abc} T^b T^c. The second term: i f^{abc} T^b T^c = i f^{abc} · ½[T^b, T^c] (the symmetric part dies against the antisymmetric f) = i f^{abc} · ½ (i f^{bcd} T^d) = −½ f^{abc} f^{bcd} T^d = −½ C(G) T^a, using f^{abc} f^{dbc} = C(G) δ^{ad}. So

  T^b T^a T^b = ( C(Q) − ½ C(G) ) T^a.

So the QED-like graph, which in QED would give the C(Q) piece, here gets shifted by −½ C(G) — the first place the non-Abelian self-interaction sneaks into the vertex. The second vertex diagram is genuinely new: it has the *three-gluon* vertex inside it (quark emits two gluons that meet at a triple-gluon vertex). Its color factor reduces, via g f^{abc} T^c T^b = −i g · ½ C(G) T^a (same identity as above), to a pure C(G) piece, and working the Lorentz/momentum algebra at zero external momentum the divergent part comes out as +(3/2) C(G) relative to the QED normalization. Add the two:

  Γ-divergence ∝ ( C(Q) − ½ C(G) ) + ( (3/2) C(G) ) = C(Q) + C(G),

so

  δ₁ = − (g²/16π²) · ( C(Q) + C(G) ) · (1/ε),  Res[δ₁] = − ( C(Q) + C(G) ).

Sanity flag I want to keep: δ₁ ≠ δ₂ now — they differ by exactly C(G), the signature of the non-Abelian Ward (Slavnov–Taylor) identity failing to be the simple QED one. Good; that's expected, not a mistake.

Now the gluon vacuum polarization, δ₃, where I have to be careful and where the magnetic intuition has to be vindicated by signs. There isn't one diagram, there are five. (1) The quark loop, like the QED electron loop. (2) The gluon loop — two gluon propagators joined by two three-gluon vertices. (3) The gluon "tadpole" / sideways loop from the four-gluon vertex. (4) The ghost loop. (5) The counterterm. Let me take them in order.

The quark loop is the screening piece, the one that mimics QED. Strip color and it's the electron loop in vacuum polarization; the color trace over one quark multiplet gives the index, tr(T^a T^b) = R(r) δ^{ab}, and summing the n_f flavors gives R_net δ^{ab}. The QED loop has the well-known transverse divergence with coefficient −4/3 per unit of charge-squared (the −1/3 dielectric times the loop normalization). So

  Σ₁^{μν}(k) ∝ (k²g^{μν} − k^μk^ν) · ( − (g²/12π²) R_net /ε ),

and the counterterm to cancel it is

  δ₃(quark) = − (g²/16π²) · (4/3) R_net · (1/ε).

This is exactly the screening contribution, and its sign is the QED sign. If this were the only thing, β > 0, and the story would end like everyone expects. The whole game is whether the gauge loops beat it.

Now the gluon loop, diagram (2). Two three-gluon vertices, two gluon propagators, symmetry factor ½, color factor f^{acd} f^{bcd} = C(G) δ^{ab}. The momentum numerator is a quadratic polynomial built from the three-gluon vertex factors V^{μαβ}(k, p₁, p₂) = g^{αβ}(p₁−p₂)^μ + g^{βμ}(p₂−k)^α + g^{μα}(k−p₁)^β. This is the messiest object in the whole calculation. I expand it, average the loop direction (ℓ^μℓ^ν → (ℓ²/D) g^{μν}), shift to the Feynman-parameter loop momentum ℓ = p₁ + x k with Δ = −x(1−x)k². And here's the first warning that signs are about to bite: the gluon loop, by itself, is **not transverse**. The numerator splits into a "good" piece proportional to (k²g^{μν} − k^μk^ν) and a "bad" piece proportional to g^{μν} alone (and an ℓ² term). That bad piece would be a gauge-symmetry-violating gluon mass — it must not survive. So I cannot read off δ₃ from the gluon loop alone; I have to keep the bad piece and trust it to cancel against the other diagrams.

The four-gluon tadpole, diagram (3): one quartic vertex, one gluon propagator looping back. Color factors collapse (the f f contractions give C(G) δ^{ab} on two of the three Lorentz structures and zero on the first), the Lorentz contraction g_{αβ}[...] gives (2D−2) g^{μν}, and after the single propagator integral this contributes a purely "bad" g^{μν} term — no good transverse piece at all. It exists only to help cancel.

The ghost loop, diagram (4): two ghost propagators, two ghost–gluon vertices. Ghosts are the price of covariant quantization of a non-Abelian theory — anticommuting scalars that fix up the unphysical gluon polarizations and restore unitarity. They carry an overall minus sign for the closed loop (fermionic statistics) even though they're scalars, the momentum numerator is 2 p₂^μ p₁^ν, and the color factor is f^{acd} f^{bdc} = − C(G) δ^{ab}. Its good piece comes out proportional to −2x(1−x) C(G), plus its own bad piece.

This is the moment to slow down, because if I drop the ghost I get a non-transverse, gauge-noninvariant answer with the wrong number, and it would be easy to convince myself the theory screens. Let me total the three gauge/ghost diagrams. Adding the good (transverse) numerators:

  good(gluon) = (6 − D) + (4D − 6) x(1−x),
  good(tadpole) = 0,
  good(ghost) = − 2 x(1−x),

so

  good_total = (6 − D) + (4D − 6) x(1−x) − 2 x(1−x) = (6 − D) + 4(D − 2) x(1−x).

And the bad (g^{μν}-only) pieces from the three diagrams add to a term proportional to (D−2)[ −ℓ² + D x(1−x) k² ]/D, whose dimensionally-regularized integral I have to actually evaluate rather than wave away. Doing the ℓ integral with the standard Γ-function machinery, the bad term is proportional to

  (D/2 − 1) Γ(1 − D/2) + Γ(2 − D/2),

and using y Γ(y) + Γ(y+1) = 0 with y = 1 − D/2 this should be zero. Let me make sure I'm not fooling myself with a half-remembered Γ-function identity: evaluate (D/2 − 1) Γ(1 − D/2) + Γ(2 − D/2) numerically at D = 3.7, 3.9, 3.99 and it comes out 0 to machine precision each time (the recurrence Γ(y+1) = y Γ(y) is exactly what makes the two terms equal and opposite). So the bad, gauge-breaking piece cancels — no gluon mass — but *only* because the ghost loop was included with its correct minus sign and the tadpole with its correct structure. The transversality is the check that I assembled the gauge sector correctly. Relief: it's transverse.

Now integrate the surviving good piece. At D → 4, good_total → (6−4) + 4·2·x(1−x) = 2 + 8 x(1−x). Integrate over the Feynman parameter x from 0 to 1: ∫₀¹ [2 + 8 x(1−x)] dx = 2 + 8·∫₀¹ x(1−x) dx = 2 + 8·(1/6) = 2 + 4/3 = 10/3. Let me not trust my own arithmetic on that and check it numerically: sampling 2 + 8 x(1−x) on a fine grid of x ∈ [0,1] and integrating gives 3.33333…, which is 10/3 to the digits I bothered to compute. Good, the rational is right. The 1/ε pole of the loop integral is the usual 1/16π². So the gauge+ghost vacuum polarization is

  Σ_{234}^{μν} ∝ (k²g^{μν} − k^μk^ν) · (g²/16π²) · (C(G)/2) · (10/3) · (1/ε) = (k²g^{μν} − k^μk^ν) · (g²/16π²) · (5/3) C(G) /ε,

and the counterterm to cancel it is

  δ₃(gauge+ghost) = + (g²/16π²) · (5/3) C(G) · (1/ε).

Putting the quark loop back, the full gluon-self-energy counterterm is

  δ₃ = (g²/16π²) · [ (5/3) C(G) − (4/3) R_net ] · (1/ε),  Res[δ₃] = (5/3) C(G) − (4/3) R_net.

Notice the structure already: the gauge sector contributes a *positive* δ₃ proportional to C(G) (antiscreening seed), the quarks a *negative* piece (screening), and they fight. But the self-energy alone is +5/3 C(G), not the whole gauge coefficient. The rest must come from the *assembly*, because the vertex and self-energy enter the β-function together through the Slavnov–Taylor combination, not the self-energy alone. Let me assemble.

  Res[ 2δ₁ − 2δ₂ − δ₃ ]
   = 2·( − (C(Q) + C(G)) ) − 2·( − C(Q) ) − ( (5/3) C(G) − (4/3) R_net )
   = −2 C(Q) − 2 C(G) + 2 C(Q) − (5/3) C(G) + (4/3) R_net
   = − 2 C(G) − (5/3) C(G) + (4/3) R_net
   = − (11/3) C(G) + (4/3) R_net.

The C(Q) — the quark Casimir — has cancelled: the −2 C(Q) from 2δ₁ and the +2 C(Q) from −2δ₂ sit there and annihilate. I want to be sure that's a structural cancellation and not a coincidence of one C(Q) value, so let me feed the assembly arbitrary C(Q) and watch the combo: with C(G) = R_net = 3 fixed, taking C(Q) = 0, 4/3, 7/2, 100 all return the same combo = −7. It doesn't depend on C(Q) at all — which is the consistency requirement, since the quark's own charge can't show up in the universal running of the gauge coupling. The −2 C(G) from the vertices and the −5/3 C(G) from the self-energy combine to −11/3 C(G). And so

  **β(g) = (g³/16π²) · [ − (11/3) C(G) + (4/3) R_net ] + O(g⁵).**

For SU(N) with n_f Dirac fermions in the fundamental, C(G) = N and R_net = n_f · ½, so

  **β(g) = (g³/16π²) · [ − (11/3) N + (2/3) n_f ].**

For SU(3) — color — N = 3:

  **β(g) = − (g³/16π²) · ( 11 − (2/3) n_f ).**

The first term, the gauge contribution, is **negative**. The fermion term is positive (screening, the QED sign), and it opposes the gauge term. Define b₀ = 11 − (2/3) n_f so that β = −(g³/16π²) b₀; β < 0 — the coupling running to zero in the ultraviolet — requires b₀ > 0, i.e. n_f < 33/2 = 16.5. Let me locate the changeover by hand rather than just quoting the bound: evaluate the combo −(11/3)·3 + (4/3)·(n_f/2) for SU(3) at the flavor counts that straddle it. n_f = 16 gives −1/3 (still negative, still asymptotically free); n_f = 17 gives +1/3 (positive — screening, no asymptotic freedom). The sign of the gauge coupling's running genuinely flips between 16 and 17 flavors, which is the integer reading of 16.5. So the property survives for any quark content up to sixteen flavors. In the real world there are 6, which gives b₀ = 11 − 4 = 7, well inside the window, so the gauge antiscreening wins comfortably.

Wait. I have to stop and check this, because I do not believe it. The sign is the entire claim and everyone — Landau, the spectral-positivity argument, every prior calculation — says it should be the other way. A single dropped sign anywhere in five diagrams plus a vertex plus a self-energy would flip it, and I have in fact made sign errors getting here. So let me lean on whatever checks I can actually carry out, and be honest about which ones I can only check in principle.

The one I *can* compute on the spot is the Abelian limit, and it is the most reassuring because I already know the answer it has to give. Set C(G) → 0 (no self-interaction, no ghosts, neutral gauge boson). Then 2δ₁ − 2δ₂ − δ₃ → 2(−C(Q)) − 2(−C(Q)) − (−(4/3) R_net) = +(4/3) R_net. The C(G) machinery vanishes entirely and only the screening term survives; with R_net > 0 the combo is positive, β > 0 — exactly QED screening, the Landau pole, the answer everyone knows for electrodynamics. Plugging C(G)=0, C(Q)=1, R_net=3 into the same assembly returns combo = +4, positive as required. So my assembly degenerates to the right thing in the limit where I have an independent answer; the whole edifice isn't producing the minus sign out of an across-the-board sign slip, because in the Abelian corner it produces the *plus* sign correctly.

The second support is one I already paid for: transversality. The bad g^{μν} term died by the Γ-function identity, and that cancellation only closed because the ghost loop went in with its minus sign and the tadpole with its structure — i.e. the gauge sector that supplies the +5/3 C(G) is internally consistent, not a bag of mismatched pieces.

The two checks I'd most want but *cannot* finish here are honest gaps I should name rather than wave through. (i) Gauge-parameter independence: I worked in one covariant gauge, the individual Z's are gauge-dependent, and β must not be. I expect the ξ-dependence to live in the C(Q) and finite pieces that cancelled, leaving −11/3 intact — but I have not redone the divergent parts at a second value of ξ, so this remains an expectation I'd verify before I'd stake everything on it. (ii) Vertex universality: the Slavnov–Taylor identities demand the same coupling from the ghost–gluon, three-gluon, and four-gluon vertices, e.g. Res[2δ₁^{(3g)} − 3δ₃] should also give −11/3 C(G) + (4/3) R_net. That is a real cross-check and exactly the kind of thing that catches a single bad term — I recall one of us briefly getting +11/3 from one wrong sign and being momentarily "confirmed" in the expected non-asymptotic-freedom — but reproducing it means recomputing a different vertex's δ₁, which I have not done here. So: the Abelian limit and transversality I have actually checked; the gauge-independence and vertex-universality I am relying on and would still want to nail down. On the strength of the first two, plus the internal C(Q) cancellation, I'll proceed with the minus sign while flagging that the redundancy checks are owed.

And it lines up with the magnetic picture, which I now read off the arithmetic instead of asserting. The contribution of a particle of charge q to the vacuum permeability is Δμ = [ −1/3 + (γ s)² ] q² — a diamagnetic/orbital −1/3 plus a paramagnetic spin term (γ s)², with γ the gyromagnetic ratio. The gluon is charged, spin s = 1, and gauge invariance fixes its gyromagnetic ratio to the Yang-Mills value γ = 2, so Δμ_gluon = (−1/3 + 2²) q² = (−1/3 + 4) q² = **+11/3** q². There's the 11/3, and there's *why* it's 11/3: it's 4 (the spin-1 paramagnetism, the antiscreening) minus 1/3 (the ordinary diamagnetic screening). The spin paramagnetism overwhelms the orbital diamagnetism, the vacuum antiscreens, and by ε μ = 1 the color charge gets *weaker* toward short distance. A spin-½ quark has γ = 2 as well but s = ½ and the fermion-loop extra minus sign, giving Δμ_quark = −(−1/3 + (2·½)²) q² = −(−1/3 + 1) q² = −2/3 q² → the −2/3-per-flavor screening that opposes it. The diagrammatic −11/3 C(G) + (4/3) R_net is the field-theoretic incarnation of "gluon spin-paramagnetism beats quark-plus-orbital screening, unless there are too many quarks." This is the physical reason the no-go theorems missed it: they had no charged spin-1 source, so they never had the paramagnetic term, so they only ever saw screening.

Now cash it in. β < 0 near the origin means the renormalization-group flow drives g(M) → 0 as M → ∞: the origin is an ultraviolet-stable fixed point. Solve the one-loop flow to make this quantitative. With β(g) = − b g³ (collecting b = b₀/16π² > 0) and the scale variable t = ½ ln(s/M²),

  dḡ/dt = − b ḡ³  ⟹  d(1/ḡ²)/dt = 2b  ⟹  ḡ²(t) = g² / ( 1 + 2 b g² t ).

As t → ∞ (deep ultraviolet), ḡ²(t) → 0 like 1/(2 b t) ~ 1/ln(s) — the coupling vanishes *logarithmically*. Let me put the SU(3), n_f=6 numbers on it so "vanishes" isn't just a word: with b₀ = 7 and a starting g²(0) = 1, the formula gives g²(5) = 0.69, g²(50) = 0.18, g²(500) = 0.022, g²(5000) = 0.0023 — and the decline is exactly the slow 1/t crawl I predicted, each factor-of-ten in t pulling g² down by roughly a factor of ten only once the log is large, not faster. That is the logarithmic, not power-law, approach to zero. So matrix elements of currents between physical states approach their free-field values, up to corrections that fall like inverse powers of the logarithm of the momentum. That is precisely Bjorken scaling — *and* its violation: scaling is not exact, it is broken by *calculable logarithmic* terms set by b₀. The puzzle dissolves. A theory can be strong at the hadronic scale (where 1 + 2 b g² t is order one and ḡ is large) and simultaneously make its constituents look free at short distance (where the log in the denominator blows up and ḡ → 0). The strong force turns off at short distance not by fiat but because the gluon's own charged spin-1 nature antiscreens the vacuum.

One more thing the formula tells me, looking the other way: as t → negative (toward the infrared, long distance), the denominator 1 + 2 b g² t hits zero and ḡ² blows up. Perturbation theory says nothing reliable there; the coupling grows without bound at low momentum. That's the side of the same coin that will have to be confinement, and one-loop perturbation theory has no business pretending to describe it — exactly as it should be, since free quarks are never seen. The honest statement is only about the ultraviolet: there, and only there, the coupling is provably weak.

Let me also record the cleanest *general* form, since the derivation never used SU(3) — it used C(G) and R_net. For any simple gauge group with any fermion content, asymptotic freedom is the condition that the gauge antiscreening beat the matter screening:

  R_net = Σ_r T(r) < (11/4) C(G).

For SU(N) fundamentals, T(r) = ½ each, C(G) = N, so this is n_f/2 < 11N/4, i.e. n_f < 11N/2 — for SU(3), n_f < 33/2, the sixteen-flavor bound again.

The whole chain, end to end: scaling at SLAC says the strong coupling must vanish in the ultraviolet → exact scaling in an interacting renormalizable theory requires an ultraviolet-stable fixed point at the origin, i.e. a negative one-loop β → every theory of scalars, fermions, and Abelian gauge fields has β > 0 (screening), and the only renormalizable case the no-go arguments can't close is non-Abelian gauge theory → compute its one-loop β honestly, with the charged self-interacting gluons and the Faddeev–Popov ghosts that covariant quantization demands → the quark loop screens (+4/3 R_net) but the gluon and ghost loops, assembled through the Slavnov–Taylor combination 2δ₁ − 2δ₂ − δ₃, give −11/3 C(G), antiscreening, because a charged spin-1 gluon's paramagnetism (the +4 from γ=2, s=1) overwhelms its diamagnetism (the −1/3) → β = (g³/16π²)[−(11/3)C(G) + (4/3)R_net], negative for not-too-many fermions → the coupling runs to zero logarithmically in the ultraviolet → Bjorken scaling, with calculable logarithmic violations, and a color SU(3) gauge theory of quarks and gluons as the theory of the strong interactions.

Here is the calculation distilled to the symbolic assembly and the running coupling, the same arithmetic done by hand above.

```python
import numpy as np
from fractions import Fraction as F

# --- One-loop counterterm residues (coefficients of g^2/(16 pi^2 eps)) ---
# Derived above, diagram by diagram, in a covariant gauge with FP ghosts.
def residues(C_G, C_Q, R_net):
    res_d2 = -C_Q                       # quark self-energy (QED-like x quark Casimir)
    res_d1 = -(C_Q + C_G)               # quark-gluon vertex: (C_Q - C_G/2) + (3/2)C_G
    res_d3 = F(5, 3) * C_G - F(4, 3) * R_net   # gluon vac. pol.: +5/3 C_G (gauge+ghost), -4/3 R_net (quarks)
    return res_d1, res_d2, res_d3

# --- Assemble the one-loop beta coefficient via the Slavnov-Taylor combo ---
# beta(g) = g * Res[2 d1 - 2 d2 - d3] * g^2/(16 pi^2)
#         = (g^3/16 pi^2) * [ -(11/3) C_G + (4/3) R_net ]
def one_loop_beta_coefficient(C_G, C_Q, R_net):
    d1, d2, d3 = residues(C_G, C_Q, R_net)
    combo = 2*d1 - 2*d2 - d3            # = -(11/3) C_G + (4/3) R_net ; C_Q cancels
    return combo                        # this is the coefficient of g^3/(16 pi^2) in beta

# SU(N) data: adjoint Casimir, fundamental Casimir, index of n_f fundamentals
def su_n_data(N, n_f):
    C_G = F(N)
    C_Q = F(N*N - 1, 2*N)               # not needed for beta (cancels), kept for honesty
    R_net = F(n_f, 2)                    # n_f Dirac fundamentals, T(fund)=1/2 each
    return C_G, C_Q, R_net

# --- Running coupling: solve dg/dt = beta(g) at one loop, t = (1/2) ln(s/M^2) ---
# With beta = -(g^3/16 pi^2) b0,  g^2(t) = g0^2 / (1 + 2 b g0^2 t),  b = b0/16 pi^2.
def run_coupling_sq(g0_sq, b0, t):
    b = b0 / (16 * np.pi**2)
    return g0_sq / (1.0 + 2.0 * b * g0_sq * t)

if __name__ == "__main__":
    N, n_f = 3, 6
    C_G, C_Q, R_net = su_n_data(N, n_f)
    combo = one_loop_beta_coefficient(C_G, C_Q, R_net)   # = -(11/3)*3 + (4/3)*3 = -7
    b0 = -combo                                          # beta = -(g^3/16pi^2) b0
    print("beta coefficient  -(11/3)C_G + (4/3)R_net =", combo)   # -7
    print("b0 = 11 - 2 n_f/3  =", F(11) - F(2,3)*n_f, " -> ", b0) # 7
    # asymptotic freedom <=> b0 > 0 <=> R_net < (11/4) C_G  <=> n_f < 33/2
    print("asymptotically free:", b0 > 0)
    # coupling shrinks toward the UV (large t), starting from g^2(0) = 1:
    for t in [0.0, 5.0, 50.0, 500.0]:
        print(f"t={t:7.1f}  g^2(t)={run_coupling_sq(1.0, float(b0), t):.4f}")
```
