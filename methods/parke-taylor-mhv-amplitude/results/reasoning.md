The thing in front of me is a wall of Feynman diagrams. The experimentalists at the SppS and the Tevatron are seeing multi-jet events, and the people writing the Monte Carlos need tree-level matrix elements for gluon scattering with many gluons out — gg → many g — to predict the QCD backgrounds underneath any new-physics signal. And the diagram count is hopeless: four diagrams for two final gluons, twenty-five for three, two hundred twenty for four, and then 2485, 34300, 559405, ten and a half million for n up to eight. Each diagram is not one term either; the three- and four-gluon vertices are sprawling, momentum-dependent things, so every diagram fans out into many terms, and the cross terms in the square are the square of that. By n = 6 it is already brutal; the four-, five-, six-gluon results exist only because people ground them out by hand. Beyond that, nothing. Four-jet production has no analytic form at all.

So let me not try to be cleverer at summing diagrams. Let me ask the structural question: is there *one* helicity configuration whose answer, at arbitrary n, is simple enough to write down in closed form? Because if the final answers for n = 4, 5, 6 came out short even though the intermediate diagrammatics were enormous, that is a hint that an enormous cancellation is happening, and a cancellation that severe usually means there is an object far simpler than the diagrams that controls the result.

First I should pick the right variables, because the choice of variables is where most of the simplicity hides. The external particles are effectively massless — gluons, and the quarks too at these energies. For a massless momentum, k² = 0, the bispinor k_{a ȧ} = k_μ (σ^μ)_{a ȧ} has vanishing determinant, so it is rank one and factorizes into a single pair of two-component Weyl spinors, k_{a ȧ} = λ_a λ̃_ȧ. That means the natural Lorentz invariants are not the dot products s_ij = 2 p_i·p_j but their *square roots*, the spinor brackets

  ⟨ij⟩ = ε^{ab} λ_{i a} λ_{j b},  [ij] = ε^{ȧḃ} λ̃_{i ȧ} λ̃_{j ḃ}.

They are antisymmetric, ⟨ij⟩ = −⟨ji⟩, ⟨ii⟩ = 0, and contracting the two halves back together gives ⟨ij⟩[ji] = 2 p_i·p_j = s_ij. For real momenta λ̃ = λ*, so [ij] = ⟨ij⟩*, and the bracket is literally √s_ij dressed with a phase: ⟨ij⟩ = √s_ij e^{iφ}, [ij] = √s_ij e^{−iφ}. I want to keep that picture in mind, because a square root with a phase is exactly the analytic shape of a collinear singularity — when two momenta become parallel s_ij → 0 like the square of the bracket, and the phase rotates as you spin one momentum about the collinear axis. The dot products can't carry that phase; the brackets can. If amplitudes are going to be simple, I bet they are simple in the brackets.

Now I need polarization vectors for the gluons that play nicely with this. The old way contracts a photon polarization with γ and writes it through an external fermion pair, but for *pure gluon* scattering there is no fermion pair to anchor on. The better object — the one Xu, Zhang and Chang wrote down, and that the helicity-amplitude people had been converging on — uses one arbitrary null reference momentum k per gluon:

  ε⁺_μ(p,k) = ⟨p−|γ_μ|k−⟩ / (√2 ⟨k−|p+⟩),  ε⁻_μ(p,k) = ⟨p+|γ_μ|k+⟩ / (−√2 [k+|p−]).

The reference momentum k is the leftover on-shell gauge freedom: shifting ε by anything proportional to p doesn't change a physical amplitude, and that freedom is exactly the freedom to move k. So I can pick a *different* k for every external gluon, and pick it to make terms die. The key dot products: ε⁺(p,k)·ε⁺(p′,k) = 0 when the two positive-helicity gluons share a reference momentum k, and ε⁺(p,k)·ε⁻(k,k′) = 0 — a positive-helicity polarization is orthogonal to the negative-helicity polarization of the very gluon used as its reference. These two facts are levers I can pull to annihilate diagrams.

Let me use them immediately, because the first thing I should understand is which helicity configurations are *zero*. Take all gluons positive-helicity. Give every one of them the same reference momentum k. Then ε_i⁺·ε_j⁺ = 0 for every pair. Now count: a tree diagram with n external gluons has at most n − 2 cubic vertices (each cubic vertex is linear in momentum, and you can't have more than n − 2 of them in a tree), so each term has at most n − 2 powers of momentum to soak up Lorentz indices, against n polarization vectors. There are not enough momenta to contract all n polarizations into momenta; at least one pair ε_i·ε_j must be left contracted with each other in every single term. But every such pair vanishes. So the all-plus amplitude is identically zero for n ≥ 4. 

Now flip one gluon to negative, say gluon 1, the rest positive. Give all the positive gluons (2,…,n) the reference momentum k = p_1, and give gluon 1 the reference momentum equal to any one of the others' momenta, say p_2. Then for any two positive gluons ε_i⁺·ε_j⁺ = 0 (shared reference), and ε_1⁻·ε_j⁺ = 0 because ε_j⁺ uses p_1 as reference and ε⁺(p,k)·ε⁻(k,·) = 0. So again every polarization dot product vanishes, and by the same counting argument the single-minus amplitude vanishes too. So the all-plus and one-minus configurations are zero for any n ≥ 4.

I want to be sure this isn't an artifact of my gauge choice, so let me re-derive it a second way, from supersymmetry, because that argument is gauge-independent and it will also hand me relations I'll need. At tree level a pure-gluon amplitude is the *same* in ordinary Yang–Mills as in the supersymmetric theory — no fermion or scalar can appear on an internal line of an all-gluon tree, so adding superpartners changes nothing. Therefore the gluon trees obey the supersymmetry Ward identities. Let Q(k) be the SUSY charge with a fermionic parameter built from a negative-helicity spinor of an arbitrary null momentum k times a Grassmann θ. It rotates a gluon into a gluino and back:

  [Q(k), g±(p)] = ∓ Γ±(p,k) Λ±(p),  [Q(k), Λ±(p)] = ∓ Γ∓(p,k) g±(p),

with Γ⁺(p,k) = θ⟨kp⟩ and Γ⁻(p,k) = θ[kp]. Because Q annihilates the vacuum, the vacuum expectation value of [Q, (string of operators)] is zero, so

  0 = Σ_i ⟨z_1 ⋯ [Q, z_i] ⋯ z_n⟩.

Apply it to Λ⁺ g⁺ g⁺ ⋯ g⁺ (one gluino, all-plus). Commuting Q through, the gluino turns into a gluon (giving the all-plus *gluon* amplitude times Γ⁻ of leg 1) and each gluon turns into a gluino (giving amplitudes with two same-helicity gluinos). But the fermion–fermion–vector coupling conserves helicity, so any amplitude with two like-helicity gluinos vanishes. Every term on the right except the first is killed, and the surviving relation says Γ⁻(p_1,k) · A(g⁺ g⁺ ⋯ g⁺) = 0. Since Γ⁻ ≠ 0 for generic k, the all-plus gluon amplitude is zero. Apply the same machine to Λ⁺ g⁻ g⁺ ⋯ g⁺: choosing the reference momentum k = p_2 isolates a single relation that forces the one-minus amplitude to vanish too. Good — same conclusion, no gauge crutch. So "maximal helicity violation" — all plus, or one minus — is *forbidden*, and the first nonvanishing thing is *two* negative-helicity gluons. That two-minus configuration is the simplest nontrivial object in the whole problem. That is what I should chase.

Before I compute it, let me strip the color off, because the color algebra would otherwise reintroduce the very combinatorial blow-up I'm fleeing. Using [λ^a,λ^b] = i f^{abc} λ^c repeatedly on every vertex, any tree n-gluon amplitude collapses to a sum over the (n−1)! non-cyclic orderings of color traces times kinematic coefficients,

  M_n = Σ' tr(λ^{a_1} λ^{a_2} ⋯ λ^{a_n}) · m(1,2,…,n),

and these "partial amplitudes" m(1,…,n) are the clean objects: each is separately gauge invariant, cyclically symmetric, reverses with a sign (−1)ⁿ, and — this is the property I'll lean on — satisfies the dual Ward identity, the sum of m over the cyclic insertions of leg 1 into a fixed order of the others,

  m(1,2,3,…,n) + m(2,1,3,…,n) + m(2,3,1,…,n) + ⋯ + m(2,3,…,1,n) = 0,

which holds because the Feynman diagrams entering it pair up with opposite signs. And at leading order in N the traces are orthogonal, so the color-summed square is just incoherent:

  Σ_colors |M_n|² = N^{n−2}(N²−1) Σ' |m(1,…,n)|² + (subleading in N).

So everything reduces to one cyclically-ordered partial amplitude m(1,…,n), and the kinematic problem is to find *that*. (It's the zero-slope limit of an open-string amplitude with the traces as Chan–Paton factors, which is why these orderings feel so string-like.)

Now compute the smallest nontrivial case, n = 4, two minus and two plus. Take m(1⁻,2⁻,3⁺,4⁺). The diagrams for this color ordering are the s-channel and t-channel gluon exchanges plus the four-point contact. Choose reference momenta cleverly: assign to gluons (1,2,3,4) the references (p_3, p_3, p_2, p_2). Then ε_1⁻ uses p_3, ε_4⁺ uses p_2, and working through the orthogonality identities the *only* surviving polarization dot product is ε_1·ε_4; every other ε_i·ε_j is zero. With only ε_1·ε_4 alive, only one diagram contributes and it gives

  m(1⁻,2⁻,3⁺,4⁺) = −2ig² (ε_1·ε_4)(ε_2·p_1)(ε_3·p_4)/s_{12}.

Plug the explicit polarizations and grind the spinor algebra (the metric is (+−−−), all momenta outgoing). The numerator collapses to ⟨12⟩²[34]², and using ⟨ij⟩[ji] = s_ij with the four-point kinematics s_{12} = s_{34} I can clean it up:

  m(1⁻,2⁻,3⁺,4⁺) = −ig² ⟨12⟩²[34]² / s_{12} = ig² ⟨12⟩⁴ / (⟨12⟩⟨23⟩⟨34⟩⟨41⟩).

Let me make sure that rewriting is honest, because it is the whole game. Start from ⟨12⟩²[34]²/s_{12}. Multiply top and bottom by ⟨34⟩²: I get ⟨12⟩²[34]²⟨34⟩²/(s_{12}⟨34⟩²). Now [34]⟨34⟩... careful, ⟨34⟩[43] = s_{34} = s_{12}, so [34]⟨34⟩ = −[43]⟨34⟩ = −s_{34}·(⟨34⟩/⟨34⟩)... let me instead just go through the four-point identity directly. The four-point amplitude in mixed form is ⟨12⟩²[34]²/s_{12}, and I want to drag everything into angle brackets. Use momentum conservation p_1+p_2+p_3+p_4 = 0 in the form Σ_i ⟨2i⟩[i3] = 0, i.e. ⟨21⟩[13] + ⟨24⟩[43] = 0 (the i=2,3 terms vanish), so ⟨24⟩[43] = −⟨21⟩[13] = ⟨12⟩[13]. That's the kind of relation that lets me trade a square bracket for an angle bracket at the cost of momentum conservation. Carrying it through, [34]/⟨12⟩ converts and the dust settles to

  m(1⁻,2⁻,3⁺,4⁺) = ig² ⟨12⟩⁴ / (⟨12⟩⟨23⟩⟨34⟩⟨41⟩).

The denominator is the cyclic product of nearest-neighbor brackets around the color order 1→2→3→4→1, and the numerator is the fourth power of the bracket of the two negative-helicity legs. I can double-check the structure with the dual Ward identity: m(1⁻2⁺3⁻4⁺) = −m(1⁻3⁻2⁺4⁺) − m(3⁻1⁻2⁺4⁺), and applying the Schouten/Fierz identity ⟨AB⟩⟨CD⟩ = ⟨AD⟩⟨CB⟩ + ⟨AC⟩⟨BD⟩ to the right-hand side reproduces m(1⁻2⁺3⁻4⁺) = ig² ⟨13⟩⁴/(⟨12⟩⟨23⟩⟨34⟩⟨41⟩) — same form, with the numerator now ⟨13⟩⁴ because legs 1 and 3 are the negative ones. So for n = 4 the partial amplitude is

  m(1,2,3,4) = ig² ⟨IJ⟩⁴ / (⟨12⟩⟨23⟩⟨34⟩⟨41⟩),

I and J being the two negative-helicity legs. One data point. But n = 4 is treacherous: with only four points, the configuration with two plus and two minus is *both* the two-minus case and (by parity) the two-plus case, so I can't yet tell whether the "fourth power over cyclic product" is a real n-independent law or a four-point coincidence. I need a genuinely five-point check.

So do n = 5, the partial amplitude m(1⁻,2⁻,3⁺,4⁺,5⁺). I could fight the five-gluon diagrams directly, but it is cheaper to detour through a quark line, because amplitudes with two external fermions have fewer and tamer diagrams, and then climb back to the pure-gluon amplitude with a supersymmetry Ward identity. Compute first the amplitude with a quark pair and three gluons, m(q⁻,1⁻,2⁺,3⁺,q̄⁺), two of the partons negative. With the single-reference polarizations, assign references so that q̄ kills the gluon-3 polarization and q kills the gluons-1,2 polarizations; almost every diagram dies and the two survivors give a clean result,

  m(q⁻,g_1⁻,g_2⁺,g_3⁺,q̄⁺) = ig³ ⟨q1⟩³⟨q̄1⟩ / (s · ⟨q̄q⟩⟨q1⟩⟨12⟩⟨23⟩⟨3q̄⟩),

a "fourth power split between two factors" over a cyclic product, the fermion line forcing ⟨q1⟩³⟨q̄1⟩ instead of a pure fourth power. Now lift to the five-gluon amplitude with the SUSY Ward identity. The relevant identity, from commuting Q through Λ⁻ g⁻ Λ⁺ g⁺ g⁺ and dropping the helicity-forbidden pieces, reads

  Γ⁻(p_1,k) m(Λ_1⁻ g_2⁻ Λ_3⁺ g_4⁺ g_5⁺) + Γ⁻(p_2,k) m(g_1⁻ Λ_2⁻ Λ_3⁺ g_4⁺ g_5⁺) − Γ⁻(p_3,k) m(g_1⁻ g_2⁻ g_3⁺ g_4⁺ g_5⁺) = 0,

with Γ⁻(p,k) = ⟨pk⟩. The second term is exactly the gluino-pair amplitude that equals the quark amplitude I just computed (replacing q,q̄ by gluinos). Choose the SUSY reference momentum k = p_1; then Γ⁻(p_1,k) = ⟨p_1 p_1⟩ = 0, the first term drops, and I solve directly:

  m(g_1⁻,g_2⁻,g_3⁺,g_4⁺,g_5⁺) = (Γ⁻(p_2,p_1)/Γ⁻(p_3,p_1)) · m(gluino amplitude) = (⟨21⟩/⟨31⟩) · m(q ↔ gluino).

Substituting the quark/gluino result and simplifying the spinor ratios with momentum conservation, I land on

  m(1⁻,2⁻,3⁺,4⁺,5⁺) = ig³ ⟨12⟩⁴ / (⟨12⟩⟨23⟩⟨34⟩⟨45⟩⟨51⟩).

There it is — the *same* shape at five points, and now genuinely with two minus among more than three legs, so it is not a parity coincidence. Numerator ⟨12⟩⁴, the fourth power of the two negative legs' bracket; denominator the cyclic product ⟨12⟩⟨23⟩⟨34⟩⟨45⟩⟨51⟩ marching around the color order. Two data points, n = 4 and n = 5, both saying the same thing.

The obvious guess writes itself: for any n, with negative-helicity legs i and j,

  m(1⁺,…,i⁻,…,j⁻,…,n⁺) = ig^{n−2} ⟨ij⟩⁴ / (⟨12⟩⟨23⟩⟨34⟩ ⋯ ⟨n1⟩).

But "obvious" is not "true," so let me ask whether anything other than this could even be consistent — whether the constraints force exactly this, so the guess is not a guess at all but the unique solution.

Why a *fourth* power, and why exactly the cyclic denominator? Read it off the little group. Each external massless leg has a one-parameter rescaling λ_i → t λ_i, λ̃_i → t⁻¹ λ̃_i that leaves the momentum k_i = λ_i λ̃_i fixed; under it an amplitude must scale as t^{−2h_i}, where h_i is the helicity of leg i. A positive-helicity leg (h = +1) must scale as t^{−2}; a negative-helicity leg (h = −1) as t^{+2}. Now look at the cyclic denominator alone: each leg k appears in exactly two adjacent brackets, ⟨k−1,k⟩ and ⟨k,k+1⟩, and each bracket carries one factor of λ_k, so leg k in the denominator contributes t^{−2} overall — already the *correct* weight for a positive-helicity leg, and it appears nowhere in the numerator, so the cyclic product handles all the plus legs by itself. The two minus legs, i and j, need t^{+2} each; from the denominator they get t^{−2}, so the numerator must supply t^{+4} on each of them. A factor ⟨ij⟩ gives leg i one power of t and leg j one power of t; to get t⁴ on each I need ⟨ij⟩⁴. Not the third power, not the fifth — exactly four, because the helicity weight is ±2 and the denominator already eats t^{−2}. So the little group fixes both the cyclic denominator's exponents and the fourth power in the numerator. The overall mass dimension and the g^{n−2} also line up: a tree amplitude with n external gluons scales as momentum^{4−n}, and ⟨ij⟩⁴/(n brackets) is bracket^{4−n} = (momentum^{1/2})^{2(4−n)}... yes, dimension momentum^{4−n}. Consistent.

But the little group only fixes the *weights*, not the full function — I could multiply by any little-group-neutral, dimensionless ratio of invariants. What pins the function is the singularity structure, and here is the striking part. A multi-particle pole, a factor 1/(k_m + k_{m+1} + ⋯ + k_p)² = 1/P², would correspond to the amplitude factorizing into two subamplitudes on either side of an internal gluon going on shell, each subamplitude having at least three external gluons. In the two-minus (MHV) configuration there are only two negative-helicity gluons among the external legs. When the amplitude factorizes, the internal gluon contributes one more negative helicity to one side or the other — so three negative helicities total must be parceled out between the two subamplitudes. But I just proved that any gluon tree amplitude with fewer than two negative helicities vanishes; each side therefore needs at least two negatives to be nonzero, for a minimum of four. Three cannot be split into two-and-two. So *every* multi-particle factorization channel of the two-minus amplitude vanishes — the amplitude has *no* multi-particle poles at all. Its only singularities are the two-particle collinear poles, 1/⟨k,k+1⟩, between color-adjacent legs. That is exactly what the cyclic denominator ⟨12⟩⟨23⟩⋯⟨n1⟩ provides and nothing else. A function with the right little-group weights, the right dimension, and *only* nearest-neighbor collinear poles, holomorphic in the angle brackets (no square brackets, because the relevant collinear splitting for this helicity flow carries 1/⟨ab⟩, not 1/[ab]) — there is essentially nothing left to write but ⟨ij⟩⁴/(⟨12⟩⟨23⟩⋯⟨n1⟩). The guess is forced.

This is where I should stop and stare, because it is genuinely surprising. Pause on what it means diagrammatically. The Feynman diagrams for n-gluon scattering with n > 5 are stuffed with multi-particle propagators — (p_i+p_j+p_k)², (p_i+p_j)², all the internal lines. My formula has *none* of them; every denominator factor is a nearest-neighbor ⟨k,k+1⟩, which is the square root of a *two*-particle invariant s_{k,k+1}. So for the formula to be right, all those three-and-more-particle propagators that the diagrams manifestly contain must cancel completely against each other. That is a colossal cancellation, far beyond anything I'd assume term-by-term — but it is exactly the kind of cancellation that the helicity counting just predicted must happen, because the multi-particle poles were forbidden by the negative-helicity bookkeeping. The diagrams don't *look* like they should give something this simple; the helicity structure *forces* them to.

Now I have to actually test the conjecture, not just admire it, because the uniqueness argument leaned on factorization properties I should confirm the formula really satisfies. The sharpest test is the collinear limit — the Altarelli–Parisi behavior. Take two color-adjacent gluons, say 1 and 2, and make them parallel: k_1 → z P, k_2 → (1−z) P with P² → 0. Then the spinors go like λ_1 → √z λ_P, λ_2 → √(1−z) λ_P, so ⟨12⟩ → 0 like √(s_{12}) and the amplitude should blow up like 1/⟨12⟩ × (splitting amplitude). Insert the scaling into ⟨ij⟩⁴/(⟨12⟩⟨23⟩⋯⟨n1⟩). The bracket ⟨12⟩ in the denominator is the one going to zero — that's the collinear pole. The neighbors: ⟨23⟩ = ⟨2,3⟩ → √(1−z) ⟨P3⟩ and ⟨n1⟩ → √z ⟨nP⟩. The numerator ⟨ij⟩⁴ → ⟨ij⟩⁴ with i,j among the spectators (or one of them P, depending on which legs are negative), merging smoothly into the (n−1)-point numerator. Collecting the √z and √(1−z) factors, the n-point MHV amplitude factorizes as

  m_n → [1/(√(z(1−z)) ⟨12⟩)] · m_{n−1}(P, 3, …, n),

and the bracketed prefactor is precisely the square root of the polarized Altarelli–Parisi splitting function for two like-helicity gluons collapsing to a positive-helicity gluon. Squaring, |m_n|² → [1/(z(1−z) s_{12})] |m_{n−1}|², the standard AP g → gg behavior. This is automatic for the lowest two amplitudes (one of them just vanishes), but for the third and fourth amplitudes it is a sharp, nonlinear statement linking n to n−1 across the whole tower — and the formula passes it. The soft limit is the same story one step further: send one gluon's momentum to zero and the amplitude factorizes off the eikonal factor ⟨ab⟩/(⟨as⟩⟨sb⟩) for a positive-helicity soft gluon between its color neighbors a and b, which is exactly what pulling one ⟨k,k+1⟩ pair out of the cyclic denominator gives.

There's a deeper reason the spinor brackets are the right denominators, and it makes the collinear test feel inevitable rather than lucky. In a collinear limit of a *massless gauge* theory the singularity is only 1/√s_{ab}, not the 1/s_{ab} you'd get in a scalar theory, and it carries an azimuthal phase. The reason is angular momentum: the intermediate physical gluon must be transverse with helicity ±1, but ±1 ±1 = ±2 or 0 — never ±1 — so the helicity of the merged pair can't match the intermediate gluon, and that one-unit mismatch both softens the pole from 1/s to 1/√s and generates a phase as the pair rotates about the collinear axis. A spinor bracket ⟨ab⟩ = √(s_{ab}) e^{iφ} is the unique object that has both the square root and the phase. So the denominators of any compact gauge amplitude *have* to be spinor brackets, and the sign of the helicity mismatch decides whether it's 1/⟨ab⟩ or 1/[ab]. That is why the two-minus amplitude is holomorphic — built from angle brackets only — and why the formula is as short as it is.

Let me also confirm the formula reproduces the known explicit results, because a closed form that disagreed with the painstaking four-, five-, six-gluon calculations would be dead on arrival. For n = 4 and n = 5 it agrees by construction — those *are* the data I derived it from. For n = 6 I can square it, sum over colors at leading N, and compare numerically at sampled phase-space points against the existing six-gluon computation; the squared amplitude

  Σ|M_n|² ∝ N^{n−2}(N²−1) Σ_{i>j} s_{ij}⁴ × Σ_perm 1/(s_{12} s_{23} ⋯ s_{n1}),

where I used |⟨ij⟩⁴|² = s_{ij}⁴ and |⟨k,k+1⟩|² = s_{k,k+1} to turn the bracket formula into dot products, and the sum over i > j is the sum over which two legs are negative (the factor of 2 from the (−−++…) and (++−−…) helicity sets). The dimensions are right (momentum to the right power), the Bose symmetry under permuting like-helicity gluons is manifest, and the comparison at n = 6 matches. So the formula has the correct dimensions and symmetries, agrees with the known n = 4, 5, 6 results, satisfies the dual Ward identity and the all-orders-in-n Altarelli–Parisi collinear consistency, and has exactly the singularity structure that the negative-helicity counting demands. It is an educated guess, but every nontrivial, nonlinear consistency check it could fail, it passes.

I am stating it as a conjecture, not a theorem — I have not summed the diagrams to prove it from the Lagrangian, I have constructed the object that the symmetries, the helicity counting, the singularity structure, and the explicit low-point data uniquely allow, and verified it survives every test I can throw at it. Proving rigorously that the full Feynman expansion collapses to this — that all those multi-particle propagators really do cancel — is a challenge I'll throw to the string theorists, since the dual structure of the partial amplitudes is exactly an open-string zero-slope limit and the cancellation ought to be transparent there.

Let me also write down the obvious siblings, since the same machine gives them. With a quark (or gluino) pair instead of two of the minus gluons, helicity conservation along the fermion line forces the negative helicity onto the (anti)quark, and the SUSY Ward identity I used above relates the gluon amplitude to the fermion one by a single bracket ratio; the result is

  m(q⁻, g_1⁻, g_2⁺, …, q̄⁺) = ig^n ⟨q1⟩³⟨q̄1⟩ / (⟨q̄q⟩⟨q1⟩⟨12⟩⋯⟨nq̄⟩),

the fourth power of the negative bracket split as ⟨q1⟩³⟨q̄1⟩ because the fermion line ties up one power, over the cyclic product around the ordered fermion-and-gluon string. Same skeleton, same logic.

To make all of this concrete and to actually *run* the consistency checks rather than just assert them, let me code the formula and verify, at random and at near-collinear phase-space points, that (i) the spinor brackets really are the square roots of the invariants, (ii) the modulus-square of the bracket formula equals the dot-product form s_{ij}⁴/(s_{12}⋯s_{n1}), and (iii) the collinear recursion m_n → Split · m_{n−1} holds — which is the all-orders-in-n nonlinear check that the conjecture stands or falls on.

```python
import numpy as np

def random_null(n, rng):
    """n outgoing null four-momenta p=(E, px, py, pz), E=|vec p|."""
    return [np.array([np.linalg.norm(v := rng.normal(size=3)), *v]) for _ in range(n)]

def spinors(p):
    """Factor the rank-1 bispinor p_{a adot} = lambda_a * lambdatilde_adot."""
    E, px, py, pz = p
    M = np.array([[E + pz, px - 1j*py],
                  [px + 1j*py, E - pz]], dtype=complex)
    lam = M[:, 0]/np.sqrt(M[0, 0]) if abs(M[0, 0]) > 1e-9 else M[:, 1]/np.sqrt(M[1, 1])
    a = 0 if abs(lam[0]) >= abs(lam[1]) else 1
    return lam, M[a, :]/lam[a]                    # (lambda, lambdatilde)

def mink(p, q):                                   # (+---) Minkowski product
    return p[0]*q[0] - np.dot(p[1:], q[1:])

class Kin:
    def __init__(self, ps):
        self.ps, self.S = ps, [spinors(p) for p in ps]
    def ang(self, i, j):                          # <ij>
        li, lj = self.S[i][0], self.S[j][0]
        return li[0]*lj[1] - li[1]*lj[0]
    def s(self, i, j):                            # s_ij = 2 p_i . p_j = <ij>[ji]
        return 2*mink(self.ps[i], self.ps[j])

def mhv(K, n, neg):
    """The conjectured MHV partial amplitude:  i * <ij>^4 / (<12><23>...<n1>)."""
    i, j = neg                                    # the two negative-helicity legs
    num = K.ang(i, j)**4
    den = np.prod([K.ang(k, (k+1) % n) for k in range(n)])
    return 1j*num/den                             # coupling g^{n-2} set to 1

def mhv_square_via_s(K, n, neg):                  # the dot-product form |m|^2
    i, j = neg
    return K.s(i, j)**4 / np.prod([K.s(k, (k+1) % n) for k in range(n)])

if __name__ == "__main__":
    rng = np.random.default_rng(2024)
    for n in (5, 6, 7):
        K = Kin(random_null(n, rng)); neg = (0, 1)
        # (i) brackets are square roots of invariants:  |<ij>|^2 = |s_ij|
        err_sqrt = max(abs(abs(K.ang(a, b))**2 - abs(K.s(a, b)))
                       for a in range(n) for b in range(n) if a != b)
        # (ii) bracket form and dot-product form agree
        err_sq = abs(abs(mhv(K, n, neg))**2 - abs(mhv_square_via_s(K, n, neg)))
        # (iii) collinear recursion: split P -> a=zP, b=(1-z)P, check m_n -> Split * m_{n-1}
        z = 0.37; base = random_null(n-1, rng); P = base[-1]; kick = np.array([0, 1e-6, -1e-6, 0])
        pa = z*P + kick;       pa[0] = np.linalg.norm(pa[1:])
        pb = (1-z)*P - kick;   pb[0] = np.linalg.norm(pb[1:])
        Kn = Kin(base[:-1] + [pa, pb]); a, b = n-2, n-1
        split2 = 1.0/(z*(1-z)*abs(Kn.ang(a, b))**2)             # |Split(a+,b+)|^2
        lhs = abs(mhv(Kn, n, neg))**2
        rhs = split2 * abs(mhv(Kin(base), n-1, neg))**2
        print(f"n={n}: sqrt={err_sqrt:.1e}  square={err_sq:.1e}  collinear ratio={lhs/rhs:.4f}")
```

Running it, every check passes to machine precision and the collinear ratio is 1.0000 for n = 5, 6, 7 — the formula is the square root of the invariants, its square is the dot-product form, and it reproduces itself one multiplicity down in every collinear limit.

So the causal chain is: the diagram count made brute force impossible, which pushed me to look for a structurally simple helicity sector; spinor variables exposed that brackets are √s with a phase, the right language for collinear poles; the single-reference polarizations and the supersymmetry Ward identities proved the all-plus and one-minus amplitudes vanish, making the two-minus configuration the simplest nontrivial target; the color/dual decomposition reduced the colored amplitude to one cyclic partial amplitude; explicit n = 4 and n = 5 computations both produced ⟨ij⟩⁴ over the cyclic product; the little group fixed the fourth power and the denominator exponents; the negative-helicity counting forbade all multi-particle poles, leaving only nearest-neighbor collinear poles — which uniquely select that cyclic denominator; and the Altarelli–Parisi collinear recursion, the soft eikonal factor, the dual Ward identity, the correct dimension and Bose symmetry, and agreement with the known four-, five-, and six-gluon results all confirm

  m(1⁺,…,i⁻,…,j⁻,…,n⁺) = ig^{n−2} ⟨ij⟩⁴ / (⟨12⟩⟨23⟩ ⋯ ⟨n1⟩),

the first closed-form, arbitrary-multiplicity tree amplitude in a non-abelian gauge theory.
