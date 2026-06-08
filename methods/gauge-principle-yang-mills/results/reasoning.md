Let me start where the discomfort actually is. Isotopic spin is, by everything we know about the strong force, an excellent symmetry: the proton and neutron are two states of one nucleon, the nuclear forces don't care which is which, and I can rotate the doublet into itself with a 2×2 unitary matrix of determinant one — an element of SU(2) — without changing any physics. The total isotopic spin is conserved. Good. But look at how that symmetry is actually stated. I pick *one* rotation S and apply it everywhere in spacetime, the same matrix at every point. And here is the thing that won't let me rest: the symmetry itself is telling me that the choice of which internal direction counts as "the proton" is physically meaningless — no internal direction is preferred, that's the whole content of the invariance. Yet the global rule says that once I've fixed the orientation of the isotopic frame here, at this point, I am no longer free to fix it differently over there. A convention with no physical content, adopted at one point, is somehow binding across the entire universe. That's strange. If the orientation is truly meaningless, why should the relative orientation of the isospin frames at two separated points be a fixed, determined thing rather than something I can choose freely, independently, point by point?

And there's a second pressure pushing the same way. We believe in localized fields — everything built up point by point, no rigid action-at-a-distance scaffolding. A symmetry I can only define globally, that demands one matrix applied identically at all points simultaneously, doesn't really fit that picture. A local field theory ought to let me make the symmetry transformation locally too.

So let me just *demand* it. Let me ask that the physics be invariant not under a single global SU(2) rotation but under an independent rotation at every spacetime point: ψ(x) → S(x)ψ(x), with S(x) an arbitrary SU(2)-valued function of x. If I can pull this off, the relative isospin orientation between two points becomes physically meaningless, exactly as the symmetry seems to want.

Now, is there a precedent for promoting a global symmetry to a local one? There is, and it's electromagnetism, though it's not usually told this way. A charged matter field is a complex wavefunction ψ, and its overall phase is meaningless: ψ → e^{iα}ψ does nothing. Weyl saw — after the false start where he tried to rescale *lengths* locally and the path-dependence killed it — that in quantum mechanics you should rescale the *phase* locally instead: ψ(x) → e^{iα(x)}ψ(x). And the beautiful thing is what that costs you and what it buys. It costs you the free derivative term: ∂_μ acting on e^{iα(x)}ψ doesn't just rotate the phase, it also brings down a derivative of α,

  ∂_μ(e^{iα}ψ) = e^{iα}(∂_μψ + i(∂_μα)ψ),

so ψ̄γ^μ∂_μψ is no longer invariant — there's a leftover (∂_μα) term with nothing to cancel it. And the cure is to introduce a vector field A_μ and a *covariant* derivative D_μ = ∂_μ − ieA_μ, with A_μ arranged to transform inhomogeneously, A_μ → A_μ + (1/e)∂_μα, so that the extra (1/e)∂_μα·(−ie) = −i∂_μα exactly eats the +i∂_μα from the derivative. Then D_μψ → e^{iα}(D_μψ), transforming just like ψ itself, and ψ̄γ^μD_μψ is locally invariant. The local symmetry *forced* the electromagnetic field into existence and fixed how it couples. That's the template. That's what I want to copy.

So copy it. Replace the abelian phase e^{iα(x)} by the non-abelian S(x) ∈ SU(2). The free Dirac kinetic term ψ̄γ^μ∂_μψ — was it even invariant under the global version? Under constant S, yes: ψ̄ → ψ̄S†, ψ → Sψ, and since S is constant it slides right through ∂_μ, with S†S = 1, so the kinetic term is untouched. That's *why* global isotopic invariance works and demands nothing new. But now make S depend on x. Then

  ∂_μ(S(x)ψ) = (∂_μS)ψ + S(∂_μψ),

and that first term, (∂_μS)ψ, is the wrecker — the exact analog of the (∂_μα) term in the abelian case. The kinetic term is broken. Fine, this is expected; this is the whole point. The local symmetry breaks the bare derivative, and I have to repair it with a compensating field.

So introduce a compensating field — but now it can't be a single real function. The thing it has to cancel, (∂_μS)ψ, lives in the same 2×2 matrix space that S does; under SU(2) the "directions" you can rotate in are three (the group has three generators T^a = σ^a/2, the isotopic angular-momentum matrices, with [T^a,T^b] = iε^{abc}T^c). So the compensating field has to be matrix-valued: A_μ = A_μ^a T^a, three vector fields, one for each generator, packaged as a Hermitian traceless 2×2 matrix. And I write the covariant derivative in the same form as before,

  D_μ = ∂_μ − igA_μ,

with g the coupling that plays the role e played. The demand is identical to the abelian demand: D_μψ should transform like ψ, i.e. D_μψ → S(D_μψ). Let me just impose that and solve for how A_μ must transform.

I want (∂_μ − igA'_μ)(Sψ) = S(∂_μ − igA_μ)ψ for all ψ. Expand the left side:

  ∂_μ(Sψ) − igA'_μSψ = (∂_μS)ψ + S∂_μψ − igA'_μSψ.

The right side is S∂_μψ − igSA_μψ. The S∂_μψ terms match and cancel. What's left:

  (∂_μS)ψ − igA'_μSψ = −igSA_μψ.

This has to hold for every ψ, so as a matrix equation:

  −igA'_μS = −igSA_μ − (∂_μS),

  A'_μS = SA_μ + (1/(ig))∂_μS = SA_μ − (i/g)∂_μS,

and multiplying on the right by S⁻¹,

  A'_μ = SA_μS⁻¹ − (i/g)(∂_μS)S⁻¹.

There it is. The first piece, SA_μS⁻¹, is just the field rotating homogeneously — that's what you'd naively expect of a matrix field. The second piece, −(i/g)(∂_μS)S⁻¹, is the inhomogeneous term, the non-abelian descendant of the (1/e)∂_μα in electromagnetism. Let me sanity-check the abelian limit: if S = e^{iα} commutes with everything, SA_μS⁻¹ = A_μ, and −(i/g)(∂_μS)S⁻¹ = −(i/g)(i∂_μα)e^{iα}e^{−iα} = (1/g)∂_μα. So A'_μ = A_μ + (1/g)∂_μα — exactly Weyl's electromagnetic gauge transformation, with g in the role of e. Good, it reduces correctly. (One can also write the inhomogeneous term as +(i/g)S∂_μS⁻¹, using ∂_μ(S⁻¹S) = 0 ⇒ ∂_μS⁻¹ = −S⁻¹(∂_μS)S⁻¹; same object.) So D_μψ transforms covariantly, ψ̄γ^μD_μψ is locally invariant, and I have my coupling between matter and the field. The first part is mathematically easy, just as I expected from the abelian case.

Now I need the field's own dynamics — a kinetic term for A_μ, which means I need the analog of F_μν. In electromagnetism F_μν = ∂_μA_ν − ∂_νA_μ, the curl, and it's gauge invariant: under A_μ → A_μ + (1/e)∂_μα the extra piece is (1/e)(∂_μ∂_να − ∂_ν∂_μα) = 0 by symmetry of mixed partials. So my first guess is the obvious one: just take the same curl, F_μν = ∂_μA_ν − ∂_νA_μ, now with matrix-valued A. Let me check how it transforms. I need ∂_μA'_ν − ∂_νA'_μ with A'_ν = SA_νS⁻¹ − (i/g)(∂_νS)S⁻¹. When I differentiate SA_νS⁻¹ I get (∂_μS)A_νS⁻¹ + S(∂_μA_ν)S⁻¹ + SA_ν(∂_μS⁻¹), and when I differentiate the inhomogeneous term I get second derivatives of S. The clean piece I want is S(∂_μA_ν − ∂_νA_μ)S⁻¹. But it is buried under a pile of extra terms: derivatives of S hitting A, and the second-derivative terms from the inhomogeneous part.

Let me see whether the extras cancel. In the abelian case they did: SA_νS⁻¹ = A_ν, the (∂_μS)A_νS⁻¹ + A_ν(∂_μS⁻¹) pieces collapse because everything commutes and S⁻¹S = 1 with ∂_μS⁻¹ = −S⁻¹(∂_μS)S⁻¹ leaving things that telescope, and the second derivatives of α cancel by symmetry. But here S does *not* commute with A_ν. So a term like (∂_μS)A_νS⁻¹ is not equal to S A_ν S⁻¹ times anything simple; it sits there. And when I antisymmetrize in μ↔ν, the second-derivative-of-S terms cancel (mixed partials), but the cross terms — (∂_μS) paired with A_ν, and the inhomogeneous (∂_νS)S⁻¹ paired with the derivative of the homogeneous part — do *not* cancel. They are quadratic in the fields, products of (∂S) with A and of (∂S)S⁻¹ with itself. The naive curl does not transform homogeneously. There's a residue.

This is exactly the wall I keep hitting. The first steps are easy and then the next step throws up these complicated leftover terms that won't go away, and they get messier the harder I push. I've come back to this several times and each time the formulae proliferate and I give up. Let me actually stare at what the leftover terms *are* this time instead of just despairing at them. They're quadratic — products of two fields — and when I track the field strength through into the Lagrangian and the equations of motion, cubic ones appear too. Undesired, complicated, quadratic and cubic. They're all of that one type: products of A's, products coming from the non-commutativity, things that would be identically zero if S and A commuted.

So the question flips. Instead of asking "how do I make these terms cancel against each other" — they won't, they're irreducibly there because the group doesn't commute — let me ask: could I cancel them by *adding* a term to F_μν at the start? If the unwanted terms are quadratic in A, then a quadratic term built into the definition of F_μν, one that transforms so as to generate the *opposite* of those leftovers, would kill them. I'm not trying to remove the non-commutativity; I'm trying to feed it a counterterm of its own kind.

What quadratic, antisymmetric-in-μν, matrix object can I build from A_μ and A_ν? The symmetric product A_μA_ν + A_νA_μ is no good — it's symmetric. The antisymmetric one is the commutator, [A_μ,A_ν] = A_μA_ν − A_νA_μ. And the commutator is *exactly* the object that measures non-commutativity — it vanishes identically in the abelian case, which is precisely the case where my naive curl already worked and I want my correction to disappear. That's the tell. Let me try

  F_μν = ∂_μA_ν − ∂_νA_μ − ig[A_μ,A_ν]

and see if the commutator term's transformation residue cancels the curl's residue.

But rather than grind through the brute-force transformation of every piece, let me get F_μν the clean way, the way the covariant derivative itself hands it to me. I already know D_μψ → S D_μψ for every ψ. That means the *operator* D_μ transforms as D_μ → S D_μ S⁻¹ (it acts on ψ, and Sψ → S(D_μψ) = (SD_μS⁻¹)(Sψ)). So any combination of D's that I build will transform homogeneously by conjugation. The simplest gauge-covariant object made from two covariant derivatives is their commutator, [D_μ, D_ν] — and because it's built purely from the D's, it must go to S[D_μ,D_ν]S⁻¹. Let me compute it acting on ψ:

  [D_μ, D_ν]ψ = (∂_μ − igA_μ)(∂_ν − igA_ν)ψ − (μ ↔ ν).

Expand the first product:

  ∂_μ∂_νψ − ig∂_μ(A_νψ) − igA_μ∂_νψ + (−ig)²A_μA_νψ
  = ∂_μ∂_νψ − ig(∂_μA_ν)ψ − igA_ν∂_μψ − igA_μ∂_νψ − g²A_μA_νψ.

Subtract the μ↔ν term. The ∂_μ∂_νψ cancels (symmetric). The derivative-on-ψ terms: −igA_ν∂_μψ − igA_μ∂_νψ minus −igA_μ∂_νψ − igA_ν∂_μψ — these cancel completely. What survives is purely multiplicative, no derivatives hitting ψ:

  [D_μ,D_ν]ψ = [−ig(∂_μA_ν − ∂_νA_μ) − g²(A_μA_ν − A_νA_μ)]ψ
            = −ig[(∂_μA_ν − ∂_νA_μ) − ig(A_μA_ν − A_νA_μ)]ψ,

where I pulled out a common −ig. Let me be careful with that factor: (−ig)(−ig) = (−i)²g² = −g², so −g²[A_μ,A_ν] is exactly (−ig)(−ig)[A_μ,A_ν]. Therefore −ig times −ig[A_μ,A_ν] gives the quadratic term with the correct sign. Factoring −ig out front:

  [D_μ,D_ν] = −ig(∂_μA_ν − ∂_νA_μ − ig[A_μ,A_ν]).

Define

  F_μν = ∂_μA_ν − ∂_νA_μ − ig[A_μ,A_ν],

so that [D_μ,D_ν] = −ig F_μν. And now the transformation property is free: since [D_μ,D_ν] → S[D_μ,D_ν]S⁻¹, and the −ig is just a constant,

  F_μν → S F_μν S⁻¹.

Homogeneous. Covariant. No inhomogeneous junk, no leftover quadratic terms — they're *absorbed into the definition* by the −ig[A_μ,A_ν] piece. That's the cancellation. The simple quadratic term, the commutator, introduced at the very beginning, miraculously kills all the undesired complicated terms that the naive curl threw off. This is the gold I kept walking past. The thing that wrecked every earlier attempt — the non-commutativity generating quadratic leftovers — is cured by a term that is *itself* built from the non-commutativity.

And look at what that extra term *is*, physically. In the abelian case [A_μ,A_ν] = 0, the term vanishes, F_μν = ∂_μA_ν − ∂_νA_μ is linear in A, and I recover electromagnetism exactly — the photon, chargeless, not coupling to itself. But here, for non-abelian SU(2), [A_μ,A_ν] ≠ 0, and F_μν contains a piece *quadratic* in A. The field strength depends nonlinearly on the potential. And when I build the kinetic term — by analogy with −¼F_μνF^{μν} of electromagnetism, the only Lorentz-invariant, gauge-covariant scalar I can form is Tr F_μνF^{μν}, which is invariant because F → SFS⁻¹ rotates inside the trace — the normalization Tr(T^aT^b) = ½δ^{ab} gives

  L_gauge = −½ Tr(F_μνF^{μν}) = −¼ F_μν^a F^{aμν}.

When I expand F = (∂A − ∂A) − ig[A,A], the cross term between (∂A) and [A,A] is *cubic* in A, and the [A,A]·[A,A] term is *quartic* in A. The field's own Lagrangian contains three-field and four-field self-couplings. The field interacts with itself. It carries the very isotopic spin that is its source, the way the photon would have to carry charge if it coupled to itself — only the photon doesn't, because electromagnetism is abelian and its [A,A] is zero. This is the qualitative break from electromagnetism: the non-abelian gauge field is necessarily self-interacting, and it is self-interacting *because* the group doesn't commute. The same fact — non-commutativity — that forced the [A,A] term into F_μν is the fact that makes the field its own source.

Let me make sure the whole assembly is consistent and read off its content. The locally-invariant Lagrangian is

  L = −¼ F_μν^a F^{aμν} + ψ̄(iγ^μ D_μ − m)ψ,  D_μ = ∂_μ − igA_μ.

The interaction term inside ψ̄iγ^μD_μψ is +g ψ̄γ^μA_μψ = +g ψ̄γ^μA_μ^a T^a ψ, the analog of the electron's coupling to the photon, with the isotopic-spin matrices T^a in place of the charge. Vary with respect to ψ̄ and I get the Dirac equation with the covariant derivative, (iγ^μD_μ − m)ψ = 0. Vary with respect to A and I get the field equation for A_μ. And here the self-interaction shows up as a striking fact about conservation. The matter current j_μ^a = ψ̄γ_μT^aψ — the isotopic-spin current of the nucleons alone — is *not* conserved by itself; its divergence picks up a term proportional to the cross-product of A with j, because the field rotates the current as it propagates. What *is* conserved is the total current, the matter current plus a contribution built from A and F — the field carries its own isotopic spin, and only when I add the field's isospin to the matter's isospin do I get a conserved total **T** = ∫ ℑ_4 d³x. The gauge field is a source of itself. In the abelian theory the photon's current contribution would be zero and the matter current alone is conserved; here it cannot be, because the gauge quanta are charged under the very symmetry they gauge.

Now, what are the quanta of this field? It's a vector field, so spin 1. It's a triplet under SU(2) — three components A^a — so isotopic spin 1. Its quanta therefore come in three isotopic states, and since they can be emitted and absorbed in transitions between the proton and the neutron, which differ in charge by one unit, the three quanta must carry electric charges +e, 0, and −e. Spin-one, isospin-one particles, one neutral and two oppositely charged. That's a clean, falsifiable prediction.

And there's a problem I can't make go away, and I should be honest that it's a real one. The mass. A mass term for the gauge field would be m²A_μ^aA^{aμ}, but that term is not invariant under the local transformation — under A_μ → SA_μS⁻¹ − (i/g)(∂_μS)S⁻¹ it picks up the inhomogeneous piece and is ruined, exactly as m²A_μA^μ is ruined in electromagnetism, which is why the photon is massless. So the gauge invariance that gave me everything also forbids a mass term, and the quanta of my field come out massless. Charged, massless, spin-one particles. For electromagnetism massless is fine — the photon is massless. But charged massless particles that couple to nucleons should have been seen, and they haven't been. I have no satisfactory answer to this; the principle is so clean and the structure so forced that I'm convinced the idea is right, but the mass of the quanta is genuinely open. I'll write it down as it stands — the local-symmetry construction is too compelling to set aside over a difficulty I can't yet resolve — and flag the mass question as unsolved.

Let me also be clear about what was forced and what was a choice, because almost nothing here was a free choice. The covariant derivative form D_μ = ∂_μ − igA_μ is forced by demanding D_μψ → SD_μψ. The inhomogeneous transformation of A_μ is forced by the same demand — solved, not posited. The matrix/triplet nature of A_μ is forced by SU(2) having three generators. The −ig[A_μ,A_ν] term in F_μν is forced by requiring F to transform homogeneously, and equivalently it falls straight out of [D_μ,D_ν]. The self-interaction, the masslessness, the charges and spin of the quanta — all consequences, none of them inputs. The single genuine modeling decision is the very first one: to demand the isotopic-spin symmetry hold *locally* rather than globally. Everything else is the unwinding of that demand.

So let me set down the final structure the local principle produces. The doublet transforms ψ → S(x)ψ with S(x) ∈ SU(2). The ordinary derivative is replaced by the covariant derivative

  D_μ = ∂_μ − ig A_μ,  A_μ = A_μ^a T^a,  T^a = σ^a/2,

chosen so that D_μψ → S D_μψ, which forces the compensating field to transform as

  A_μ → S A_μ S⁻¹ − (i/g)(∂_μ S) S⁻¹,

with the inhomogeneous second term reducing to A_μ → A_μ + (1/g)∂_μα in the abelian limit. The gauge-covariant field strength is

  F_μν = ∂_μ A_ν − ∂_ν A_μ − ig [A_μ, A_ν],

obtained from −ig F_μν = [D_μ, D_ν], transforming homogeneously as F_μν → S F_μν S⁻¹, and carrying the self-interaction term −ig[A_μ,A_ν] that has no counterpart in electromagnetism. The complete locally-invariant Lagrangian is

  L = −¼ F_μν^a F^{aμν} + ψ̄(iγ^μ D_μ − m)ψ,

whose F² term, expanded through the commutator, contains cubic and quartic self-couplings of the gauge field — the field is its own source, the total conserved isotopic spin includes the field's contribution, and the three spin-one, isospin-one quanta carry charges ±e and 0.

Trace the chain once more. A global internal symmetry fixes a meaningless orientation rigidly across all of spacetime; demanding it instead hold locally breaks the bare derivative through the (∂_μS)ψ term; the cure is a compensating field with a covariant derivative D_μ = ∂_μ − igA_μ and an inhomogeneously-transforming A_μ; the naive field strength fails to transform cleanly because SU(2) doesn't commute, throwing off quadratic terms; those are absorbed by a commutator term −ig[A_μ,A_ν], equivalently F_μν = [D_μ,D_ν]/(−ig); and that commutator term, nonzero precisely because the group is non-abelian, makes the field self-interacting — the decisive structural difference from electromagnetism, paid for by quanta that come out massless.
