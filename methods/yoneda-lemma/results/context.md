## Research question

In the language of categories, an object is given to us with no internal structure at all — only a name, and a web of arrows to and from the other objects of its category. We routinely *define* objects this way: a product `A × B` is whatever object admits a pair of projections through which every pair of maps `(f, g)` factors uniquely; a free group on a set `S` is whatever group through which every set-map out of `S` factors uniquely; a tensor product `U ⊗ V` is whatever object for which linear maps out of it correspond to bilinear maps out of `U × V`. In every case the object is specified entirely by a statement about its *maps*: "maps out of this object correspond, naturally, to such-and-such data."

This raises a sharp and basic question. If two objects have the same pattern of maps — if, for every other object `B`, the maps from `A` to `B` correspond exactly (and compatibly, as `B` varies) to the maps from `A'` to `B` — must `A` and `A'` be the same object? More ambitiously: is an object *completely determined* by the totality of arrows into it (or out of it)? Can we replace the opaque object by the concrete, set-valued bookkeeping of "who maps to whom," lose nothing, and recover the object back?

A solution would have to do two things. First, it would have to say exactly what data a "pattern of maps out of `A`, compatible with the category" amounts to — and show that this data is small and computable, not an unmanageable infinitude. Second, it would have to show that passing from an object to its pattern of maps is *faithful and full* with the correct variance: for outgoing maps, an arrow `A' → A` gives a comparison from the pattern of `A` to the pattern of `A'`; dually, for incoming maps, an arrow `A → A'` gives a comparison from the pattern of maps into `A` to the pattern of maps into `A'`. If both hold, then "the pattern of maps" is a perfect stand-in for the object: an object is determined up to isomorphism by it, and each compatible identification of patterns comes from a unique isomorphism of objects.

## Background

The setting is the apparatus of Eilenberg and Mac Lane (1945, "General theory of natural equivalences"): **categories**, **functors**, and **natural transformations**. A category `C` has objects and, for each ordered pair `(a, b)`, a hom-set `C(a, b)` of arrows, with associative composition and identities `1_a`. A functor `F : C → D` sends objects to objects and arrows to arrows, preserving identities and composition. A natural transformation `τ : S ⇒ T` between two functors `S, T : C → B` is a family of components `τ_c : Sc → Tc`, one per object `c`, such that for every arrow `f : c → c'` the square

```
        S c  --τ_c-->  T c
   S f |                | T f
        S c' --τ_c'-->  T c'
```

commutes. Naturality is the formal expression of "the same construction, made without any arbitrary choices" — the components are determined by a single uniform rule, so they cannot be tweaked object-by-object.

The original motivation for these definitions was exactly the phenomenon of an object related to another *canonically* versus *by an arbitrary choice*. A finite-dimensional vector space `V` is isomorphic to its dual `V*`, but only after choosing a basis; whereas `V` is isomorphic to its **double** dual `V**` with no choice at all. Eilenberg and Mac Lane's diagnostic example is the abelian-group analogue: for `D(G) = Hom(G, R/Z)` the character group, the double character group gives a map `t_G : G → D(D(G))`, `t_G(g)(χ) = χ(g)`, and this `t : I ⇒ DD` *is* a natural transformation — "a precise expression for the elementary observation that the definition of `t` depends on no artificial choices of bases, generators, or the like." For finite `G` it is an isomorphism. By contrast `G ≅ D(G)` also holds for finite abelian `G`, but that isomorphism depends on writing `G` as a product of cyclics and so is *not* natural. The whole point of the definitions was to make this distinction — canonical versus choice-dependent — into mathematics. Mac Lane records the resulting slogan of the founders: "'category' has been defined in order to be able to define 'functor' and 'functor' has been defined in order to be able to define 'natural transformation'."

Two further pieces of standing apparatus are load-bearing here.

**Hom-functors.** Fix an object `a` of a locally small category `C` (locally small = every `C(a, b)` is an honest set). Then `b ↦ C(a, b)` is a functor `C(a, −) : C → Set`: an arrow `g : b → b'` induces the function `C(a, g) : C(a, b) → C(a, b')` given by post-composition `p ↦ g ∘ p`. This functor is "how `a` sees the category": it records every arrow out of `a` and how those arrows transform under maps of the target. Dually, `b ↦ C(b, a)`, `C(−, a) : C^op → Set`, records every arrow *into* `a` (a *presheaf* on `C`), with `g : b' → b` acting by pre-composition `p ↦ p ∘ g`.

A functor `X : C → Set` is called **representable** when `X ≅ C(a, −)` for some object `a`; the pair of `a` and the isomorphism is a **representation**, and `a` is the *representing object*. Up to isomorphism, a representable functor is just a hom-functor. Representability is the abstract face of every universal-property definition: the Sierpiński space `S` represents "open subsets of `−`" (continuous maps `X → S` correspond to open subsets of `X`); the polynomial ring `Z[x₁, …, xₙ]` represents "`n`-tuples of ring elements" (ring maps `Z[x₁,…,xₙ] → R` correspond to tuples `(r₁, …, rₙ) ∈ Rⁿ`); the free vector space `F(S)` represents `V ↦ Set(S, U(V))`. These examples show that universal properties are already statements that a set-valued functor is a hom-functor; what remains unclear is how rigid such a representation is, and whether maps between represented functors must come from maps between representing objects.

**Elements as maps.** In `Set`, an element of a set `X` is the same as a function `∗ → X` from a one-point set, the bijection `Set(∗, X) ≅ X` being `f ↦ f(∗)` and natural in `X`. This reframes "element" as "map out of a distinguished probe object," and suggests the **generalized element**: a map `S → X` of any shape `S` is a "generalized element of `X` of shape `S`." In `Top`, shape-`1` generalized elements are points and shape-`R` ones are curves; in `Grp`, a map `Z → G` is exactly an element of `G` (a homomorphism `φ : Z → G` is determined by `φ(1) ∈ G`, and any value is allowed); in `Field`, a map `K → L` realizes `L` as an extension of `K`. So each object "sees" the rest of the category through a characteristic kind of probe, and the whole web of such probes is precisely the hom-functor.

The diagnostic facts that set up the problem are visible already in tiny cases. For a diagram `F : ω → Set` indexed by the ordinal `ω = (0 → 1 → 2 → ⋯)`, the representable `ω(k, −)` is empty below `k` and a singleton at and above `k`; a natural transformation `α : ω(k, −) ⇒ F` carries no information below `k`, and above `k` the naturality relation `α_{n+1} = f_{n,n+1}(α_n)` shows that the entire family is forced by the single element `α_k ∈ F_k`, and any element of `F_k` may be chosen. For a group `G` viewed as a one-object category `BG`, a natural transformation from the representable into a `G`-set `X` is a `G`-equivariant map `φ : G → X`; equivariance forces `φ(g) = g · φ(e)`, so `φ` is pinned down by `φ(e) ∈ X` alone, and every choice of `φ(e)` gives the equivariant map `g ↦ g · φ(e)`. These are facts about existing diagrams, knowable before any general theorem.

## Baselines

The prior art is the family of techniques for showing two universally-defined objects "the same," and for recognizing one construction inside another, that were in use before any single organizing principle was isolated.

- **Eilenberg–Mac Lane naturality, checked by hand (1945).** Core idea: declare an isomorphism `S(c) ≅ T(c)` *natural* when its components commute with all induced maps, and use that as the criterion for "canonical, no arbitrary choices." The machinery is exactly right but is applied one example at a time — vector space vs. double dual, abelian group vs. double character group — each verification a fresh diagram chase. The gap: there is no statement that says, in advance and uniformly, *what* the natural transformations between two given functors even are, so each comparison is bespoke.

- **Universal arrows / universal properties (the working definition of objects).** Core idea: an object is specified by a universal arrow — e.g. `(r, u : c → Sr)` is universal from `c` to `S` exactly when every `f : c → Sd` factors as `f = Sf' ∘ u` for a unique `f' : r → d`. Equivalently (Mac Lane, Prop. 1) this is a bijection `D(r, d) ≅ C(c, Sd)` natural in `d`. This pins objects down up to isomorphism and is the daily tool, but it leaves the cleaner question unaddressed: it speaks of one functor `C(c, S−)` being a hom-functor, without a general account of *maps between* hom-functors, hence without a statement that the assignment object ↦ hom-functor is itself full and faithful with the appropriate variance.

- **Representable functors / universal-property bookkeeping.** Core idea: study a construction through the functor it is claimed to represent, so that the object is characterized by natural bijections `C(a, b) ≅ X(b)`. This is the right ambition: replace an opaque object by a concrete `Set`-valued functor. The gap it leaves open is the very thing that would license the replacement: *how much* of the object survives in its functor, and does a map between two such functors have to come from an actual map of objects?

- **Generalized elements / probing by maps.** Core idea: recover internal information about an object by mapping standard probe-objects into (or out of) it — points, curves, free generators — generalizing "element = map from a singleton." Powerful and intuitive, but informal: it does not by itself bound how many probes are needed, nor guarantee that two objects giving the same probe-data are isomorphic.

The shared gap across all four: each treats one functor, or one comparison, in isolation, and each comparison stays a bespoke diagram chase. There is no uniform account of the maps out of a hom-functor that would let these techniques be computed rather than re-derived case by case.

## Evaluation settings

The natural yardsticks are not benchmark numbers but the standard categorical situations in which any proposed principle must give the right, known answer:

- **Diagram categories.** `F : ω → Set` (sequences of sets and maps) and finite posets, where natural transformations can be written out by hand and counted.
- **One-object categories.** A monoid `M` or group `G` as a category `BG`; functors `BG → Set` are `M`-sets / `G`-sets, and natural transformations are equivariant maps — a setting where the answer ("equivariant maps `G → X` ↔ elements of `X`") is independently checkable.
- **Concrete representables.** The forgetful functors `Grp → Set` (represented by `Z`), `CRing → Set` (represented by `Z[x]`), `Vect_k → Set`; the Sierpiński space representing open sets; the power-set functor `Set^op → Set` represented by `{⊥, ⊤}`; the matrix category `Mat_R` whose representable `Hom(−, n)` consists of matrices with `n` rows.
- **Classical sanity checks.** Cayley's theorem (every group embeds in a permutation group) and the fact that every row operation on `n`-row matrices is left multiplication by a fixed `n × n` matrix — results that any correct general principle should reproduce as immediate corollaries.

The metric, throughout, is exactness and naturality: the proposed correspondence must be a bijection, and it must be compatible with change of object and change of functor, with no arbitrary choices.

## Code framework

The formal scaffold consists of the categorical primitives needed for a theorem-and-proof calculation:

- a locally small category `C`, with objects, hom-sets `C(a, b)`, identity arrows, and associative composition;
- set-valued functors `F : C → Set`, with object assignments, arrow actions, identity preservation, and composition preservation;
- covariant hom-functors `C(a, −) : C → Set`, with arrows acting by post-composition;
- natural transformations `τ : S ⇒ T`, with components `τ_c : S(c) → T(c)` satisfying `T(u) ∘ τ_c = τ_d ∘ S(u)` for every arrow `u : c → d`;
- objectwise comparisons of such functors, checked only by the naturality equations.

Correctness is tested by symbolic diagram chases: all components must be typed, every naturality square must commute, identities and composition must be preserved, and comparisons between hom-functors must respect the variance of pre-composition and post-composition.
