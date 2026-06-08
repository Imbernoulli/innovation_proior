Here is the thing that keeps nagging me. I have these objects in a category, and I almost never get to look *inside* them. A product `A × B`, a free group, a tensor product — I never define them by what they're made of. I define them by what maps to and from them do: maps out of `A × B` are pairs of maps, maps out of the free group on `S` are set-functions out of `S`, maps out of `U ⊗ V` are bilinear maps. The object is a black box and the only handle I have on it is the cloud of arrows attached to it. So let me take that seriously and ask the blunt question: if two objects have the same cloud of arrows, are they the same object? Is the object *nothing but* its cloud of arrows?

I should pin down what "the cloud of arrows out of `A`" even is, as a single mathematical gadget. Fix an object `A` in a locally small category `C`. For every other object `B` I have the set `C(A, B)` of arrows from `A` to `B`. And these sets don't sit there inertly: if `g : B → B'`, then post-composing turns an arrow `p : A → B` into `g ∘ p : A → B'`, so I get a function `C(A, B) → C(A, B')`. That is exactly the data of a functor `C → Set`: object `B` goes to the set `C(A, B)`, arrow `g` goes to "compose with `g`." Write it `H^A = C(A, −)`. This functor *is* the cloud — it records every arrow out of `A` and how those arrows move when I push the target around. It's how `A` sees the category `C`. The local-smallness assumption is exactly what makes every `C(A, B)` a set rather than a proper class. Fine.

Now I can sharpen the blunt question. Two questions, really. (i) Comparison: because this is outgoing-arrow data, an arrow `A' → A` ought to give me a comparison of clouds in the reversed direction, a map of functors `H^A → H^{A'}` by precomposition; the variance is already telling me to be careful. I want to know whether *every* compatible comparison of clouds comes from an honest arrow, and whether distinct arrows give distinct comparisons. If yes, then arrows-between-objects and arrows-between-clouds are the same thing with that variance accounted for, and the cloud loses nothing. (ii) But before I can even ask (i), I have to understand maps between these functors at all. And actually I should be greedier than (i): instead of comparing `H^A` only to another hom-functor `H^{A'}`, let me ask what the maps are from `H^A` to a *completely arbitrary* functor `F : C → Set`. Because if I understand `Nat(H^A, F)` for every `F`, then the comparison question (i) is just the special case `F = H^{A'}`.

So: what is a natural transformation `α : H^A ⇒ F`? It is a family of functions, one per object `d`, `α_d : C(A, d) → F(d)`, subject to naturality. Naturality says: for every arrow `g : d → d'`, the square

```
   C(A, d)  --α_d-->   F(d)
   (g∘−) |               | F(g)
   C(A, d') --α_d'-->  F(d')
```

commutes. That is a lot of functions and a lot of squares — one component for every object of a possibly enormous category. It looks hopeless to write down. Let me not stare at the general case; let me get my hands dirty on the smallest possible examples and see if there's a pattern.

Take the category `ω`, the ordinal `0 → 1 → 2 → ⋯`, and a functor `F : ω → Set`, which is just a sequence of sets `F_0, F_1, F_2, …` with maps `f_{n,n+1} : F_n → F_{n+1}` between consecutive ones. Pick a representing object `k`. What is `H^k = ω(k, −)`? There's an arrow `k → n` exactly when `k ≤ n`, and there's at most one, so `ω(k, n)` is empty when `n < k` and a single point when `n ≥ k`. Now a natural transformation `α : ω(k, −) ⇒ F` has components `α_n : ω(k, n) → F_n`. Below `k` the domain is empty, so those components are nothing — no information. At and above `k` the domain is a one-point set, so `α_n` just picks out an element `α_n ∈ F_n`. And naturality of the step `n → n+1` forces `α_{n+1} = f_{n,n+1}(α_n)`. So once I know `α_k`, induction gives me every `α_n` for `n ≥ k`, and the below-`k` part was vacuous. The *entire* natural transformation is pinned down by the single element `α_k ∈ F_k`. And conversely any element of `F_k` I choose for `α_k` propagates to a legal `α`. So `Nat(ω(k, −), F)` is in bijection with `F_k` — natural transformations out of the representable at `k` are just elements of `F` at `k`.

And which element of `ω(k, k)` did `α_k` get evaluated at? `ω(k, k)` is a one-point set, and its one point is the identity `id_k`. So the magic element is `α_k(id_k)`. Hold that thought.

Let me try a totally different shape to make sure that wasn't a fluke of the poset being so thin. Take a group `G`, viewed as a one-object category `BG` — one object `∗`, and the arrows `∗ → ∗` are the group elements, composition is multiplication, the identity arrow is `e`. A functor `BG → Set` is a set `X` with a `G`-action — a `G`-set. The representable functor here is `G` acting on itself by multiplication. A natural transformation from the representable to `X` works out to a `G`-equivariant map `φ : G → X`, equivariance being `φ(g · h) = g · φ(h)`. Now set `h = e`: `φ(g) = φ(g · e) = g · φ(e)`. So `φ` is *entirely* determined by the single value `φ(e) ∈ X`; conversely, if I choose any `x ∈ X` and set `φ_x(g) = g · x`, then `φ_x(g · h) = (g · h) · x = g · (h · x) = g · φ_x(h)`, so the choice really does give an equivariant map. Again: natural transformations out of the representable are in bijection with elements of the target, here `X`. And again the seed value is taken at the identity element `e`, which is the identity arrow of `BG`.

Two completely different shapes, same answer, and the same mechanism: the natural transformation has one degree of freedom, the value it assigns to the identity, and that value lives in `F` evaluated at the representing object. That cannot be an accident. Let me conjecture the general statement and then figure out *why* it has to be the identity that carries all the information.

Conjecture: `Nat(H^A, F) ≅ F(A)`, for any locally small category `C`, any object `A` of `C`, any `F : C → Set`. And the bijection is "evaluate at the identity": `α ↦ α_A(id_A)`.

Why the identity? Let me go back to a general `α : H^A ⇒ F` and just chase what naturality buys me. I want to compute an arbitrary component `α_d` on an arbitrary element. An element of `H^A(d) = C(A, d)` is an arrow `f : A → d`. Here is the one move that's available: `f` itself is an arrow, so I can feed it into the naturality square — take the square for `g = f : A → d`:

```
   C(A, A)  --α_A-->   F(A)
   (f∘−) |               | F(f)
   C(A, d)  --α_d-->   F(d)
```

Now run the identity `id_A ∈ C(A, A)` around both ways. Going down then right: `(f ∘ −)` sends `id_A` to `f ∘ id_A = f`, and then `α_d` sends `f` to `α_d(f)`. Going right then down: `α_A` sends `id_A` to `α_A(id_A)`, and then `F(f)` sends that to `F(f)(α_A(id_A))`. The square commutes, so

```
   α_d(f) = F(f)(α_A(id_A)).
```

There it is. Every component, evaluated at every arrow `f : A → d`, equals `F(f)` applied to the *one* element `α_A(id_A) ∈ F(A)`. The element `x := α_A(id_A)` is the seed, and naturality is the rule that propagates the seed to the whole transformation: the value of `α` at `f` is just `x` "transported along `f`" by the functor `F`. The reason it's the identity is now transparent — every arrow `f : A → d` is `f ∘ id_A`, so the naturality square for `f` literally routes `id_A` to `f` on the bottom, forcing `α_d(f)` to be whatever `α_A(id_A)` becomes under `F(f)`. There's nothing special I had to choose; the identity is the unique element of `C(A, A)` that names `A` itself and sits at the source of every arrow out of `A`.

So the map `α ↦ α_A(id_A)` from `Nat(H^A, F)` to `F(A)` is injective, because I've just shown two natural transformations with the same seed have all the same components. Now I need it surjective, and cleanly I'd like to exhibit the inverse: given an element `x ∈ F(A)`, build a natural transformation. But I don't get to invent the formula — naturality has already told me what it must be. If a transformation has seed `x`, then necessarily `α_d(f) = F(f)(x)`. So *define* `Ψ(x)` by exactly that:

```
   Ψ(x)_d : C(A, d) → F(d),    Ψ(x)_d(f) = F(f)(x).
```

This is forced, not chosen — it's the only candidate that could possibly be natural with the prescribed seed. I have to check three things: that `Ψ(x)` is actually a natural transformation, that evaluating it at the identity gives back `x`, and that this `Ψ` undoes the evaluation map on the nose.

First, is `Ψ(x)` natural? Take any arrow `g : d → e` — and note `d` need not be `A`; I have to check the square for a *generic* arrow, not just arrows out of `A`. The square to check is

```
   C(A, d)  --Ψ(x)_d-->   F(d)
   (g∘−) |                  | F(g)
   C(A, e)  --Ψ(x)_e-->   F(e)
```

Take `f ∈ C(A, d)`. Down then right: `(g ∘ −)` sends `f` to `g ∘ f`, then `Ψ(x)_e` sends that to `F(g ∘ f)(x)`. Right then down: `Ψ(x)_d` sends `f` to `F(f)(x)`, then `F(g)` sends that to `F(g)(F(f)(x))`. These agree precisely because `F` is a functor: `F(g ∘ f) = F(g) ∘ F(f)`. The square commutes for every `g`. So `Ψ(x)` really is a natural transformation — and notice functoriality of `F` is the entire content of the check; that's the whole reason `F` had to be a functor and not just an assignment of sets.

Second, evaluate `Ψ(x)` at the identity: `Ψ(x)_A(id_A) = F(id_A)(x) = id_{F(A)}(x) = x`, using that `F` preserves identities. So evaluation-at-identity composed with `Ψ` is the identity on `F(A)`. That's `Ψ` a right inverse.

Third, the other composite: start with an arbitrary `α`, take its seed `x = α_A(id_A)`, build `Ψ(x)`, and check `Ψ(x) = α`. Two natural transformations are equal iff all components agree, and two functions agree iff they agree on every element, so I must show `Ψ(x)_d(f) = α_d(f)` for every object `d` and every `f : A → d`. But `Ψ(x)_d(f) = F(f)(x) = F(f)(α_A(id_A))`, and I proved up above — that was the very naturality square that started all this — that `F(f)(α_A(id_A)) = α_d(f)`. They match. So `Ψ(α_A(id_A)) = α`. That's `Ψ` a left inverse.

A two-sided inverse on both sides means evaluation-at-the-identity, `ev : Nat(H^A, F) → F(A)`, `α ↦ α_A(id_A)`, is a bijection, with explicit inverse `x ↦ (f ↦ F(f)(x))`. The conjecture is a theorem. And as a byproduct it tells me the collection `Nat(H^A, F)` is a genuine *set* — a priori it might have been a proper class, but it's in bijection with the set `F(A)`. The natural transformations out of a representable functor are just the elements of the target at the representing object. This is the Yoneda lemma. The bijection alone still has to answer the question I started with: whether the object is *determined* by its cloud, and that needs the comparison case and a naturality statement.

But first I should make sure the bijection isn't just a per-`A`, per-`F` accident — that it's compatible with varying `A` and varying `F`, i.e. natural in both. Otherwise I haven't really shown the cloud carries the structure, only that two sets happen to have the same size for each fixed input.

Naturality in `F` first. Suppose `β : F ⇒ G` is a natural transformation of target functors. There are two ways to get from `Nat(H^A, F)` to `G(A)`: either evaluate in `F` and then push the element forward by `β_A`, or push the whole transformation forward by `β` first (vertical composite `β · α`, with components `(β · α)_c = β_c ∘ α_c`) and then evaluate in `G`. I need these to agree. Take `α : H^A ⇒ F`. Going one way: `ev^F(α) = α_A(id_A)`, then `β_A(α_A(id_A))`. Going the other: `β · α`, then `ev^G(β · α) = (β · α)_A(id_A) = β_A(α_A(id_A))`. They are literally the same expression by the definition of vertical composition. So the bijection is natural in `F`.

Naturality in `A` (well, in the object — this is contravariant, since `H^{(−)}` is). Suppose `f : A → A'`. Precomposition `f^* = (− ∘ f)` turns `H^{A'} = C(A', −)` into `H^A = C(A, −)`, hence sends a transformation `α : H^A ⇒ F` to `α · f^* : H^{A'} ⇒ F`. I want: evaluate-then-`F(f)` equals `f^*`-then-evaluate. One way: `ev^{A}(α) = α_A(id_A)`, then `F(f)(α_A(id_A))`. Other way: `(α · f^*)_{A'}(id_{A'})`. Let me compute that. The `A'`-component of `α · f^*` is `α_{A'} ∘ (f^*)_{A'}`, and `(f^*)_{A'}(id_{A'}) = id_{A'} ∘ f = f`, so `(α · f^*)_{A'}(id_{A'}) = α_{A'}(f) = F(f)(α_A(id_A))` — the last step being exactly the naturality square again. Both routes land on `F(f)(α_A(id_A))`. Natural in the object too. So the bijection `Nat(H^A, F) ≅ F(A)` is a single natural isomorphism, compatible with all changes of `A` and `F` at once — evaluation-at-the-identity is itself a natural transformation between two functors of the pair `(A, F)`.

Now the comparison case — the whole reason I started. Specialize the arbitrary `F` to a hom-functor: `F = H^{A'} = C(A', −)`. Then `F(A) = C(A', A)`, and the lemma reads

```
   Nat(H^A, H^{A'}) ≅ C(A', A).
```

Read that carefully. A natural transformation between the two clouds `H^A → H^{A'}` corresponds to a single arrow `A' → A`, uniquely. So the comparisons of clouds are *exactly* the arrows between the objects — there's no extra, exotic comparison that isn't induced by an honest arrow, and no two distinct arrows induce the same comparison. The direction is the one variance forces. An arrow `h : A' → A` induces a natural transformation `H^h : H^A ⇒ H^{A'}` by precomposition, `(H^h)_d : C(A, d) → C(A', d)`, `p ↦ p ∘ h`. I still owe the naturality square for this induced comparison. If `g : d → e` and `p : A → d`, then going across and then down gives `g ∘ (p ∘ h)`, while going down and then across gives `(g ∘ p) ∘ h`; associativity makes these the same arrow `A' → e`. So `H^h` really is natural. Evaluating `H^h` at the identity in the right place gives `(H^h)_A(id_A) = id_A ∘ h = h`. So evaluation-at-identity sends `H^h` back to `h` — `h ↦ H^h` is right inverse to the Yoneda bijection, and since that bijection is already a bijection, `h ↦ H^h` *is* the inverse. Therefore the assignment

```
   C(A', A) → Nat(H^A, H^{A'}),    h ↦ H^h
```

is a bijection for all `A, A'`. That is precisely the statement that the functor `A ↦ H^A` is **full** (every natural transformation between clouds comes from an arrow) and **faithful** (distinct arrows give distinct natural transformations). I should be careful about variance: `H^h` for `h : A' → A` goes `H^A → H^{A'}`, so the assignment `A ↦ H^A = C(A, −)` reverses arrows — it's a full and faithful functor `C^op → Set^C`, the Yoneda embedding. (Equivalently, taking maps *into* objects, `A ↦ C(−, A)` is a full and faithful functor `C → Set^{C^op}` into presheaves, embedding `C` into its presheaf category — the same content for the dual question "what maps *into* `A`?")

Full-and-faithfulness is exactly what I needed to answer the original blunt question. A full and faithful functor reflects isomorphisms and is injective on objects up to isomorphism: if the clouds `H^A` and `H^{A'}` are isomorphic as functors, an isomorphism `φ : H^A ⇒ H^{A'}` corresponds to a unique arrow `h : A' → A`, and its inverse corresponds to a unique arrow `k : A → A'`. The composite `H^k ∘ H^h : H^A ⇒ H^A` is the identity, so the corresponding arrow `h ∘ k : A → A` must be `id_A`; similarly `H^h ∘ H^k` gives `k ∘ h = id_{A'}`. Thus `A ≅ A'`. More precisely, each specified isomorphism of clouds has a unique underlying isomorphism of objects in the reversed direction; without specifying the cloud-isomorphism I should say "determined up to isomorphism," not pretend there is only one possible object-isomorphism. An object is its functor of points in that precise sense: I may replace any object by the set-valued functor `C(A, −)` (or `C(−, A)`), and I lose neither the object up to isomorphism nor the arrows between objects.

And this immediately turns the case-by-case craft I started from into a calculation. To know all natural transformations between two representable functors I no longer chase diagrams — I read off `C(A', A)`. If two universally-defined objects are given with the same universal property, the chosen identification of their representing functors determines the unique compatible object-isomorphism; the objects are determined up to isomorphism, and the compatibility data removes the remaining ambiguity. A representation of a functor `X` is the same data as a *universal element* `u ∈ X(A)` — because under the lemma the natural transformation `H^A ⇒ X` corresponding to `u` is `f ↦ X(f)(u)`, and demanding it be an isomorphism is exactly demanding that every `x ∈ X(B)` factor as `X(x̄)(u)` for a unique `x̄ : A → B`. And small classical facts fall out: every natural endomorphism of `Mat_R(−, n)` is left multiplication by a fixed `n × n` matrix (and the lemma even names it — apply the operation to the identity matrix, since the seed is the image of `id`); the regular action recovers a group inside permutations of its underlying set, which is Cayley's theorem. All of these are the single seed-at-the-identity mechanism, specialized.

This is the Yoneda lemma. Let `C` be a locally small category, let `A` be an object of `C`, and let `F : C → Set` be a functor. Then evaluation at the identity,

```
   ev : Nat(C(A, −), F) → F(A),    ev(α) = α_A(id_A),
```

is a bijection, with inverse `Ψ(x)_d(f) = F(f)(x)` for `x ∈ F(A)` and `f ∈ C(A, d)`. The bijection is natural in `F`, and it is natural in the object variable with the contravariance coming from `A ↦ C(A, −)`.

Take `α : C(A, −) ⇒ F`. If `f : A → d`, the naturality square of `α` for the arrow `f`, evaluated at `id_A ∈ C(A, A)`, gives

```
   α_d(f) = F(f)(α_A(id_A)).        (∗)
```

Given `x ∈ F(A)`, define `Ψ(x)_d(f) = F(f)(x)`. This family is natural: for `g : d → e` and `f : A → d`, one route gives `Ψ(x)_e(g∘f) = F(g∘f)(x)`, while the other gives `F(g)(Ψ(x)_d(f)) = F(g)(F(f)(x))`, and these are equal because `F(g∘f) = F(g)∘F(f)`. Evaluation sends this constructed transformation back to the seed, since `ev(Ψ(x)) = Ψ(x)_A(id_A) = F(id_A)(x) = x`. In the other direction, starting from `α`, the transformation constructed from its seed satisfies `Ψ(α_A(id_A))_d(f) = F(f)(α_A(id_A)) = α_d(f)` by `(∗)` for every `d` and every `f : A → d`. So `Ψ` and `ev` are inverse functions.

The compatibility in `F` is just as strict. If `β : F ⇒ G`, then `ev^G(β·α) = (β·α)_A(id_A) = β_A(α_A(id_A)) = β_A(ev^F(α))`. For the object variable, if `f : A → A'`, precomposition `f^* = (−∘f) : C(A', −) ⇒ C(A, −)` sends `α : C(A, −) ⇒ F` to `α·f^* : C(A', −) ⇒ F`, and evaluating at `A'` gives `(α·f^*)_{A'}(id_{A'}) = α_{A'}(id_{A'}∘f) = α_{A'}(f) = F(f)(α_A(id_A))`, exactly the same element obtained by evaluating `α` first and then applying `F(f)`.

Specializing to `F = C(A', −)` gives

```
   Nat(C(A, −), C(A', −)) ≅ C(A', A).
```

An arrow `h : A' → A` induces `H^h : H^A ⇒ H^{A'}` by precomposition, `(H^h)_d(p) = p ∘ h`. This is natural because for any `g : d → e` and `p : A → d`, the two routes through the square give `g ∘ (p ∘ h)` and `(g ∘ p) ∘ h`, equal by associativity. Evaluation recovers `h` by `(H^h)_A(id_A) = id_A ∘ h = h`. Therefore `h ↦ H^h` is the inverse to the evaluation bijection, so `A ↦ C(A, −)` is a full and faithful functor `C^op → Set^C`. Dually, `A ↦ C(−, A)` is a full and faithful functor `C → Set^{C^op}`. If a functor is represented by both `A` and `A'`, the induced isomorphism between the corresponding hom-functors gives a unique object-isomorphism compatible with the two representations; equivalently, each isomorphism `H^A ≅ H^{A'}` comes from one and only one isomorphism between `A` and `A'` in the reversed direction.

So the chain is: an object is opaque, accessible only through its arrows; package all arrows out of `A` into one functor `H^A`; a natural transformation out of `H^A` is, by the naturality square applied to each arrow `f = f ∘ id_A`, forced to be `f ↦ F(f)(seed)` where the seed is the image of `id_A`, and any seed works, so `Nat(H^A, F) ≅ F(A)` by evaluation at the identity; specializing the target to another hom-functor turns this into `Nat(H^A, H^{A'}) ≅ C(A', A)`, making `A ↦ H^A` full and faithful with the contravariant direction; and full-and-faithfulness means the cloud of arrows determines the object up to isomorphism, with each chosen isomorphism of clouds coming from a unique object-isomorphism. An object *is* its functor of points.
