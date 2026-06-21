I want a notion of volume that belongs to the symmetry itself. Counting measure works when the group is discrete. Lebesgue measure works when the group is `R^n`. Compact Lie groups suggest invariant volume. But those are not enough; they are examples with extra structure. I cannot just splice counting measure, Euclidean coordinates, and manifold volume together. A general locally compact group may have no coordinates, no smooth form, and no finite total size. The group operation and the topology are all I should be allowed to use.

The invariance requirement is clear before the construction is clear. If `E` is measurable, moving it by a group element should not change its size:

`mu(xE) = mu(E)`.

If this is going to support integration, it cannot be only a finitely additive size on convenient sets. It has to be a regular Borel measure, finite on compact sets, because compact sets are the only bounded pieces that local compactness reliably gives me. Local compactness is doing the first real piece of work: near the identity there are neighborhoods whose closures are compact, and compact subsets can be covered by finitely many translates of such neighborhoods. Without that finite-covering fact, I do not see how to control local mass.

The first tempting move is to define the size of a compact set by the number of translates of a small neighborhood needed to cover it. For compact `K` and an open neighborhood `U`, let `(K:U)` be the smallest number of left translates of `U` that cover `K`. This is beautifully group-theoretic: translating `K` does not change `(K:U)`. It is also too crude. Covering numbers are subadditive, not additive; two overlapping compact sets can be counted twice. If I stop here, I have a coarse geometry, not an integral.

Maybe functions are smoother than sets. A compactly supported continuous function can be dominated by translates of another positive compactly supported function. If `f,g in C_c(G)^+` and `g` is not zero, define a functional-looking covering gauge

`[f:g] = inf sum_i c_i`

over all finite inequalities

`f <= sum_i c_i L_{x_i} g`, with `c_i > 0`.

This keeps the finite-covering idea but gives coefficients room to behave like mass. Finiteness comes from compactness of `supp(f)`: where `g` is positive on some neighborhood, finitely many translates of that neighborhood cover `supp(f)`, and a sufficiently large coefficient dominates `f` on each translated patch. Left invariance is already baked in, because translating `f` just translates the covering family:

`[L_y f:g] = [f:g]`.

Still, `[f:g]` depends on the measuring bump `g`. I need to let the measuring bump become local while normalizing the scale. Fix one nonzero `f_0 in C_c(G)^+`. For each sufficiently small positive bump `g`, look at

`I_g(f) = [f:g] / [f_0:g]`.

The denominator chooses the unit of volume. These `I_g` are positive, homogeneous, and left invariant. They are not yet linear in an obvious way. The missing additivity is where local compactness and continuity meet: if the support of `g` is forced into a small enough neighborhood, then on that neighborhood continuous functions barely vary under translation. A cover efficient for `f_1 + f_2` cannot save much beyond separately covering `f_1` and `f_2`, because locally the two functions look almost constant at the scale of `g`. Subadditivity is immediate; the reverse inequality appears only in the limiting process as the bump shrinks.

So the object I should extract is not a single `I_g`, but a cluster point of the normalized gauges as the support of `g` shrinks toward the identity. For each fixed `f`, the normalized values are bounded by covering comparisons with `f_0`, and the product compactness argument gives a subnet converging pointwise on `C_c(G)^+`. The limiting object `I` is positive, homogeneous, additive on nonnegative functions, and extends by differences to a positive linear functional on all of `C_c(G)`. Since every `I_g` is left invariant, the limit is left invariant:

`I(L_x f) = I(f)`.

It is nonzero because `I(f_0)=1`. It is positive because all covering gauges are positive. This is the point where the rough covering idea has turned into analysis: abstract symmetry has produced an invariant integral on compactly supported continuous functions.

Now I can stop trying to assign mass directly to arbitrary sets. The Riesz representation theorem does that cleanly. A nonzero positive linear functional on `C_c(G)` corresponds to a regular Borel measure `mu`, finite on compact sets, with

`I(f) = int_G f dmu`

for every `f in C_c(G)`. The identity `I(L_x f)=I(f)` transfers to the measure by testing against compactly supported continuous functions and using regularity:

`mu(xE)=mu(E)`

for Borel sets `E`. Nonzero positivity of the functional, together with regularity and local compactness, gives positive mass to nonempty open sets: choose a nonzero nonnegative continuous function supported inside the open set, and its integral is positive after the construction is normalized on some positive test function.

The existence part is now conceptually settled. Every locally compact group carries its own left-invariant volume. The result is not a recombination of familiar measures; the finite-covering structure of compact sets and the functional representation theorem force the measure out of the group topology.

I still need to understand whether this volume is canonical or just one choice among many. Suppose `mu` and `nu` are two nonzero regular left-invariant Borel measures. I can compare them on test functions instead of sets. The same compact-covering estimates that built the functional also bound any positive left-invariant functional by any other once a normalization is fixed. If `I_mu(f)=int f dmu` and `I_nu(f)=int f dnu`, then after choosing `f_0 >= 0` with both integrals positive, the ratios

`I_mu(f) / I_mu(f_0)` and `I_nu(f) / I_nu(f_0)`

must agree for every `f in C_c(G)^+`. The reason is that both functionals obey the same translation-invariant local covering inequalities, and in the shrinking-bump comparison the only free parameter is the chosen unit `f_0`. Extending by linearity gives

`I_nu(f) = c I_mu(f)` for all `f in C_c(G)`,

where `c = I_nu(f_0)/I_mu(f_0) > 0`. Regularity then upgrades equality on test functions to equality of measures:

`nu = c mu`.

That is the precise sense in which the volume is canonical. There is no distinguished total mass unless the group is compact and I normalize `mu(G)=1`, or unless I choose some other unit. But once the unit is chosen, all left-invariant integration is fixed.

One caveat remains, and it is easy to miss if I think only about abelian examples. A left-invariant measure need not be right invariant. For a fixed `a in G`, define a new measure by right translation, `mu_a(E)=mu(Ea)`. Left translation still preserves it:

`mu_a(xE)=mu(xEa)=mu(Ea)=mu_a(E)`.

So `mu_a` is another left-invariant regular Borel measure. Uniqueness forces it to be a scalar multiple of `mu`. Thus there is a positive scalar depending on `a`; with the common convention,

`mu(Ea) = Delta(a)^{-1} mu(E)`.

The map `Delta:G -> (0,infty)` is continuous and multiplicative, because two right translations compose and the corresponding scalars multiply. This is the modular function. If `Delta` is identically `1`, the group is unimodular and the same measure is both left and right invariant. Compact groups, discrete groups, abelian groups, and many familiar Lie groups fall into that calmer case, but the theorem itself is more general. The canonical object is left-invariant volume; right invariance is an additional property, not part of the basic guarantee.

So the final shape is: local compactness gives finite compact coverings; compact coverings are refined into normalized positive functionals on `C_c(G)`; a limiting argument gives a nonzero left-invariant positive linear functional; Riesz turns it into a regular Borel measure; uniqueness leaves only a scalar; right translation reveals the modular function. Abstract symmetry is now measurable and integrable.
