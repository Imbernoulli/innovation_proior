## Research question

Fourier analysis begins with a concrete surprise: functions on a group can often be studied by testing them against simple homomorphisms into the unit circle. On finite cyclic groups this gives roots of unity; on `R` it gives exponentials `x |-> exp(2 pi i xi x)`; on the circle it gives integer modes `z |-> z^n`. These examples look like separate formulas, but the recurring object is the same: a continuous character

```
chi : G -> T,
```

where `T={z in C : |z|=1}`.

The problem is to understand whether these characters are merely useful coordinates or whether they contain the whole locally compact abelian group. A satisfactory theorem should start from a locally compact abelian group `G`, form the group of continuous circle-valued characters, put the right topology on that character group, and recover `G` canonically from the characters of its character group. If this works, Fourier analysis is not just a transform formula; it is the expression of a structural duality between groups.

## Background

Locally compact abelian groups are the natural setting where topology, Haar measure, and group structure coexist. Local compactness supplies Haar measure and makes compact subsets meaningful; commutativity makes the character group abelian under pointwise multiplication; continuity keeps the characters aligned with the topology of `G`.

For a locally compact abelian group `G`, the dual group is

```
G^ = Hom_cont(G, T),
```

with pointwise multiplication. The topology on `G^` is the compact-open topology, equivalently uniform convergence on compact subsets of `G`. This topology is not decorative. If the dual were given only the algebraic group structure, it would forget how characters vary; if it were given the wrong topology, the second dual would not carry the right comparison topology back to `G`.

The canonical comparison map is evaluation:

```
e_G : G -> G^^,
e_G(x)(chi) = chi(x).
```

Each `x in G` defines a character on `G^`; the question is whether every character on `G^` arises this way and whether the topology is exactly recovered. The expected answer must also explain the basic examples: `R` is dual to `R`, `Z` is dual to `T`, finite abelian groups are dual to finite abelian groups, and compact groups trade places with discrete groups.

## Baselines

- **Fourier series on compact abelian examples.** On `T`, the characters are `z |-> z^n`, indexed by `Z`. They diagonalize translation and give Fourier coefficients. This is powerful, but by itself it can look like a special orthogonal expansion tied to the circle rather than a theorem about all locally compact abelian groups.

- **Fourier transform on Euclidean space.** On `R^n`, the characters are `x |-> exp(2 pi i <xi,x>)`, again indexed by `R^n`. The transform converts translation into multiplication by characters. The limitation is conceptual: the formula hides the fact that the frequency variable `xi` is a point of the dual group, not merely a parameter inserted into an integral.

- **Finite abelian character tables.** A finite abelian group has enough characters to separate points, and its dual has the same cardinality. This gives a clean algebraic model of recovery from characters. The limitation is that finite groups have no real topological difficulty; compactness, discreteness, and continuity collapse into simpler notions.

- **Algebraic homomorphisms into fields or into `C*`.** Algebraic characters can detect group structure in favorable settings, but they do not encode the local compact topology needed for Haar measure and Fourier analysis. Continuous circle-valued characters are the stable analytic object.

## Evaluation settings

The theorem should be checked against the examples where the answer is visible without abstraction: finite abelian groups, `Z`, `T`, `R`, `R^n`, products, quotients, compact groups, and discrete groups. The compact/discrete reversal is especially important: if `G` is compact, `G^` should be discrete; if `G` is discrete, `G^` should be compact.

The success criterion is exact recovery, not approximation. The evaluation map `e_G` must be a continuous open group isomorphism from `G` onto `G^^`. The Fourier transform should then be understood as sending functions or measures on `G` to functions on `G^`, with characters acting as the bridge between the two groups.

## Proof artifact

The final artifact should state the dual group construction, the compact-open topology, and the evaluation map:

```
G^ = Hom_cont(G,T),
e_G(x)(chi)=chi(x).
```

It should prove the Pontryagin duality theorem for locally compact abelian groups:

```
e_G : G -> G^^
```

is a topological group isomorphism.

The proof should make clear why the circle group is the target of characters, why compact-open topology is the correct topology on the dual, how characters separate points strongly enough to make `e_G` injective, why every character of `G^` is evaluation at some point of `G`, and how compact subgroups and discrete quotients explain the reversal between compact and discrete groups. It should close by tying this structural duality to Fourier transform: the transform is analysis performed across the paired groups `G` and `G^`.
