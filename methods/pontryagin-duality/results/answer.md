# Pontryagin Duality

## Theorem

Let `G` be a locally compact abelian group. Its Pontryagin dual is

```
G^ = Hom_cont(G,T),
```

the group of continuous characters from `G` to the circle group `T={z in C : |z|=1}`, with pointwise multiplication and the compact-open topology.

The evaluation map

```
e_G : G -> G^^,
e_G(x)(chi) = chi(x)
```

is an isomorphism of topological groups.

## Immediate Consequences

If `G` is compact, then `G^` is discrete. If `G` is discrete, then `G^` is compact.

The standard examples fit the theorem:

```
Z^ ~= T
T^ ~= Z
R^ ~= R
(finite abelian group)^ ~= finite abelian group
```

The isomorphism `R^ ~= R` depends on the normalization convention, usually

```
xi |-> (x |-> exp(2 pi i x xi)).
```

## Proof Artifact

For `x in G`, define `e_G(x)` on `G^` by evaluation:

```
e_G(x)(chi)=chi(x).
```

This is a character of `G^`, since

```
e_G(x)(chi psi) = (chi psi)(x) = chi(x)psi(x).
```

It is continuous because evaluation at a fixed point is continuous for the compact-open topology: the singleton `{x}` is compact. Thus `e_G` maps `G` into `G^^`. It is a homomorphism because

```
e_G(x+y)(chi)=chi(x+y)=chi(x)chi(y)=e_G(x)(chi)e_G(y)(chi).
```

Characters separate points in locally compact abelian groups: if `x != 0`, there exists a continuous character `chi:G->T` with `chi(x) != 1`. Therefore `e_G(x)` is not the trivial character, so `e_G` is injective.

Surjectivity is the central content. Let `Phi in G^^`, so `Phi:G^->T` is a continuous character. The Pontryagin duality theorem asserts that such a `Phi` is not an additional kind of frequency object: there is an `x in G` such that

```
Phi(chi)=chi(x)
```

for every `chi in G^`. In other words, every continuous character of the character group is evaluation at a point of the original group.

The topology is recovered at the same time. The compact-open topology on `G^` makes compact families of characters the tests for neighborhoods in `G^^`. Under `e_G`, these tests are exactly uniform control of `chi(x)` over compact subsets of `G^`, which recovers the original locally compact group topology on `G`. Hence `e_G` is continuous, open, and has continuous inverse.

Thus `G ~= G^^` naturally.

## Fourier Interpretation

For Haar-integrable `f` on `G`, the Fourier transform is indexed by the dual group:

```
f^(chi) = int_G f(x) conjugate(chi(x)) dx,    chi in G^.
```

This formula is not merely a change of coordinates. The frequency side is itself a locally compact abelian group, and dualizing it returns the original group. Fourier analysis is therefore a duality of groups expressed analytically through integration against characters.
