# Downward Löwenheim-Skolem Theorem

Let `L` be a first-order language, let `M` be an `L`-structure, and let `A subseteq M`. There is an elementary substructure `N prec M` such that

`A subseteq N`

and

`|N| <= max(|A|, |L|, aleph_0)`.

In particular, if `L` is countable, then every infinite `L`-structure has a countable elementary substructure containing any prescribed countable subset.

## Proof

Set

`kappa = max(|A|, |L|, aleph_0)`.

For each first-order formula `phi(y,x_1,...,x_n)`, choose a witness function

`F_phi:M^n -> M`

with the following property: if `M |= exists y phi(y,a_1,...,a_n)`, then

`M |= phi(F_phi(a_1,...,a_n),a_1,...,a_n)`.

If no witness exists for a tuple, choose any element of `M`; the choice will not be used for an existential truth. If `M` is empty, the statement is trivial after taking `A` empty and `N` empty, so assume `M` is nonempty.

Build sets `A_i subseteq M` for `i < omega`. Let `A_0` contain `A` and the interpretations of all constant symbols of `L`. Given `A_i`, let `A_{i+1}` be `A_i` together with:

1. all values `f^M(a_1,...,a_n)` for every function symbol `f` of `L` and every tuple from `A_i`;
2. all values `F_phi(a_1,...,a_n)` for every formula `phi(y,x_1,...,x_n)` and every tuple from `A_i`.

Let

`N = union_{i<omega} A_i`.

There are at most `max(|L|, aleph_0)` formulas of `L`, and for every infinite `kappa`, the set of finite tuples from a set of size at most `kappa` also has size at most `kappa`. Hence each `A_i` has size at most `kappa`, and the countable union `N` has size at most `kappa`.

The set `N` is closed under the function symbols of `L`: if a finite tuple from `N` is given, all its entries occur together in some `A_i`, so the value of any function symbol on that tuple lies in `A_{i+1}`. Constants are already in `A_0`. Therefore `N` is the domain of an `L`-substructure of `M`.

Apply the Tarski-Vaught test. Suppose `a_1,...,a_n in N` and

`M |= exists y phi(y,a_1,...,a_n)`.

The tuple `a` is finite, so it is contained in some `A_i`. By construction,

`F_phi(a_1,...,a_n) in A_{i+1} subseteq N`,

and by the defining property of `F_phi`,

`M |= phi(F_phi(a_1,...,a_n),a_1,...,a_n)`.

Thus every existential statement true in `M` with parameters from `N` has a witness already in `N`. By Tarski-Vaught, `N prec M`.

The theorem explains Skolem's paradox. A countable first-order theory with an infinite model can have a countable model, even when it proves internally that uncountable sets exist. The smaller model lacks an internal bijection witnessing countability for those sets, so first-order truth is preserved while external cardinality is not pinned down.
