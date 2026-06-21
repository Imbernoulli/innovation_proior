# Shapley Value

## Theorem

Let `N` be a finite set of `n` players and let `v: 2^N -> R` be a transferable-utility
cooperative game with `v(emptyset)=0`. There is a unique payoff rule `phi` satisfying:

1. **Efficiency:** `sum_{i in N} phi_i(v) = v(N)`.
2. **Symmetry:** if players `i` and `j` have identical marginal effects in every coalition,
   then `phi_i(v) = phi_j(v)`.
3. **Dummy:** if `v(S union {i}) = v(S) + c` for all `S subseteq N \ {i}`, then
   `phi_i(v) = c`.
4. **Additivity:** `phi(v+w) = phi(v) + phi(w)`.

The unique rule is the expected marginal contribution of each player over a uniformly random
arrival order:

```text
phi_i(v) = (1 / n!) * sum_pi [ v(P_i(pi) union {i}) - v(P_i(pi)) ],
```

where the sum is over all permutations `pi` of `N`, and `P_i(pi)` is the set of players that
appear before `i` in `pi`. Equivalently,

```text
phi_i(v) =
  sum_{S subseteq N \ {i}}
    |S|! (n-|S|-1)! / n! * [ v(S union {i}) - v(S) ].
```

## Proof

First prove that the displayed rule satisfies the axioms.

For any fixed permutation, the players' marginal contributions telescope:

```text
sum_i [ v(P_i(pi) union {i}) - v(P_i(pi)) ] = v(N) - v(emptyset) = v(N).
```

Averaging over all permutations gives efficiency.

If two players have identical roles, swapping their labels pairs permutations without
changing the relevant marginal contribution multiset, so their averaged values are equal.
This gives symmetry.

If player `i` is dummy with constant contribution `c`, then every marginal contribution
`v(S union {i}) - v(S)` equals `c`. The average is therefore `c`, giving dummy.

For games `v` and `w`,

```text
(v+w)(S union {i}) - (v+w)(S)
= [v(S union {i}) - v(S)] + [w(S union {i}) - w(S)].
```

Taking the same weighted sum over coalitions gives
`phi_i(v+w)=phi_i(v)+phi_i(w)`, so additivity holds.

Now prove uniqueness. For every nonempty `T subseteq N`, define the unanimity game

```text
u_T(S) = 1 if T subseteq S,
u_T(S) = 0 otherwise.
```

In `u_T`, every player outside `T` is a dummy with zero contribution. The players in `T` are
symmetric. Efficiency requires the players in `T` to split `u_T(N)=1`, so every admissible
rule must satisfy

```text
phi_i(u_T) = 1/|T|  if i in T,
phi_i(u_T) = 0      if i notin T.
```

Every game has a unique expansion in unanimity games:

```text
v = sum_{T nonempty} a_T u_T,
```

where Möbius inversion gives

```text
a_T = sum_{R subseteq T} (-1)^{|T|-|R|} v(R).
```

Indeed, for every coalition `S`,

```text
sum_{T subseteq S, T nonempty} a_T = v(S).
```

By additivity, any admissible rule is therefore forced to be

```text
phi_i(v) = sum_{T containing i} a_T / |T|.
```

The scalar coefficient causes no extra freedom: in the scaled unanimity game `a_T u_T`,
players outside `T` are zero dummies, players inside `T` are symmetric, and efficiency forces
each member of `T` to receive `a_T/|T|`.

Thus at most one rule can satisfy the four axioms.

It remains only to match this forced rule to the permutation formula. In `u_T`, player `i`
has marginal contribution `1` exactly when `i in T` and all other members of `T` have already
arrived before `i`; otherwise the marginal contribution is `0`. If `i in T`, the probability
that `i` is last among the `|T|` members of `T` in a uniformly random permutation is
`1/|T|`. If `i notin T`, the probability of a positive marginal contribution is zero.
Therefore the permutation formula assigns exactly the forced value on each unanimity game.
By additivity, it assigns the forced value on every game.

So the four axioms do not merely motivate the formula; they uniquely determine it. The
Shapley value is fair credit as expected marginal contribution over all arrival orders.
