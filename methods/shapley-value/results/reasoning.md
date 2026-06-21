I need to divide the worth of a grand coalition among the players, but every simple local
answer feels too dependent on the viewpoint I choose. If I look only at what player `i` can do
alone, I miss complementarity. If I look only at what disappears when `i` is removed from the
grand coalition, I privilege the last coalition and can count the same value several times.
If I pick one order of arrival and give each player their marginal increment in that order,
the accounting is beautifully efficient, because the increments telescope from `v(emptyset)`
to `v(N)`, but the order is arbitrary. The game gives me coalition values, not a historical
sequence.

So the object has to be a value map: for every characteristic function `v`, it returns a
payoff vector `phi(v)`. The constraints I trust are structural. The whole worth should be
distributed, so `sum_i phi_i(v) = v(N)`. If two players play the same role in every coalition,
renaming them cannot change their payoffs. If a player adds nothing to any coalition, that
player should get nothing; more generally, if `v(S union {i}) = v(S) + c` for every `S` not
containing `i`, then the player should get `c`. And if I add two independent descriptions of
worth, `v` and `w`, a player's credit in the sum should be the sum of the credits: the
accounting unit should be linear in the game.

Let me try to make the order idea non-arbitrary. For a permutation `pi` of the players, let
`P_i(pi)` be the set of players that appear before `i`. In that arrival order, player `i`
contributes

```text
v(P_i(pi) union {i}) - v(P_i(pi)).
```

For that particular order, summing over all players telescopes exactly to `v(N)`. No value is
lost. The only arbitrary part is choosing the order. If all orders are treated equally, the
arbitrariness disappears:

```text
phi_i(v) = (1 / n!) * sum_pi [ v(P_i(pi) union {i}) - v(P_i(pi)) ].
```

Equivalently, each subset `S subseteq N \ {i}` occurs as the predecessor set of `i` in
`|S|!(n-|S|-1)!` permutations, because the players in `S` can be ordered before `i`, the
players outside `S union {i}` can be ordered after `i`, and `i` is fixed between the two
blocks. Therefore the same formula is

```text
phi_i(v) =
  sum_{S subseteq N \ {i}}
    |S|! (n-|S|-1)! / n! * [ v(S union {i}) - v(S) ].
```

This rule immediately passes the four tests. Efficiency is inherited from each order: for
every permutation, the marginal increments sum to `v(N)-v(emptyset)=v(N)`, and averaging over
orders preserves the sum. Symmetry holds because if two players have identical marginal
effects under every relabeling-relevant coalition, swapping their names simply pairs the
permutations and gives the same average. The dummy condition holds because every marginal
increment for a dummy player is the same constant `c`, so the average is `c`. Additivity holds
because marginal increments are linear:

```text
(v+w)(S union {i}) - (v+w)(S)
= [v(S union {i}) - v(S)] + [w(S union {i}) - w(S)].
```

Averaging preserves that equality.

Existence is clear, but uniqueness is the real pressure test. The axioms should not merely
permit this averaging formula; they should force it. Additivity suggests I should decompose
an arbitrary game into simple basis games where the answer is forced by efficiency, symmetry,
and dummy.

For each nonempty coalition `T`, define a unanimity game `u_T` by

```text
u_T(S) = 1 if T subseteq S,
u_T(S) = 0 otherwise.
```

This game pays one unit exactly when all members of `T` are present. In `u_T`, any player
outside `T` is dummy with zero contribution, because adding that player never changes whether
`T` is contained in the coalition. The players inside `T` are symmetric, because the game
only asks whether all of them are present, not which one is which. Efficiency says the total
payoff is `u_T(N)=1`. Therefore the only possible payoff is

```text
phi_i(u_T) = 1/|T|  if i in T,
phi_i(u_T) = 0      if i notin T.
```

Good. If every game can be written uniquely as a linear combination of these unanimity games,
then additivity will determine the value everywhere, provided I know the value of a scaled
unanimity game. But that is forced without any extra assumption. In `a u_T`, players outside
`T` are still zero dummies, players inside `T` are still symmetric, and efficiency says their
total payoff is `a`. So each member of `T` must get `a/|T|`.

Now I need the decomposition. For every nonempty `T`, I want coefficients `a_T` such that

```text
v(S) = sum_{T subseteq S, T nonempty} a_T
```

for every `S`. This is exactly Möbius inversion on the subset lattice. The coefficients are

```text
a_T = sum_{R subseteq T} (-1)^{|T|-|R|} v(R).
```

So any value satisfying the axioms must be

```text
phi_i(v) = sum_{T containing i} a_T / |T|.
```

That is already a unique expression. The remaining question is whether this equals the
arrival-order average. It should, because in a unanimity game `u_T`, player `i` contributes
one exactly when `i in T` and all the other players of `T` have arrived before `i`; if `i` is
not in `T`, the marginal contribution is always zero. Among the `|T|` players in `T`, all
relative orders are equally likely, and `i` is last among them with probability `1/|T|`.
Thus the permutation formula gives `1/|T|` for `i in T` and zero otherwise. By additivity it
agrees with the forced unanimity-game expression on every linear combination, hence on every
game.

So the fair credit rule is not a recombination of several heuristics. It is the average
marginal contribution over all possible arrival orders, and the axioms make that formula
unavoidable. Efficiency demands that each order's telescoping accounting survive in total.
Symmetry forbids privileging names or one arbitrary order. Dummy removes players whose
marginal increments carry no information. Additivity lets the whole argument reduce to
unanimity games, where symmetry and efficiency leave only one possible division. The final
formula is therefore

```text
phi_i(v) =
  (1 / n!) * sum_pi [ v(P_i(pi) union {i}) - v(P_i(pi)) ]

  =

  sum_{S subseteq N \ {i}}
    |S|! (n-|S|-1)! / n! * [ v(S union {i}) - v(S) ].
```

The expression looks like an average because it is one: draw a uniformly random order in
which the grand coalition forms, watch the extra value player `i` creates at the moment they
arrive, and take the expectation. The theorem says that this expected marginal contribution
is exactly the unique payoff rule compatible with efficiency, symmetry, dummy players, and
additivity.
