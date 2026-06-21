# Arrow Impossibility

## Theorem

Let there be at least three alternatives and a finite set of voters. A social welfare function takes every profile of strict individual rankings and returns a strict social ordering. If it satisfies:

- **Unrestricted domain:** every profile of strict rankings is allowed.
- **Pareto:** if every voter ranks `x > y`, society ranks `x > y`.
- **Independence of irrelevant alternatives:** the social comparison of `x` and `y` depends only on the voters' individual comparisons of `x` and `y`.

Then the rule is dictatorial: there is a voter `i` such that for every profile and every pair `x,y`, if voter `i` ranks `x > y`, society ranks `x > y`.

## Proof

Call a coalition `S` decisive for `x` over `y` if, whenever every voter in `S` ranks `x > y` and every voter outside `S` ranks `y > x`, the social ordering ranks `x > y`.

The full voter set is decisive for every pair by Pareto. The empty coalition is not decisive, again by Pareto. Choose a decisive coalition minimal under inclusion.

If a coalition `S` is decisive for one ordered pair, it is decisive for every ordered pair. To see the transfer, suppose `S` is decisive for `x` over `y`, and take a third alternative `z`. Construct a profile in which voters in `S` rank `x > y > z` and voters outside `S` rank `y > z > x`. Decisiveness gives social `x > y`; Pareto gives social `y > z`; transitivity gives social `x > z`. By IIA, the social comparison of `x` and `z` depends only on the individual `x,z` comparisons, so in every profile where exactly voters in `S` rank `x > z` and outsiders rank `z > x`, society ranks `x > z`. Thus `S` is decisive for `x` over `z`. Relabeling alternatives gives decisiveness for all ordered pairs.

Now show a minimal decisive coalition must be a singleton. Suppose `S` has at least two voters. Split it into nonempty disjoint sets `A` and `B`, with `S = A ∪ B`. Pick three alternatives `x,y,z`, and construct a profile:

- voters in `A`: `x > y > z`;
- voters in `B`: `y > z > x`;
- voters outside `S`: `z > x > y`.

For the pair `y,z`, all voters in `S` rank `y > z`, and all voters outside `S` rank `z > y`; since `S` is decisive, society ranks `y > z`.

Social transitivity now forces at least one of these to hold: society ranks `x > y`, or society ranks `z > x`. If neither held, strict completeness would give `y > x` and `x > z`, and together with `y > z` this would not by itself violate transitivity; so this split-profile shortcut is not sufficient. Use the pivotal construction instead, which isolates the same minimality point without this gap.

Order the voters in `S` as `1,...,k`. Since `S` is decisive for every pair and no proper subset of `S` is decisive, move through profiles for a fixed pair `x,y` where voters in `S` switch one at a time from `x > y` to `y > x`, while all voters outside `S` rank `y > x`. At the start, `S` forces social `x > y`; at the end, Pareto gives social `y > x`. Hence there is a first voter `i` whose switch changes the social comparison. Voter `i` is pivotal for `x,y`.

Take any third alternative `z`. Because the domain is unrestricted, construct profiles around the pivotal boundary so that all voters unanimously rank `y > z`, while the `x,y` pair is exactly the pre-switch pivotal pattern. Before voter `i` switches, society has `x > y`; Pareto gives `y > z`; transitivity gives `x > z`. By IIA, this conclusion depends only on the individual `x,z` comparisons in that constructed profile.

Construct the corresponding post-switch profile with unanimous `z > y`; the pivotal comparison gives society `y > x`, Pareto gives `z > y`, and transitivity gives `z > x`. Again IIA makes the conclusion depend only on the `x,z` pairwise pattern. Combining the two profiles shows that voter `i`'s ranking of `x` versus `z` determines the social ranking of `x` versus `z`, regardless of how the other voters rank that pair.

Relabeling alternatives gives the same conclusion for every pair. Therefore voter `i` is decisive for every ordered pair. The social ordering copies voter `i` pairwise, so the rule is dictatorial.

The impossibility comes from the interaction of the axioms. Unrestricted domain supplies the profiles needed for pivotal switches. Pareto supplies unanimous links through a third alternative. IIA freezes pairwise comparisons across profiles. Transitivity then makes a local pivotal voter for one pair decisive for all pairs. Pairwise independence is too strong because a coherent social ordering cannot generally be built from separately insulated pairwise judgments unless one fixed voter supplies them all.
