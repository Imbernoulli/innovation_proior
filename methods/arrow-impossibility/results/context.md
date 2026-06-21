## Research question

How can individual rankings over at least three alternatives be aggregated into a single social ordering? Each voter supplies a complete and transitive preference relation, and the rule must return a complete and transitive social preference relation. The intended rule is not a scoring rule over utilities or a procedure for bargaining over intensities; it uses only ordinal rankings.

Several requirements are imposed. The rule should be defined for every possible profile of individual rankings. If everyone ranks `x` above `y`, society should rank `x` above `y`. The social comparison between `x` and `y` should depend only on how individuals compare `x` and `y`, not on where they rank unrelated alternatives. And the rule should not simply copy one fixed voter for every profile and every pair.

The question is how rules with these properties behave when the social output is required to be a full ordering rather than a disconnected list of pairwise answers.

## Background

Ordinal social choice begins from the fact that interpersonal utility comparisons are unavailable or deliberately excluded. Individual inputs are rankings, so a social welfare function takes a profile of rankings and returns a social ranking. The inputs carry only ordinal information, with no cardinal measure of intensity.

Pairwise majority is the natural democratic baseline. For two alternatives it behaves cleanly: if more voters prefer `x` to `y` than prefer `y` to `x`, choose `x`. With three alternatives, majority comparisons can cycle. A profile may have one group ranking `x > y > z`, another `y > z > x`, and another `z > x > y`; then majorities give `x > y`, `y > z`, and `z > x`.

The unrestricted-domain requirement keeps all such profiles in play, including ones with no single-peaked structure or common ideology. The social ordering requirement keeps the output coherent across triples. Pareto unanimity asks the rule not to reverse a pair on which everyone agrees. Independence of irrelevant alternatives asks the rule's judgment on a pair to be determined entirely by the voters' judgments on that pair.

Under these assumptions each pair is decided locally, from the individual comparisons on that pair alone, while transitivity is a global constraint that ties the pairwise judgments together into one consistent ordering.

## Baselines

- **Pairwise majority rule.** For each pair, choose the option preferred by a majority. It expresses a strong pairwise democratic intuition and, at the pair level, uses only the comparison on that pair.

- **Borda count.** Assign points by rank position and add them across voters. After tie-breaking it returns an ordering, and the social comparison between `x` and `y` draws on the full set of alternatives, since inserting, removing, or moving an unrelated alternative in individual rankings can change the points.

- **Lexicographic dictatorship.** Fix an ordered list of voters and use the first voter to decide every non-tied comparison. It returns a complete and transitive ordering and decides each pair from that pair's comparison alone.

- **Domain restrictions.** If preferences are single-peaked on a line, majority rule avoids cycles. This applies to the restricted domain rather than to every profile of rankings.

## Evaluation settings

The basic setting has a finite voter set, at least three alternatives, and strict individual rankings. The output is a strict social ordering. The core tests are unrestricted domain, Pareto unanimity, independence of irrelevant alternatives, transitivity of the social ordering, and non-dictatorship.

For proof checking, it is enough to reason about three alternatives at a time, because any larger agenda contains triples. Profiles are varied freely: for any two alternatives, a chosen coalition can be made to prefer one over the other while the remaining voters prefer the reverse, and a third alternative can be positioned above, between, or below them as needed.

A coalition is said to be decisive for `x` over `y` if whenever all members of the coalition prefer `x` to `y` and all nonmembers prefer `y` to `x`, the social ordering ranks `x` above `y`. This coalition language is the natural internal yardstick for reasoning about which subsets of voters control which pairwise outcomes.
