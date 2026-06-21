## Research question

How can individual rankings over at least three alternatives be aggregated into a single social ordering? Each voter supplies a complete and transitive preference relation, and the rule must return a complete and transitive social preference relation. The intended rule is not a scoring rule over utilities or a procedure for bargaining over intensities; it uses only ordinal rankings.

The desiderata look modest. The rule should be defined for every possible profile of individual rankings. If everyone ranks `x` above `y`, society should rank `x` above `y`. The social comparison between `x` and `y` should depend only on how individuals compare `x` and `y`, not on where they rank unrelated alternatives. The rule should not simply copy one fixed voter for every profile and every pair.

The question is whether these requirements can coexist once the social output is required to be a full ordering rather than a disconnected list of pairwise majority answers.

## Background

Ordinal social choice begins from the fact that interpersonal utility comparisons are unavailable or deliberately excluded. Individual inputs are rankings, so a social welfare function takes a profile of rankings and returns a social ranking. This makes the aggregation problem cleaner, but it also removes cardinal information that might otherwise break ties or measure intensity.

Pairwise majority is the natural democratic baseline. For two alternatives it behaves well: if more voters prefer `x` to `y` than prefer `y` to `x`, choose `x`. With three alternatives, however, majority comparisons can cycle. A profile may have one group ranking `x > y > z`, another `y > z > x`, and another `z > x > y`; then majorities give `x > y`, `y > z`, and `z > x`. Pairwise plausibility does not automatically assemble into a transitive social ordering.

The unrestricted-domain requirement keeps all such profiles in play. It blocks the escape route of assuming single-peaked preferences, common ideology, or any other structure that prevents cycles. The social ordering requirement keeps the output coherent across triples. Pareto unanimity prevents the rule from reversing a pair on which everyone agrees. Independence of irrelevant alternatives requires the rule's judgment on a pair to be determined entirely by the voters' judgments on that pair.

Those assumptions make each pair look locally isolated, while transitivity forces the pairwise judgments to fit together globally. The tension is not recombination of existing rankings into a clever composite ranking. It is the demand that every pair be decided without looking sideways, while the final object must still be one consistent ordering.

## Baselines

- **Pairwise majority rule.** For each pair, choose the option preferred by a majority. It satisfies a strong pairwise democratic intuition and ignores irrelevant alternatives at the pair level, but it can produce cycles, so it does not always return a social ordering.

- **Borda count.** Assign points by rank position and add them across voters. It always returns an ordering after tie-breaking and uses all alternatives to stabilize the aggregate. Its gap is exactly that the social comparison between `x` and `y` can change when an unrelated alternative is inserted, removed, or moved in individual rankings.

- **Lexicographic dictatorship.** Fix an ordered list of voters and use the first voter to decide every non-tied comparison. It gives a complete and transitive ordering and respects independence, but it violates the non-dictatorship demand.

- **Domain restrictions.** If preferences are single-peaked on a line, majority rule avoids cycles. This solves a different problem: the hard case is a rule that works for every profile of rankings.

## Evaluation settings

The basic setting has a finite voter set, at least three alternatives, and strict individual rankings. The output is a strict social ordering. The core tests are unrestricted domain, Pareto unanimity, independence of irrelevant alternatives, transitivity of the social ordering, and non-dictatorship.

For proof checking, it is enough to reason about three alternatives at a time, because any larger agenda contains triples. Profiles are varied freely: for any two alternatives, a chosen coalition can be made to prefer one over the other while the remaining voters prefer the reverse, and a third alternative can be positioned above, between, or below them as needed.

The decisive-coalition formulation is the natural internal yardstick. A coalition is decisive for `x` over `y` if whenever all members of the coalition prefer `x` to `y` and all nonmembers prefer `y` to `x`, the social ordering ranks `x` above `y`. The proof asks whether the family of decisive coalitions can avoid collapsing to a singleton.


