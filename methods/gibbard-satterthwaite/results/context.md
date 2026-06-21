## Research question

We want a voting rule that takes a finite profile of strict individual rankings over a finite set of social alternatives and returns one alternative. The issue is not recombining rankings into an attractive compromise; it is whether incentive compatibility can coexist with genuine collective choice. The goal is demanding but natural: every alternative should be reachable for some profile, no single voter should always get their top-ranked outcome, and voters should never be able to obtain a strictly better outcome by misreporting their preferences.

The tension is between collective choice and incentives. A rule can be responsive to many voters, or it can try to make truthful reporting a dominant strategy. With at least three alternatives, the question is whether those two aims can coexist under the mild nonimposition requirement that no alternative is ruled out in advance.

## Background

A preference profile is an ordered list of voters' strict rankings. A social choice rule maps each profile to a single alternative. The incentive property is strategy-proofness: fixing everyone else's reports, no voter can switch from their true ranking to another report and thereby obtain an outcome they strictly prefer.

Onto, also called nonimposition, says that each alternative appears as the chosen outcome at some profile. It rules out the trivial escape of making an alternative unavailable. Dictatorship says that there is a voter whose top-ranked alternative is selected at every profile. Dictatorship is strategy-proof and onto, but it is exactly the kind of concentration of power the rule is meant to avoid.

Two elementary consequences drive the direct route. The first is monotonicity. If a strategy-proof rule selects alternative `a` at a profile, and a new profile only moves `a` upward relative to alternatives it previously beat in each voter's ranking, the rule must still select `a`. Otherwise a voter on the path from the first profile to the second would benefit by reporting the other ranking. The second is weak Pareto optimality. If every voter ranks `a` above `b`, an onto and strategy-proof rule cannot select `b`; onto provides some profile where `a` is selected, and monotonicity lets one move to a profile with `a` above `b` for all voters while preserving the relevant outcome.

This line sits near Arrow's impossibility theorem but does not simply repeat it. Arrow studies social welfare functions that return complete social rankings and imposes independence of irrelevant alternatives. The voting-rule problem returns only one chosen alternative and imposes dominant-strategy truthfulness. One common proof route translates strategy-proofness into an independence-like property and then invokes Arrow; the direct route instead builds a dictator by tracking how a decisive or pivotal voter emerges from monotonicity and nonimposition.

## Baselines

Plurality chooses the alternative with the most first-place votes. It is onto and non-dictatorial, but a voter may have reason to abandon a favorite with little support and vote for a more viable compromise. The rule reacts strongly to reports, and that responsiveness creates profitable misreports.

Runoff-style rules use first-place support and then eliminate or compare finalists. They are also onto and non-dictatorial, but a voter can sometimes change which alternatives survive to the later stage. The strategic problem is not just ranking a favorite first; it is controlling the agenda induced by the reported profile.

Borda count assigns scores by rank position and chooses the highest total score. It uses more information than plurality, but the additional rank information creates more dimensions for strategic movement. A voter may lower a strong rival or raise a compromise, changing the score comparison in a favorable direction.

Pairwise majority and Condorcet-oriented procedures make majority comparisons central. With three or more alternatives, majority preferences can cycle, so any single-valued selection rule must choose a way to break or resolve cycles. That choice point can become strategically relevant.

Dictatorship is the incentive-compatible benchmark. It is strategy-proof because the decisive voter's best report is truthful and no other voter can affect the outcome. It is onto when the dictator can rank any alternative first. Its gap is normative rather than technical: it abandons collective choice.

## Evaluation settings

The natural mathematical setting is a finite set of voters, a finite set of at least three alternatives, unrestricted strict preference rankings, and deterministic single-valued voting rules. The properties to check are strategy-proofness, onto/nonimposition, and dictatorship.

The proof task is not empirical. The relevant yardstick is a theorem: either exhibit a rule satisfying strategy-proofness, onto, and non-dictatorship, or show that these requirements are mutually inconsistent. Counterexamples to familiar rules help motivate the question, but the target result is universal over all deterministic rules on the unrestricted domain.

## Code framework

For this theorem, the field-appropriate artifact is a proof rather than executable code. The proof framework has only neutral mathematical objects:

```text
Input:
  finite voters N
  finite alternatives A with |A| >= 3
  strict rankings over A
  voting rule f: rankings^N -> A

Assumptions:
  f is strategy-proof
  f is onto

Proof slots:
  establish monotonicity from strategy-proofness
  establish weak Pareto from onto plus monotonicity
  identify a voter who is decisive for an alternative
  extend decisiveness across alternatives and voters
  conclude dictatorship
```
