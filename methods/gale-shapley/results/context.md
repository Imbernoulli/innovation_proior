## Research Question

A finite admissions market has applicants on one side and institutions on the other. Each applicant ranks acceptable institutions, each institution ranks acceptable applicants, and each institution may have a quota. The immediate goal is not to maximize a numerical score. It is to assign applicants so that no applicant and institution can both improve by abandoning the assigned outcome and pairing with each other instead.

In the one-to-one specialization, each participant can be matched to at most one participant on the opposite side. In the many-to-one admissions version, an institution may hold several applicants up to capacity. The same obstruction appears in both cases: a pair outside the assignment can make the assignment socially and procedurally fragile if both sides prefer each other to their current positions.

## Background

The usual admissions process has two kinds of uncertainty. An institution does not know which applicants will accept offers, how applicants rank other institutions, or what other institutions will do. An applicant may face a waiting list at a preferred institution while holding a current offer elsewhere. The process can ask both sides to act before the final market state is clear.

Classical matching results address related but distinct settings. A marriage theorem certifies when representatives or bipartite matches exist, for participants without ordered preferences. A cost-assignment algorithm optimizes a matrix objective, treating the outcome as a sum of independent costs. Here the participants have ordered preferences and the criterion is the blocking-pair condition.

## Baselines

**Exhaustive matching search.** Generate all one-to-one assignments and test each for blocking pairs. The number of candidates is factorial in the number of participants.

**Cost or rank minimization.** Convert rankings into scores and solve an assignment problem. This produces an optimum for the chosen numerical score.

**Greedy high-rank pairing.** Commit mutual first choices or other highly ranked pairs first, then proceed to the remaining participants.

**Serial choice.** Let one side choose in a fixed order from the remaining options. The procedure terminates after each chooser has selected once.

## Evaluation Setting

The core setting uses two equal finite sides, strict complete rankings, and no ties. A correct construction should return a perfect one-to-one matching, terminate after finitely many pair considerations, and have no blocking pair.

The admissions setting allows institutional quotas, unacceptable partners, unequal side sizes, and unmatched participants. The same tests remain: feasibility with quotas, absence of blocking pairs, finite termination, and comparison with other stable assignments when several exist. Incentive behavior is a separate diagnostic because stability of the outcome does not automatically make truthful reporting safe for every participant.

## Code Framework

The framework has strict preference lists, inverse-rank tables for constant-time comparisons, a partial or complete matching representation, and a checker for blocking pairs. The construction remains to be filled in.

```python
def make_rank_table(preferences):
    return {
        agent: {other: rank for rank, other in enumerate(order)}
        for agent, order in preferences.items()
    }


def blocking_pairs(left_prefs, right_prefs, matching):
    right_rank = make_rank_table(right_prefs)
    left_partner = dict(matching)
    right_partner = {right: left for left, right in matching.items()}
    blocks = []

    for left, order in left_prefs.items():
        current_right = left_partner.get(left)
        for right in order:
            if right == current_right:
                break
            current_left = right_partner.get(right)
            if current_left is None or right_rank[right][left] < right_rank[right][current_left]:
                blocks.append((left, right))
    return blocks


def construct_matching(left_prefs, right_prefs):
    # TODO: fill in the constructive rule.
    pass
```
