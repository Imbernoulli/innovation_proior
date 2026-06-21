We are asked to pair two sides of a market, such as applicants and institutions, so that no applicant and institution outside their current assignment would both prefer each other to their current partner. The obstruction is local and pairwise, not global. If we try to optimize a numerical score built from rankings, we may find a matching that is optimal on paper but still contains a pair that would rather defect. A greedy strategy that commits highly ranked pairs first has no general certificate for the remaining agents, because an early commitment can prevent a later blocking pair from being recognized. Serial choice by one side ignores the other side's preferences until it is too late, so a chosen applicant may later prefer a later chooser and be preferred in return. The right certificate is not a low cost but a history of justified rejections that explains why every tempting outside pair is impossible.

The method that solves this is the Gale-Shapley deferred acceptance algorithm. One side, the proposers, repeatedly offers itself to its most preferred receiver that it has not yet asked. Each receiver holds at most one proposer tentatively and compares any new proposal against the currently held proposer. If the new proposer is preferred, the receiver switches and the old proposer is released; otherwise the new proposer is rejected. A rejection is permanent: once a receiver has rejected a proposer for someone better, that receiver can only end up with someone even better as more proposals arrive. Proposers move downward through their lists, so every agent proposes to each receiver at most once. The process stops when no free proposer has an untried receiver, and the held pairs form the final matching.

This construction guarantees termination because there are finitely many possible proposals. In the equal-size complete case it returns a perfect matching: if a proposer were unmatched after exhausting all receivers, every receiver would be holding a distinct proposer, which accounts for all proposers and contradicts the unmatched proposer. Stability follows from the history of each outside pair. If a proposer never proposed to a given receiver, then the proposer prefers its final match. If it did propose and was rejected, the receiver was holding a preferred proposer at that moment and only improved afterward, so the receiver does not prefer the rejected proposer at the end. Therefore no unmatched pair can block the matching. The asymmetry of the procedure also gives each proposer the best partner it could obtain in any stable matching, while the receiving side generally receives a less favorable stable outcome.

```python
from collections import deque


def make_rank_table(preferences):
    return {
        agent: {candidate: rank for rank, candidate in enumerate(order)}
        for agent, order in preferences.items()
    }


def gale_shapley(proposer_prefs, receiver_prefs):
    receiver_rank = make_rank_table(receiver_prefs)
    next_choice = {proposer: 0 for proposer in proposer_prefs}
    held_by_receiver = {receiver: None for receiver in receiver_prefs}
    free = deque(proposer_prefs)

    while free:
        proposer = free.popleft()
        if next_choice[proposer] >= len(proposer_prefs[proposer]):
            continue

        receiver = proposer_prefs[proposer][next_choice[proposer]]
        next_choice[proposer] += 1
        held = held_by_receiver[receiver]

        if held is None:
            held_by_receiver[receiver] = proposer
        elif receiver_rank[receiver][proposer] < receiver_rank[receiver][held]:
            held_by_receiver[receiver] = proposer
            free.append(held)
        else:
            free.append(proposer)

    return {
        proposer: receiver
        for receiver, proposer in held_by_receiver.items()
        if proposer is not None
    }
```

The same idea extends to many-to-one admissions by giving each institution a quota. A college holds up to its quota of applicants and, whenever a new application arrives, keeps the best quota among its current held set plus the new applicant, rejecting the rest. A rejection then means the college is filled with applicants it ranks higher than the rejected one, and the cutoff can only rise as better applicants arrive. If a college finishes below quota, it has not rejected any acceptable applicant who would rather attend. The one-to-one deferred acceptance algorithm is the canonical core of this family, and stability remains the same rejection certificate scaled by capacity.
