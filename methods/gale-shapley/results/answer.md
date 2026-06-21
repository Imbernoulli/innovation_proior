# Gale-Shapley Deferred Acceptance

## Problem

Input two equal-size sets `P` and `R`. Each proposer `p in P` has a strict complete ranking of receivers, and each receiver `r in R` has a strict complete ranking of proposers.

A matching `M` is stable if there is no blocking pair `(p, r)` such that `p` prefers `r` to `M[p]` and `r` prefers `p` to the proposer matched with `r`.

## Algorithm

```text
Initially every proposer is free and every receiver holds nobody.

While some free proposer p still has an untried receiver:
    r = the highest-ranked receiver on p's list not yet proposed to
    p proposes to r

    If r holds nobody:
        r holds p
    Else if r prefers p to the currently held proposer p_old:
        r holds p
        p_old becomes free
    Else:
        p remains free

Return the pairs held by receivers.
```

The invariant is: proposals are tentative, but rejection is permanent. Proposers move downward through their lists; receivers only trade up among proposals already seen.

## Python Artifact

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

## Guarantees

**Termination.** Each proposer proposes to each receiver at most once. With `n` proposers and `n` receivers, the loop has at most `n^2` proposal events.

**Perfect matching in the complete equal-size case.** Once a receiver has any proposal, the receiver always holds one. If a proposer ended unmatched after exhausting all receivers, every receiver would be holding a distinct proposer, leaving no unmatched proposer. Contradiction.

**Stability.** Consider any unmatched pair `(p, r)`. If `p` never proposed to `r`, then `p` prefers the final partner to `r`. If `p` did propose to `r`, then `r` rejected `p` at some time for a preferred held proposer and only improved afterward. Thus `r` does not prefer `p` to the final partner. No outside pair can block.

**Proposer-optimality.** A receiver is possible for a proposer if some stable matching pairs them. No proposer is ever rejected by a possible receiver. At the first such rejection, the receiver keeps a better proposer who has not previously been rejected by any possible receiver ranked above this one; in a stable matching that paired the rejected proposer with this receiver, the better proposer and this receiver would therefore block. Thus each proposer receives the best possible partner among all stable matchings.

## Many-to-One Admissions

For a college with quota `q`, replace the single held proposer with a waiting list of up to `q` applicants. After new applications arrive, the college keeps its top `q` among old held applicants plus new applicants, or all of them if fewer than `q` are present, and rejects the rest. A rejection means the college is filled with applicants it ranks higher than the rejected applicant; later replacements only improve that cutoff. If a college finishes below quota, it has not rejected any acceptable applicant who would rather attend. Those are the many-to-one versions of the one-to-one rejection cases.

## Incentive Caveat

The proposing side receives the favorable stable outcome. Stability alone is not a proof that every participant on both sides can safely report preferences truthfully.
