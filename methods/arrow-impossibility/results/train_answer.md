The problem is to aggregate many individual rankings over at least three alternatives into one complete and transitive social ordering, using only ordinal information. Each voter supplies a strict ranking, and the rule must return a strict ranking. The requirements look mild: the rule must accept every possible profile of rankings, it must respect unanimous preferences, and the social comparison between two alternatives must depend only on how voters compare those two alternatives. With only two alternatives these demands are easy to meet, but once three or more alternatives are present, pairwise decisions can be locally sensible yet globally inconsistent.

Existing ideas show exactly where the difficulty lies. Pairwise majority treats each pair democratically and ignores irrelevant alternatives, but the Condorcet cycle shows that it can produce x over y, y over z, and z over x, which is not a social ordering. The Borda count always returns an ordering, but its comparison of x and y can change when an unrelated alternative is inserted or moved in the voters' rankings, so it violates independence of irrelevant alternatives. A lexicographic dictatorship satisfies the first three axioms, but it violates the non-dictatorship requirement by always copying one fixed voter. Domain restrictions such as single-peakedness avoid cycles, but they do so by excluding the very profiles that make the problem hard. This suggests that the tension is structural rather than a quirk of any particular voting rule.

The result is Arrow's impossibility theorem. It states that on an unrestricted domain with at least three alternatives, every social welfare function satisfying unrestricted domain, Pareto unanimity, and independence of irrelevant alternatives must be dictatorial. The proof does not inspect every possible scoring formula. Instead it tracks which coalitions of voters have enough power to force a pairwise social outcome.

A coalition S is called decisive for x over y if, whenever every voter in S ranks x above y and every voter outside S ranks y above x, the social ordering also ranks x above y. Pareto makes the whole electorate decisive for every ordered pair, and the empty coalition is never decisive. Choose a decisive coalition that is minimal under inclusion. If such a coalition is decisive for one ordered pair, then it is decisive for every ordered pair. To see why, suppose S is decisive for x over y and introduce a third alternative z. Construct a profile in which voters in S rank x above y above z, while voters outside S rank y above z above x. Decisiveness gives society x above y, Pareto gives society y above z, and transitivity then gives society x above z. Because independence of irrelevant alternatives freezes the social comparison of x and z to depend only on individual comparisons of x and z, the same conclusion must hold in every profile where exactly the members of S rank x above z and everyone else ranks z above x. Relabeling alternatives transfers decisiveness to all ordered pairs.

It remains to show that a minimal decisive coalition must contain exactly one voter. Suppose it contained at least two voters and split it into nonempty disjoint subsets A and B. Order the voters in the coalition and move them one by one from ranking x above y to ranking y above x, while all voters outside the coalition rank y above x. At the start the coalition forces society to rank x above y; at the end Pareto forces society to rank y above x. Therefore some voter i is the first whose switch changes the social comparison of x and y. By inserting a third alternative z in the right places around the pivotal boundary, one obtains profiles where transitivity forces x above z before the switch and z above x after the switch. Independence of irrelevant alternatives then makes voter i decisive for x over z regardless of how the other voters rank that pair. Repeating the argument for all pairs shows that voter i is decisive for every ordered pair, so the social ordering copies voter i's ranking. The only rules that survive the three axioms are therefore dictatorial.

The theorem shows that pairwise independence is too strong: it denies the rule the cross-alternative information needed to keep a social ordering coherent, while transitivity still demands coherence. The only consistent source of pairwise judgments that remains is a single fixed voter.

```python
from itertools import permutations, product, combinations

ALTS = ('x', 'y', 'z')


def prefers(ranking, a, b):
    return ranking.index(a) < ranking.index(b)


def dictatorial_factory(voter_idx):
    def rule(profile):
        return profile[voter_idx]
    return rule


def borda_rule(profile):
    n = len(profile[0])
    scores = {alt: 0 for alt in profile[0]}
    for r in profile:
        for pos, alt in enumerate(r):
            scores[alt] += n - 1 - pos
    return tuple(sorted(scores, key=lambda a: (-scores[a], a)))


def check_pareto(rule, profiles):
    for profile in profiles:
        soc = rule(profile)
        for a, b in permutations(profile[0], 2):
            if all(prefers(r, a, b) for r in profile) and not prefers(soc, a, b):
                return False
    return True


def check_iia(rule, profiles):
    pairs = list(permutations(profiles[0][0], 2))
    for a, b in pairs:
        groups = {}
        for profile in profiles:
            key = tuple(prefers(r, a, b) for r in profile)
            if key not in groups:
                groups[key] = prefers(rule(profile), a, b)
            elif prefers(rule(profile), a, b) != groups[key]:
                return False
    return True


def find_dictator(rule, profiles, n_voters):
    pairs = list(permutations(profiles[0][0], 2))
    for i in range(n_voters):
        ok = True
        for profile in profiles:
            soc = rule(profile)
            for a, b in pairs:
                if prefers(profile[i], a, b) and not prefers(soc, a, b):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return i
    return None


def arrow_check(rule, alternatives, n_voters):
    profiles = list(product(permutations(alternatives), repeat=n_voters))
    pareto_ok = check_pareto(rule, profiles)
    iia_ok = check_iia(rule, profiles)
    dictator = find_dictator(rule, profiles, n_voters)
    print(f"Pareto: {pareto_ok}, IIA: {iia_ok}, dictator found: {dictator}")
    return pareto_ok, iia_ok, dictator


if __name__ == "__main__":
    # Pairwise majority cycles on the Condorcet profile.
    cycle = (('x', 'y', 'z'), ('y', 'z', 'x'), ('z', 'x', 'y'))
    print("Condorcet cycle (majority pairwise outcomes):")
    for a, b in [('x', 'y'), ('y', 'z'), ('z', 'x')]:
        wins_a = sum(prefers(r, a, b) for r in cycle)
        wins_b = len(cycle) - wins_a
        print(f"  {a} vs {b}: {a} wins {wins_a}-{wins_b}")

    # A dictatorial rule satisfies the axioms.
    print("\nDictatorial rule (voter 0 decides):")
    arrow_check(dictatorial_factory(0), ALTS, 2)

    # Borda count violates independence of irrelevant alternatives.
    print("\nBorda count:")
    arrow_check(borda_rule, ALTS, 2)
```
