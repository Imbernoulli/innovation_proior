The problem is to design a voting rule that lets a finite group of voters choose one alternative from at least three options. Each voter holds a strict ranking of the alternatives, and the rule must be responsive enough that every alternative can win for some profile. It must also be incentive compatible in the strongest sense: no voter should ever profit by reporting a ranking different from the true one, no matter what everyone else reports. Familiar rules quickly fail this test. Plurality rewards voters for abandoning a weak favorite and supporting a stronger compromise. Borda count lets voters manipulate the relative scores of rivals. Runoff rules create incentives to engineer which alternatives survive to the final round. The question is not whether any particular named rule is manipulable, but whether any deterministic rule can satisfy all three requirements at once: onto, strategy-proof, and non-dictatorial.

The answer is negative. The result is the Gibbard-Satterthwaite theorem. For at least three alternatives and unrestricted strict preference rankings, every deterministic voting rule that is onto and strategy-proof must be dictatorial. Equivalently, any onto non-dictatorial rule is manipulable. The intuition is that strategy-proofness is much stronger than it first appears: it forces the rule to be monotone, so raising the winning alternative in some voter's ranking can never make it lose. Once monotonicity is in place, the onto property prevents the rule from ever selecting a Pareto-dominated alternative. These two structural facts then create pivotal configurations in which one voter's top-ranked alternative must be decisive. With three or more alternatives, that decisiveness cannot be split across different voters without contradiction, so a single voter must dictate the outcome.

The proof proceeds in three stages. First, strategy-proofness implies monotonicity. If raising the winning alternative relative to the alternatives it previously beat could change the outcome, then along a one-voter-at-a-time path there would be a first voter whose report change alters the outcome, and strict preferences would make that change profitable for that voter in one direction or the other. Second, monotonicity plus onto implies weak Pareto optimality: if every voter ranks a above b, the rule cannot select b, because onto gives a profile where a wins, and monotonicity lets both profiles be moved to a common comparison profile where a must both win and lose. Third, the decisive voter is located. In the two-voter case, place voter 1 with a above b above all else and voter 2 with b above a above all else. Weak Pareto says the outcome is a or b, and whichever wins identifies a voter who can force that alternative by ranking it first. Repeating over all pairs shows one voter is decisive for every alternative, hence a dictator. The general case reduces to two voters by grouping all but one voter into a single block and applying induction.

The theorem is not constructive in the sense of producing a new algorithm; it is an impossibility result. But the reasoning can be turned into a verification procedure: enumerate small voting rules, test whether they are onto and strategy-proof, and confirm that any rule passing both tests is dictatorial. The code below implements this check for two voters and three alternatives.

```python
import itertools

alternatives = [0, 1, 2]
rankings = list(itertools.permutations(alternatives))

def top(rank):
    return rank[0]

def prefers(rank, a, b):
    return rank.index(a) < rank.index(b)

def is_onto(rule):
    """Every alternative wins at some profile."""
    seen = set()
    for profile in itertools.product(rankings, repeat=2):
        seen.add(rule(profile))
    return seen == set(alternatives)

def is_strategy_proof(rule):
    """No voter can gain by misreporting."""
    for profile in itertools.product(rankings, repeat=2):
        true0, true1 = profile
        outcome = rule(profile)
        # voter 0 deviations
        for fake0 in rankings:
            dev_out = rule((fake0, true1))
            if prefers(true0, dev_out, outcome):
                return False
        # voter 1 deviations
        for fake1 in rankings:
            dev_out = rule((true0, fake1))
            if prefers(true1, dev_out, outcome):
                return False
    return True

def is_dictatorial(rule):
    """Some voter's top alternative always wins."""
    for voter in [0, 1]:
        if all(rule(p) == top(p[voter]) for p in itertools.product(rankings, repeat=2)):
            return True
    return False

def count_onto_strategyproof_rules():
    count = 0
    for values in itertools.product(alternatives, repeat=len(rankings) ** 2):
        table = {}
        for i, profile in enumerate(itertools.product(rankings, repeat=2)):
            table[profile] = values[i]
        rule = lambda p, table=table: table[p]
        if is_onto(rule) and is_strategy_proof(rule):
            count += 1
            assert is_dictatorial(rule), "found a counterexample to Gibbard-Satterthwaite"
    return count

if __name__ == "__main__":
    n = count_onto_strategyproof_rules()
    print(f"Number of onto, strategy-proof rules for 2 voters and 3 alternatives: {n}")
    print("All of them are dictatorial.")
```

The theorem therefore sets a hard boundary. When collective choice is taken seriously and truthfulness is demanded in dominant strategies, the only deterministic escape from manipulation is to concentrate all power in one voter. Any rule that spreads influence across the electorate must admit a profile at which some voter benefits from lying.
