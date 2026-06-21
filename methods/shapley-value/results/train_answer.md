The canonical method is the Shapley value, a fair-credit rule for transferable-utility cooperative games. I am given a finite player set N and a characteristic function v that maps every subset S of N to a real number v(S), with v(∅)=0. My goal is to produce a payoff vector φ(v) that assigns each player i a share of the total worth v(N). The difficulty is that a player's usefulness depends on which other players are already present, so no single marginal contribution tells the whole story. The Shapley value resolves this ambiguity by averaging a player's marginal contribution over every possible order in which the grand coalition could form.

I begin with the four structural requirements that any reasonable credit rule should satisfy. Efficiency demands that the entire worth v(N) be distributed, so the sum of all payoffs equals v(N). Symmetry demands that two players who are structurally identical receive identical payoffs; formally, if interchanging their names leaves every coalition value unchanged, their credits must be equal. The dummy axiom demands that a player who adds a fixed constant c to every coalition receive exactly c; in particular, a player who adds nothing receives zero. Finally, additivity demands that the value assigned to a sum of two games be the sum of the values assigned to each game separately. These four axioms are not merely desirable heuristics; they uniquely characterize the Shapley value.

The key insight is to remove the arbitrariness of a single arrival order. For a fixed permutation π of the players, let P_i(π) be the set of players that appear before player i. In that order, player i's marginal contribution is v(P_i(π) ∪ {i}) − v(P_i(π)). Because these marginal contributions telescope from v(∅) to v(N), their sum over all players equals v(N) for that order. However, choosing one particular order privileges an arbitrary narrative of how the coalition assembled. To avoid that privilege, I average over all n! permutations equally:

φ_i(v) = (1/n!) Σ_π [ v(P_i(π) ∪ {i}) − v(P_i(π)) ].

This is the Shapley value. It can be rewritten by grouping permutations according to the predecessor set S of player i. Each subset S ⊆ N \ {i} occurs as a predecessor set in exactly |S|! (n − |S| − 1)! permutations, giving the equivalent weighted-sum form

φ_i(v) = Σ_{S ⊆ N \ {i}} [|S|! (n − |S| − 1)! / n!] [ v(S ∪ {i}) − v(S) ].

The coefficient in front of each marginal contribution is therefore the probability that, in a uniformly random arrival order, the players in S arrive before i while the remaining players arrive after i.

I can verify directly that this rule satisfies the four axioms. Efficiency follows because each permutation's marginal contributions telescope to v(N), and averaging preserves the total. Symmetry follows because swapping the names of two structurally identical players pairs permutations in a way that leaves the average unchanged. The dummy axiom follows because if every marginal contribution equals c, then their average equals c. Additivity follows because marginal contributions themselves are linear: (v + w)(S ∪ {i}) − (v + w)(S) equals [v(S ∪ {i}) − v(S)] + [w(S ∪ {i}) − w(S)], and the same linear combination carries through the weighted average.

What makes the Shapley value especially compelling is uniqueness. To see why no other rule can satisfy the axioms, I decompose an arbitrary game into unanimity games. For each nonempty coalition T, define u_T by u_T(S) = 1 if T ⊆ S and u_T(S) = 0 otherwise. In u_T, every player outside T is a dummy with zero contribution, while the players inside T are perfectly symmetric. Efficiency forces the players in T to split the unit worth, and symmetry forces them to split it equally. Therefore any admissible rule must give φ_i(u_T) = 1/|T| if i ∈ T and φ_i(u_T) = 0 otherwise. Every cooperative game v has a unique expansion v = Σ_{T≠∅} a_T u_T, where the coefficients a_T are given by Möbius inversion on the subset lattice: a_T = Σ_{R ⊆ T} (−1)^{|T|−|R|} v(R). By additivity, any admissible rule is forced to be φ_i(v) = Σ_{T ∋ i} a_T / |T|. This expression is unique, and it coincides with the permutation-average formula because both assign the same values on each unanimity game.

The Shapley value thus has two complementary interpretations. Conceptually, it is the unique payoff rule compatible with efficiency, symmetry, dummy players, and additivity. Probabilistically, it is the expected marginal contribution of each player when the grand coalition forms in a uniformly random order. Both interpretations lead to the same formula, and together they explain why this method is the standard solution for fair credit allocation in cooperative game theory.

```python
from itertools import combinations, permutations

def shapley_value(players, v):
    """Compute the Shapley value for a cooperative game.

    players: tuple of distinct player labels
    v: dict mapping frozenset coalition -> real value, with v[frozenset()] == 0
    """
    n = len(players)
    phi = {i: 0.0 for i in players}
    fact = [1]
    for k in range(1, n + 1):
        fact.append(fact[-1] * k)
    for i in players:
        others = [p for p in players if p != i]
        total = 0.0
        for r in range(n):
            for S in combinations(others, r):
                S = frozenset(S)
                weight = fact[r] * fact[n - r - 1] / fact[n]
                total += weight * (v[S | {i}] - v[S])
        phi[i] = total
    return phi

if __name__ == "__main__":
    # Example: a 3-player game with complementarity and a dummy player.
    players = ("A", "B", "C")
    v = {
        frozenset(): 0.0,
        frozenset({"A"}): 0.0,
        frozenset({"B"}): 0.0,
        frozenset({"C"}): 0.0,
        frozenset({"A", "B"}): 6.0,
        frozenset({"A", "C"}): 0.0,
        frozenset({"B", "C"}): 0.0,
        frozenset({"A", "B", "C"}): 6.0,
    }
    phi = shapley_value(players, v)
    print("Shapley values:", phi)
    print("Efficiency check (sum):", sum(phi.values()), "==", v[frozenset(players)])
    print("Dummy C check:", phi["C"], "== 0")

    # Verify symmetry: A and B should share the 6.0 equally.
    print("Symmetry check A == B:", phi["A"], phi["B"])

    # Additivity check: write v as sum of two games.
    w = {S: 0.5 * val for S, val in v.items()}
    u = {S: 0.5 * val for S, val in v.items()}
    phi_w = shapley_value(players, w)
    phi_u = shapley_value(players, u)
    print("Additivity check:", all(abs(phi[p] - (phi_w[p] + phi_u[p])) < 1e-9 for p in players))
```
