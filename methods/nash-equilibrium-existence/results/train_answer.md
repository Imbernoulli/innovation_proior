I present the Nash equilibrium existence theorem, the canonical result that every finite normal-form game has at least one equilibrium point in mixed strategies. The theorem resolves the problem left open by von Neumann's minimax theory, which only handled two-person zero-sum games. Once we allow three or more players, or two players whose payoffs are not strictly opposed, there is no longer a single "value" that summarizes the game, because the players are not pushing a common scalar in opposite directions. What remains is a stability condition: no player can improve their expected payoff by changing their own strategy unilaterally while everyone else stays fixed. A profile satisfying this for every player is a Nash equilibrium, and Nash's theorem proves that such a profile always exists.

The setup is minimal. We have finitely many players, each with finitely many pure strategies. A mixed strategy for a player is a probability distribution over their pure strategies, represented as a point on a simplex. A profile of mixed strategies lives in the product of these simplices, which is a compact convex polytope. The payoff to each player is extended to mixed strategies by taking expectations, and this extension is multilinear, meaning it is linear in each player's own mixture when the others are held fixed. This linearity is the structural fact that makes everything work.

The definition of equilibrium is therefore natural. A profile is an equilibrium if every player's strategy maximizes their own payoff given the strategies of the others. Because the payoff is linear in a player's own mixture, the maximum over all mixed strategies is attained at a pure strategy. So the condition simplifies: every pure strategy that a player uses with positive probability must be a best response. In the special case of a two-person zero-sum game, this condition coincides exactly with the von Neumann saddle point, so Nash equilibrium genuinely generalizes the minimax solution rather than replacing it with something foreign.

The central question is existence. The proof I give here uses Brouwer's fixed point theorem directly, by constructing a continuous self-map of the product of simplices whose fixed points are exactly the equilibria. The trick is to convert the best-response condition into a gain that vanishes precisely at equilibrium. For each player and each pure strategy, define the gain of deviating to that pure strategy from the current mixed strategy as the amount by which the pure strategy's payoff exceeds the current mixture's payoff, or zero if it does not exceed it. This gain is non-negative, and it is zero for every pure strategy simultaneously exactly when the player is already best-responding.

The map then nudges each player's mixture toward the better pure strategies. Starting from the current mixture, add a bonus to each pure strategy proportional to its gain, then renormalize so the result is again a probability distribution. The old mixture contributes total mass one, and the gains contribute total mass equal to their sum, so the denominator is one plus that sum, which is never zero. Every step is continuous, so the whole map is a continuous self-map of a compact convex set. Brouwer's theorem gives a fixed point.

At a fixed point, consider any pure strategy that the player actually uses. Its payoff is at most the player's current mixture payoff, because the mixture payoff is an average over the used pure strategies. Hence its gain is zero. But in the update rule, its weight is divided by one plus the sum of all gains. For the weight to remain unchanged, the denominator must be one, so the sum of all gains is zero. Since every gain is non-negative, every gain must be zero. That means no pure strategy gives a higher payoff than the current mixture, which is exactly the best-response condition. The converse is immediate: at an equilibrium all gains vanish, so the update leaves the profile unchanged. Thus the fixed points of the map are exactly the Nash equilibria, and Brouwer's theorem guarantees at least one fixed point.

The same construction has useful structural consequences. Because the map is defined entirely from the payoffs, it commutes with every symmetry of the game. Restricting to symmetric profiles, which form a non-empty compact convex subcell, gives a symmetric equilibrium. In solvable games, where equilibrium components can be mixed and matched, each player's set of equilibrium strategies is a polyhedral convex set, obtained by intersecting the simplex with finitely many linear inequalities. And strictly dominated strategies can never appear in equilibrium, which gives a practical lever for narrowing down equilibria in concrete examples.

The following Python script illustrates the theorem on a concrete non-zero-sum two-player game. It defines the gain function and the Brouwer map, verifies a known fully mixed Nash equilibrium by checking that every used pure strategy is a best response, confirms that the gain vector vanishes so the profile is a fixed point of the map, and shows that no unilateral deviation improves either player's payoff.

```python
import numpy as np

# A non-zero-sum 2x2 game with a unique fully mixed Nash equilibrium.
# Rows: U (index 0), D (index 1). Columns: L (index 0), R (index 1).
# payoff[0] = row player's payoff; payoff[1] = column player's payoff.
payoff = np.array([
    [[2.0, 0.0],   # row payoff: (U,L)=2, (U,R)=0
     [0.0, 3.0]],  # row payoff: (D,L)=0, (D,R)=3
    [[0.0, 3.0],   # col payoff: (U,L)=0, (U,R)=3
     [2.0, 0.0]],  # col payoff: (D,L)=2, (D,R)=0
])

# Known Nash equilibrium computed by equating payoffs of the two pure strategies.
eq = [np.array([0.4, 0.6]),   # row: U with prob 2/5, D with prob 3/5
      np.array([0.6, 0.4])]   # col: L with prob 3/5, R with prob 2/5

def expected_payoff(i, profile):
    """Expected payoff to player i under mixed strategy profile."""
    p = payoff[i]
    for j in range(2):
        p = np.tensordot(p, profile[j], axes=([0], [0]))
    return float(p)

def pure_payoffs(i, profile):
    """Payoff to player i from each pure strategy, opponents held fixed."""
    p = payoff[i]
    for j in range(2):
        if j == i:
            continue
        p = np.tensordot(p, profile[j], axes=([0], [0]))
    return p

def brouwer_step(profile):
    """One step of the Nash-Brouwer gain-renormalization map."""
    new_profile = []
    for i in range(2):
        current = expected_payoff(i, profile)
        pure = pure_payoffs(i, profile)
        gains = np.maximum(0.0, pure - current)
        numerator = profile[i] + gains
        denominator = 1.0 + gains.sum()
        new_profile.append(numerator / denominator)
    return new_profile

def is_equilibrium(profile, tol=1e-9):
    """Every pure strategy in the support is a best response."""
    for i in range(2):
        current = expected_payoff(i, profile)
        pure = pure_payoffs(i, profile)
        best = pure.max()
        if best - current > tol:
            return False
        used = profile[i] > tol
        if np.any(best - pure[used] > tol):
            return False
    return True

print("Equilibrium profile:", eq)
print("Equilibrium check:", is_equilibrium(eq))
print("Row player payoff:", expected_payoff(0, eq))
print("Column player payoff:", expected_payoff(1, eq))

# Verify that the gain vector vanishes, so the profile is a fixed point of T.
for i in range(2):
    pure = pure_payoffs(i, eq)
    current = expected_payoff(i, eq)
    gains = np.maximum(0.0, pure - current)
    print(f"Player {i} pure payoffs: {pure}, gains: {gains}")

stepped = brouwer_step(eq)
print("After one Brouwer step:", stepped)
print("Fixed point check:", all(np.allclose(eq[i], stepped[i]) for i in range(2)))

# Show that a unilateral deviation cannot improve either player's payoff.
for i in range(2):
    for a in range(2):
        dev = [eq[j].copy() for j in range(2)]
        dev[i] = np.array([1.0, 0.0]) if a == 0 else np.array([0.0, 1.0])
        print(f"Player {i} deviates to pure {a}: payoff = {expected_payoff(i, dev):.6f}")
```

Running this script produces a mixed profile in which each player randomizes over their pure strategies in such a way that no unilateral deviation is profitable. The numerical fixed point of the gain-renormalization map therefore confirms the abstract theorem on a concrete example. The Nash equilibrium existence theorem is the foundational guarantee that makes non-cooperative game theory possible: no matter how many players there are or how their payoffs relate, there is always at least one stable profile where every player is doing the best they can, given what everyone else is doing.
