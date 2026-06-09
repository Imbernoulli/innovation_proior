# Context

## Research question

Two people sit across a table at a game of pure opposition: every franc one wins, the other loses. Each has a finite list of plans of action. Player 1 picks a row $x \in \{1,\dots,m\}$, player 2 picks a column $y \in \{1,\dots,n\}$, simultaneously and in ignorance of each other's choice, and a fixed rule pays player 1 the amount $g(x,y)$ and player 2 the amount $-g(x,y)$. The matrix $[g(x,y)]$ is the whole game.

The phrase everyone uses — "play as advantageously as possible" — has no clear meaning here. Player 1 would like the largest possible $g$, but he does not control $g$: it depends on $y$, which player 2 controls, and player 2's interests are exactly opposed. The situation is circular: each player's best move depends on the other's, whose best move depends back on the first's. For a one-player "game" ($n=1$) the problem collapses to a plain maximization and there is nothing to say; the difficulty is intrinsic to genuine opposition with $n\ge 2$.

So the question must be sharpened into something a single player can actually compute against the worst case. Two precise quantities suggest themselves. If player 1 had to reveal his choice $x$ first, player 2 would then drive the payoff down to $\min_y g(x,y)$; anticipating this, player 1 picks $x$ to secure $\max_x \min_y g(x,y)$ — the most he can *guarantee* no matter what. Dually, $\min_y \max_x g(x,y)$ is the least player 2 can *hold him to*. A satisfactory theory would have to: (i) say what number is "the value" of the game to player 1; (ii) say what each player should actually do to realize it; and (iii) do this without any assumption about who is the cleverer psychologist or who finds out whom. Whether such a value even exists — whether the amount one can guarantee equals the amount the opponent can impose — is the open mathematical question.

## Background

**Strict opposition and the secure value.** For a zero-sum two-person game the two players' payoffs are negatives ($g_1=g$, $g_2=-g$), so the only datum is the single matrix $g(x,y)$. The "maximin" $\max_x\min_y g$ is the floor player 1 can guarantee against a fully adversarial opponent; the "minimax" $\min_y\max_x g$ is the ceiling player 2 can enforce. A general and elementary fact holds for *any* matrix: $\max_x\min_y g(x,y)\le \min_y\max_x g(x,y)$. The player who chooses second, or whose move is found out last, is never worse off. When the two coincide the game has a clean "value" and is called *strictly determined* (a saddle point in pure strategies exists); when they differ, the gap is exactly the advantage of being found out, and no pure-strategy value exists.

**Diagnostic facts about pure play.** It is easy to write matrices where the gap is strict. In "Matching Pennies" ($2\times 2$, $+1$ on the diagonal, $-1$ off) the maximin is $-1$ and the minimax is $+1$. In "Stone, Paper, Scissors" (a $3\times3$ skew-symmetric "Morra") the same gap $-1$ vs $+1$ appears. In these games, no single way of playing is better than another against an informed opponent; *everything* turns on guessing the opponent's intention, which the rules forbid you to observe. Pure strategies therefore leave a genuine hole: the secure value sits strictly below what the opponent can impose, and there is no determinate value to call "the answer."

**Émile Borel's groundwork (1921, and 1924–1927).** Borel, approaching games with a probabilist's instinct for generality, supplied three load-bearing ideas in a four-page note to the Académie des Sciences (19 December 1921). First, *strategic normalization*: a "method of play" or "code that determines for every possible circumstance what the person should do" can be treated as a single choice, so a multi-stage game reduces to one simultaneous choice among complete plans. Second, *randomized strategies*: against an opponent who profits from predicting you, a rational player should not be perfectly predictable, so a player may choose not a plan but a *probability distribution over plans*. Third, *expected-payoff maximization*: Borel justified maximizing expected payoff directly, because he took each payoff to be a probability of winning, so the expectation of a randomized strategy is just the player's ex-ante probability of winning — no separate theory of utility needed. With payoffs $\alpha_{ik}=p_{ik}-\tfrac12$ (win-probability minus a half), a symmetric game has $\alpha_{ik}+\alpha_{ki}=0$: the matrix is skew-symmetric, the value (if any) is $0$.

Borel proved that a "fair" randomized strategy exists for symmetric games with $3$ undominated strategies (the rock-paper-scissors case: a unique full-support mix giving expected payoff $0$ against everything). But in the same breath he conjectured — without proof, and repeatedly with a favoritism toward this guess — that for larger games ($n>3$) such a solution generally *fails to exist*. His argument sought a *full-support* mix giving payoff *identically zero* against everything — a vector in the kernel of the skew-symmetric matrix; a skew-symmetric matrix of even order is generically nonsingular, so it has no such kernel vector — making it "obvious" to Borel that for $n=4$ a constant-zero solution exists only for "very particular values" of the payoffs. He illustrated with a discrete Blotto game (allocate an integer budget across three battlefields, win the majority) and claimed it unsolvable. The open conjecture, stated this confidently, framed the central question — *does the maximin always equal the minimax once mixing is allowed?* — as a live and doubtful one.

**Convexity and separation (the available mathematics).** A closed convex set and a point outside it can be separated by a hyperplane: project the point onto the set, and the perpendicular at the foot of the projection is a supporting hyperplane with the whole set on one side. From this follow the "theorems of the alternative" for systems of linear inequalities (the Farkas/Gordan family): infeasibility of one linear sign system is certified by a nonnegative multiplier for the opposite system; in the strict versions exactly one side can hold, while weak versions may meet at equality. These are the standard tools of finite-dimensional convexity available at the time.

**Brouwer's fixed-point theorem.** A continuous map of a compact convex set into itself has a fixed point. This topological fact was on the table as a way to assert the *existence* of a self-consistent configuration without constructing it — the natural hammer for a circular "each player optimal against the other" condition.

## Baselines

These are the prior positions a determinate-value theory must measure itself against.

- **Pure-strategy / strictly-determined analysis.** Define $v_{\text{low}}=\max_x\min_y g(x,y)$ and $v_{\text{up}}=\min_y\max_x g(x,y)$; always $v_{\text{low}}\le v_{\text{up}}$. *Core idea:* a saddle point $(x_0,y_0)$ with $g(x,y_0)\le g(x_0,y_0)\le g(x_0,y)$ pins the value at $g(x_0,y_0)$ and makes both players' optimal moves deterministic. *Gap:* such a saddle exists only for special matrices; Matching Pennies and Stone-Paper-Scissors have $v_{\text{low}}<v_{\text{up}}$, so this analysis simply *fails to assign a value* to the generic game — it solves only the strictly-determined subclass.

- **Borel's randomized-strategy program (1921–1927).** Replace a plan by a distribution over plans; evaluate by expected payoff; for symmetric games seek a mix guaranteeing the fair value $0$. *Core idea:* unpredictability as a defense, expectation as the criterion, skew-symmetry as the reduced form. *Gap:* Borel proved existence only up to $n=3$ and conjectured non-existence beyond — so the program established the right *language* (mixed strategies, expectation) but left the central existence question unproven, his conjecture resting on a kernel/constant-zero strategy that fails generically in even order.

- **Existence-by-fixed-point, in the abstract.** A continuous self-map of a compact convex set has a fixed point (Brouwer). *Core idea:* the equilibrium condition "each player's mix is a best response to the other's" is a fixed-point condition; if one can encode it as a continuous self-map of the product of simplices, existence follows. *Gap:* this is a hammer in search of the right map; on its own it asserts nothing about a game until a continuous "improvement" map is constructed and shown to have fixed points that coincide with equilibria.

## Evaluation settings

The natural yardsticks are not benchmark datasets but specific small games and matrix classes whose correct answer is independently known or computable. **Matching Pennies** ($2\times2$, $\pm1$): the test that pure strategies have no value but a $50{:}50$ mix gives value $0$. **Stone, Paper, Scissors** ($3\times3$ skew-symmetric): the symmetric/fair test, value $0$, optimal mix uniform $(\tfrac13,\tfrac13,\tfrac13)$. **General skew-symmetric matrices** (symmetric games): if a mixed value exists, symmetry forces it to be $0$; Borel's testing ground is whether some mixed strategy can guarantee the fair side of that value in cases such as the integer-budget Blotto game on three fronts. **Arbitrary real $m\times n$ matrices**: the general object, on which the claim "$\max\min=\min\max$ once mixing is allowed" must hold or fail. The relevant criteria are: existence of a value $v$; existence of optimal mixed strategies $\xi^\*,\eta^\*$; the guarantee that player 1 secures $\ge v$ and player 2 holds him to $\le v$ irrespective of the opponent; and the structure of the optimal mixes (which pure strategies they use).

## Code framework

A minimal numerical harness takes a payoff matrix, checks candidate probability vectors, evaluates the bilinear payoff, and exposes the two one-sided optimization problems. The missing slot is a single value and a saddle pair, if the one-sided optima coincide.

```python
import numpy as np

# Probability simplex Δ_k = { p ∈ R^k : p ≥ 0, Σ p = 1 } — the space of mixed strategies.
def is_distribution(p, tol=1e-9):
    return np.all(p >= -tol) and abs(p.sum() - 1.0) < tol

# Expected payoff of a mixed-strategy pair under matrix A (1 maximizes, 2 minimizes).
def payoff(A, xi, eta):
    return float(xi @ A @ eta)              # ξᵀ A η  — bilinear in (ξ, η)

# What player 1 can guarantee if forced to commit first (inner opponent is adversarial).
def secure_value_row(A, xi):
    return float(np.min(xi @ A))            # min_j (ξᵀA)_j  — worst column response to ξ

# What player 2 can hold player 1 to.
def secure_value_col(A, eta):
    return float(np.max(A @ eta))           # max_i (Aη)_i  — worst row response to η

# The two one-sided optima, written as explicit optimization slots.
def maximin(A):
    # TODO: max over ξ ∈ Δ_m of secure_value_row(A, ξ); returns (v_low, ξ*).
    pass

def minimax(A):
    # TODO: min over η ∈ Δ_n of secure_value_col(A, η); returns (v_up, η*).
    pass

# The missing object: a single value and a saddle pair, if they exist.
def solve_zero_sum_game(A):
    # TODO: return v and (ξ*, η*) with secure_value_row(A, ξ*) = v = secure_value_col(A, η*),
    #       i.e. a saddle point of ξᵀAη — IF such a thing always exists.
    pass

# Sanity fixtures from direct hand calculation.
MATCHING_PENNIES = np.array([[ 1., -1.],
                             [-1.,  1.]])   # pure: maximin=-1, minimax=+1 (a gap)
RPS = np.array([[ 0., -1.,  1.],
                [ 1.,  0., -1.],
                [-1.,  1.,  0.]])           # skew-symmetric; fair value must be 0
```
