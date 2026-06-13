## Research question

There is a complete, satisfying mathematical theory of exactly one kind of strategic
conflict: the two-person, zero-sum game. For everything else — three or more players, or
two players whose interests are not perfectly opposed — there is no general notion of
"the rational outcome" and no theorem guaranteeing one exists.

Concretely, the situation to be modeled is this. A fixed, finite set of players each chooses
an action from a finite menu, simultaneously and independently, with no possibility of
binding agreements, side-payments, communication, or enforced coalitions. Each player has a
payoff that depends on *everyone's* choices, and these payoffs need not sum to a constant —
one player's gain need not be another's loss. The question is: what is a principled notion
of a "solution" for such a game, and can one prove that a solution always exists?

A solution concept here must (i) make sense without any assumption of antagonism — it cannot
rely on a single "value" of the game, because with non-opposed interests there is no single
number that summarizes the outcome; (ii) reduce to the known, correct answer (the saddle
point / value) in the special case of two-person zero-sum games, so it is a genuine
generalization and not a different theory; and (iii) be provably non-empty — for *every*
such finite game, not just nice ones. A definition that exists only for some games, or that
requires solving the game to check, would not do.

## Background

The field rests on the framework of von Neumann and Morgenstern. A game in *normal form* is
specified by: a finite list of players; for each player a finite set of *pure strategies*;
and for each player a *payoff function* assigning a real number to every combination
(an n-tuple) of pure strategies. To allow randomization, a *mixed strategy* for a player is a
probability distribution over that player's pure strategies — a vector of non-negative
weights summing to one. Geometrically a mixed strategy is a point of a *simplex* whose
vertices are the pure strategies; the simplex is a compact, convex subset of a real vector
space, so mixed strategies can be averaged. A profile of mixed strategies, one per player,
is a point of the *product of the simplices*, itself a compact convex polytope. Payoffs
extend to mixed strategies as expectations, and this extension is *multilinear*: it is a
linear (affine) function of each single player's own mixed strategy when the others are held
fixed. Linearity in one's own mixture is a basic structural fact of the whole setting.

The one general existence result available is the *minimax theorem* (von Neumann, 1928, "Zur
Theorie der Gesellschaftsspiele"). For a two-person zero-sum game with payoff matrix `A`,
where the row player mixes with `x` and the column player with `y`,

    max_x min_y  x^T A y  =  min_y max_x  x^T A y.

The common value `v` is the *value* of the game; the optimal pair `(x*, y*)` is a *saddle
point*. The trivial direction is `max-min <= min-max`; the content of the theorem is the
reverse inequality, and it holds precisely because passing to *mixed* strategies convexifies
the problem (Borel had introduced mixed strategies for duels around 1920 but did not prove
the general equality). Two features make this work and also fence it in: there are exactly
two players, and the payoffs are strictly opposed. The opposition is what makes a single
number `v` meaningful — "what is good for me is exactly bad for you," so one value can
summarize the game and "play well" means "guarantee yourself `v`." Remove either feature and
the notion of a value evaporates: with three players, or with two players whose payoffs do
not cancel, there is no number every player is simultaneously pushing for or against.

For more than two players, the same framework offers only a *cooperative* theory, also from
von Neumann and Morgenstern: it analyzes which *coalitions* form, via characteristic
functions and imputations, assuming players can communicate, collude, and enforce
agreements. This is a different modeling assumption — binding coalitions — and it does not
address the independent-play, no-communication situation at all.

There is one older, narrower precedent for the kind of rest-point one wants. Cournot (1838),
analyzing two firms (a duopoly) each choosing output to maximize its own profit given the
other's output, identified the configuration where neither firm wishes to change its output
unilaterally — a mutual-best-response rest point, non-cooperative and not zero-sum. It is the
prototype of "stable against unilateral deviation," but it lives inside one specific economic
model and carries no general definition and no existence theory.

The other load-bearing pieces of background are topological. *Brouwer's fixed point theorem*:
every continuous map of a compact convex set (a ball, a simplex, a polytope) into itself has
a point it leaves fixed. And *Kakutani's theorem* (1941, Duke Math. J. 8, 457-459), which
generalizes Brouwer from single-valued maps to *set-valued* ones: if `S` is a non-empty
compact convex set in Euclidean space and `phi` assigns to each point of `S` a non-empty,
convex subset of `S`, and `phi` has *closed graph* (if `x_n -> x`, `y_n -> y`, and
`y_n in phi(x_n)`, then `y in phi(x)`), then there is a point `x` with `x in phi(x)`.
Kakutani built this to handle correspondences in which the image at a point can be a whole
set, rather than only single-valued maps.

## Baselines

**Minimax / two-person zero-sum theory (von Neumann 1928; von Neumann & Morgenstern 1944).**
Core idea: over mixed strategies, the maximin and minimax of a two-person zero-sum game
coincide; the game has a value `v` and a saddle point, and "good strategies" are those that
guarantee `v`. The math is the minimax equality above, proved by a separating-hyperplane /
convexity argument on the simplices. Limitation: it is structurally confined to *two* players
and to *zero-sum* payoffs. The proof and even the *statement* lean on a single scalar value,
which only exists under strict opposition. It says nothing about three players, nothing about
non-constant-sum payoffs, and offers no solution concept once a single value is unavailable.

**Cooperative n-person theory (von Neumann & Morgenstern 1944).** Core idea: model an
n-person game by what each *coalition* of players can guarantee itself (the characteristic
function), and study stable sets of imputations — divisions of the collective payoff. This
does cover many players, but by assuming the opposite of what is wanted: that players
communicate and form *binding* coalitions. Limitation: it presumes enforceable collaboration
and side-payments; it has nothing to say about players acting independently with no
agreements, which is the case in question.

**Cournot duopoly rest point (Cournot 1838).** Core idea: the outcome of an oligopoly is the
output profile at which no firm gains by unilaterally changing output — a fixed point of the
firms' reaction functions. This is exactly the right *shape* of solution (mutual best
response, no antagonism assumed). Limitation: it is tied to a particular continuous economic
model with smooth reaction curves; there is no general definition for arbitrary finite games,
and crucially no proof that such a rest point must exist in general — for general games the
reaction "functions" are set-valued and there is no guarantee of a crossing.

## Evaluation settings

The natural yardsticks here are not benchmark datasets but mathematical tests any candidate
solution concept must pass. (a) *Reduction*: on a two-person zero-sum game the candidate
solutions must coincide with the von Neumann saddle points / pairs of optimal strategies —
otherwise it is a different, competing notion rather than a generalization. (b) *Universality
of existence*: the existence claim is to be checked against *every* finite game, including
deliberately awkward small ones — two-by-two games with no pure-strategy rest point (so that
mixing is forced), games with several rest points, games whose only rest points are in the
interior of the simplices. (c) *Worked illustrations*: small explicit games given by payoff
tables, and a simplified multi-player parlor game (e.g. a stripped-down three-player poker
with a few "behavior parameters"), where one can actually locate the rest point(s) by hand
and confirm the concept produces sensible, computable answers. The protocol is derivation and
hand-computation on these cases, not measurement.

## Code framework

The computational scaffold is minimal: represent a finite game as payoff tables, normalize
a weight vector onto a simplex, and compute each player's expected payoff under a mixed
profile. What remains open is the solution concept itself and the proof that a solution
always exists.

```python
import numpy as np

# --- normal-form primitives ---

def expected_payoffs(payoff_tensors, profile):
    """payoff_tensors[i] : array shape (m_1,...,m_n) = player i's payoff over pure profiles.
       profile : list of mixed strategies (each a prob vector on that player's simplex).
       Returns the vector of expected payoffs, one per player (multilinear extension)."""
    n = len(payoff_tensors)
    payoffs = np.empty(n)
    for i in range(n):
        t = payoff_tensors[i]
        for ax, s in enumerate(profile):          # contract each player's axis with its mix
            t = np.tensordot(t, s, axes=([0], [0]))
        payoffs[i] = t                            # scalar = expected payoff to player i
    return payoffs

def project_to_simplex(weights):
    """Renormalize non-negative weights to a probability vector on the simplex."""
    w = np.clip(weights, 0.0, None)
    return w / w.sum()

# --- open slot ---

def a_solution_exists(payoff_tensors):
    """The solution concept for an independent-play, possibly non-zero-sum, n-player game,
       and the proof that one always exists in mixed strategies.
       # TODO: define what makes a `profile` a 'solution' (without assuming zero-sum /
       #       antagonism, and reducing to the saddle point in the two-person zero-sum
       #       case), and argue that every finite game has one."""
    pass
```
