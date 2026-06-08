# Context: taming multistage decision processes

## Research question

A wide and growing class of problems share one shape: a system evolves through a sequence of stages, and at each stage we must make a decision that changes the system's state and earns some return, with the goal of maximizing a function of the whole trajectory. Inventory ordering month after month; allocating a budget across competing activities and then re-allocating what remains; deciding when to replace aging machinery; scheduling patients or jobs; steering a continuous physical system optimally over a time interval; two players spending money to ruin each other. In every case the unknown is not a single number but a whole *sequence* of decisions, and the decisions interact across time — what is best now depends on what will be best later, which depends on where now leaves us.

The precise problem: given a system whose state at any time is a vector of state variables, a set of allowable decisions at each stage (each decision being a transformation of the state), and a return that accumulates over the stages, find the decisions that maximize the total (or, when the dynamics are random, the expected) return. A genuine solution has to do three things the obvious approach cannot: it must scale to many stages without the work exploding combinatorially in the number of stages; it must handle *stochastic* transitions, where committing in advance to a fixed sequence of actions is not even well defined; and it must accommodate constraints (an allocation can't exceed what's on hand) that break the smooth machinery of the classical continuous theory.

## Background

The prevailing way to attack such a problem is direct enumeration: regard a *policy* as a complete sequence of decisions, one per stage; for each feasible policy compute the resulting return; then maximize over the set of all feasible policies. This is correct and conceptually simple. Its cost is the difficulty. For a process of $N$ stages with even a moderate number $k$ of choices at each stage, the number of feasible sequences is on the order of $k^{N}$, and the maximization is over a space whose dimension grows with the number of stages. For continuous processes — where a decision must be made at every instant of a time interval — the policy is a *function* of time and the maximization is over a function space. The dimension of the resulting optimization is uncomfortably high; the price of this excessive dimensionality can make even a fast computing machine cringe.

There is a deeper failure for random processes. When a decision determines only a *distribution* over next states rather than a single next state, it is meaningless to fix a sequence of decisions in advance: you cannot decide your stage-3 action before you know what stage-2 actually produced. The enumerative, fixed-sequence view is "virtually impossible" in the stochastic case. This points, before any method exists, at a distinction the classical view blurs: an *open-loop* plan (a fixed sequence chosen at time zero) versus a *closed-loop* plan (a rule that picks each action as a function of the state actually observed). Under uncertainty the closed-loop class is strictly richer and is the only one that even makes sense.

Several lines of prior work had, in their own corners, hit upon recursing backwards through the stages:

- In statistical decision theory, **Wald (1947,** *Sequential Analysis*) generalized the gambler's-ruin problem and introduced the sequential probability ratio test, deciding when to stop sampling so as to minimize the expected number of observations. The backward-recursive structure is present but partly implicit.
- **Arrow, Blackwell, and Girshick (1949)** made it explicit: for the optimal-stopping statistical-decision problem they characterized the best rule "by induction backwards," approximating the optimum among all procedures using no more than $N$ observations and then letting $N$ grow.
- **von Neumann and Morgenstern (1944)**, in game theory, solved extensive-form games by starting at the last move and working back through the tree — what we would now call finding subgame-perfect equilibria.
- **Massé (1944)** on reservoir management and **Arrow, Harris, and Marschak (1951)** on optimal inventory each solved a specific multistage optimization by backward recursion.

The common thread — start at the end, work backwards, and at each stage make the locally best continuation — is visible across all of these, but each is tied to its own problem (when to stop sampling, this game tree, this reservoir, this inventory). None abstracts the shared structure into a single object and a single equation that all of them are instances of.

For the *continuous* deterministic case the relevant body of theory is the **calculus of variations**. To maximize $\int_0^T F(x,y)\,dt$ subject to a dynamics $dx/dt = G(x,y)$, the classical technique treats the optimizing trajectory as a point in function space and characterizes it by variational conditions, chiefly the **Euler equation**; the associated **Hamilton–Jacobi** theory expresses the optimal value as the solution of a first-order partial differential equation. This is powerful but limited: it describes the extremal as a function of *time* (an open-loop description); it relies on free variation, so inequality constraints of the form $0\le y\le x$ — exactly the constraints that pervade allocation and inventory problems — break it, because the extremum is then attained on a boundary where equalities become inequalities; and it has no purchase at all on stochastic transitions.

So the field state is: a clean enumerative formulation that does not scale and cannot handle randomness; a scattering of problem-specific backward recursions that nobody has unified; and a continuous theory (Euler / Hamilton–Jacobi) that is open-loop and deterministic-only. The observed, before-the-fact pain points — exponential blow-up of policy enumeration in the number of stages, the meaninglessness of fixed action sequences under uncertainty, and the breakdown of the variational calculus under inequality constraints — are the facts a method has to confront.

## Baselines

- **Exhaustive policy enumeration.** Enumerate all $k^{N}$ feasible decision sequences, evaluate each, take the max. Core idea: brute force over policy space. Math: $\max_{(T_1,\dots,T_N)} R\big(T_N(\cdots T_1(p)\cdots)\big)$. Gap: cost grows like $k^{N}$ (exponential in the number of stages); for continuous processes the maximization is over a function space; and for stochastic processes a fixed sequence is undefined, so the method does not even apply.

- **Backward induction on a fixed problem (Wald 1947; Arrow–Blackwell–Girshick 1949; von Neumann–Morgenstern 1944; Massé 1944; Arrow–Harris–Marschak 1951).** Core idea: solve the last stage, then the second-to-last given the worth of the last, and so on. Math (stopping example): pick the best continuation at each node, working from the final stage back. Gap: each instance is bound to its own problem — when to stop a sequential test, the moves of one game tree, one reservoir's release schedule, one inventory rule. There is no abstract state, no general state-based summary, and no single functional equation; the technique is rediscovered case by case rather than recognized as one reusable idea.

- **Calculus of variations / Euler equation / Hamilton–Jacobi theory.** Core idea: for continuous deterministic optimization, characterize the optimal trajectory by variational stationarity. Math: extremals satisfy the Euler equation $\frac{d}{dt}\frac{\partial F}{\partial \dot y} = \frac{\partial F}{\partial y}$; the optimal value solves a first-order Hamilton–Jacobi PDE. Gap: the description is open-loop (a function of time, not of the current state); inequality constraints with non-free variation are not handled; stochastic dynamics are out of scope entirely.

## Evaluation settings

The natural testbeds are the very problems that motivate the question, each with its own state, decisions, and return:

- **Allocation / investment.** A quantity $x>0$ is split into parts; each part yields a return and is then shrunk by a known factor; the process repeats with the new total. Metric: total return accumulated over the process.
- **Optimal inventory.** Stock is ordered each period against random demand; metric: expected discounted ordering-plus-holding-plus-shortage cost.
- **Equipment replacement / reservoir management / job scheduling.** Multistage operational problems with a per-stage cost or yield; metric: total or expected cost over the horizon.
- **Stochastic "gold-mining."** Two mines and one fragile machine; each use either succeeds (mining a fixed fraction) or destroys the machine; metric: expected gold mined before the machine breaks.
- **Continuous control / calculus of variations.** Maximize $\int_0^T F(x,y)\,dt$ subject to $dx/dt=G(x,y)$, $x(0)=c$, possibly with constraints $0\le y\le x$; metric: the integral return.
- **Games of survival.** Two players with bankrolls $x,y$ play a repeated zero-sum game to ruin; metric: probability one ruins the other.

The yardstick for any successful approach is whether it recovers the right optimum and an optimal rule on each of these, including the cases (stochastic transitions, inequality constraints) where the enumerative and variational baselines fail outright.

## Code framework

A bare scaffold for a finite multistage decision process, written purely in terms already present in the problem: a state, a per-stage set of allowable decisions, a transition, a per-stage return, and the brute-force objective that maximizes over the *entire* decision sequence. The one empty slot is a placeholder for whatever state-based summary and relation could avoid enumerating whole sequences.

```python
from typing import Hashable, Iterable

State = Hashable
Decision = Hashable

# --- the multistage decision process, as it is given to us ---

def decisions(state: State, stage: int) -> Iterable[Decision]:
    """Allowable decisions at this state and stage (the choice set)."""
    ...

def transition(state: State, decision: Decision, stage: int) -> State:
    """Deterministic case: the new state after applying the decision."""
    ...

def reward(state: State, decision: Decision, stage: int) -> float:
    """Return earned at this stage for taking this decision in this state."""
    ...

def terminal_reward(state: State) -> float:
    """Return credited to the final state."""
    ...

# --- the baseline that is on the table: enumerate every policy ---

def total_return(initial: State, policy_sequence) -> float:
    """Score one fixed sequence of decisions (one open-loop policy)."""
    s, total = initial, 0.0
    for stage, d in enumerate(policy_sequence):
        total += reward(s, d, stage)
        s = transition(s, d, stage)
    return total + terminal_reward(s)

def best_by_enumeration(initial: State, horizon: int) -> float:
    """Maximize over ALL feasible decision sequences. Correct but costs ~k**N."""
    # enumerate every sequence (T_1, ..., T_N), score each, take the max
    ...  # exponential in the number of stages; undefined under random transitions

# --- the unanswered slot ---

def unresolved_subproblem_summary(state: State, stages_to_go: int) -> float:
    """# TODO: a state-based summary and relation, if one exists, that
    lets us avoid enumerating whole sequences."""
    ...
```
