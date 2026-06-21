## Communication Value Without Decoding

A noisy communication channel has a mathematically defined rate, but the usual operational meaning is tied to coding: with suitable encoding, messages can be recovered reliably below capacity. There are settings where a receiver cannot or does not build such a code. The receiver still observes symbols that change beliefs about an underlying event, and the question is how to attach a concrete value to that partial information without inventing an arbitrary loss table.

## Reinvested Decisions

Consider a sequence of independent events whose outcomes become known only after a decision is made. A receiver may observe a signal before each event and can allocate current resources across possible outcomes. Gains are reinvested, so the next decision is made with a resource level produced by all previous wins and losses. A decision rule must therefore handle products of per-round multipliers, not a single isolated payoff.

## Available Inputs

The pre-decision data are the prior probabilities of the outcomes, the conditional probabilities induced by the received signal, and the posted payoff odds. In a simple two-outcome version, the inputs reduce to a win probability, a loss probability, and the net odds paid on a winning stake. In a multi-outcome version, the decision is a vector of fractions assigned across all possible outcomes after each signal.

## Existing Approaches

A one-period expected-payoff rule chooses the action with the highest immediate expected gain. A fixed-dollar rule stakes a constant amount regardless of the current resource level. An arbitrary utility or cost function can rank outcomes according to numerical values supplied from outside the communication problem.

## Minimal Test Bed

The simplest test is a repeated binary wager from initial wealth `W_0`. A proposed rule chooses a fraction of current wealth before each round using only the probability and odds. The setting can be extended to a multi-outcome signal model in which the decision is a vector of fractions across outcomes after each received signal.
