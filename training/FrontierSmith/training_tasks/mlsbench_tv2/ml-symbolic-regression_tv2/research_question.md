Most GP runs end with an expression nobody would write down: dozens of nested
protected operators tuned to the sample at hand. This variant inverts the
usual priority order. Before fitness is even consulted, fix a hard ceiling on
expression size — roughly two dozen tree nodes — and treat that ceiling the
way a hardware team treats a die-area limit: candidates above it are not
"worse", they are ineligible. All optimization pressure then works inside the
budget: among admissible trees, drive the training error down as far as the
fixed operator set allows, and let the compact structure carry the fit to the
withheld inputs on its own.

The interesting engineering is in keeping a whole population admissible
without starving it of material to search with. Crossover and mutation as
usually written are inflationary; under a hard cap they need shrink-biased
counterparts (subtree deletion, hoisting a child over its parent, grafts
accepted only when the offspring stays under the ceiling), and selection
needs a story for ties — when two trees fit equally well, the smaller one
must win, or the cap becomes decorative. Note what is deliberately absent:
no soft size penalty folded into the loss, no annealed complexity weight.
The budget is a constraint, not a regularizer, and the scoring pipeline is
untouched — held-out R2 as always, with the run's feedback line reporting
the size and printed form of the final expression, so budget adherence is
visible after the fact.

The claim to defend: over the hidden benchmark suite, a search confined to
small expressions matches or beats an unconstrained one on held-out R2, and
the final reported expression stays within the declared node ceiling on
every run. Defending it means showing the cap forced structure discovery
rather than merely truncating it.
