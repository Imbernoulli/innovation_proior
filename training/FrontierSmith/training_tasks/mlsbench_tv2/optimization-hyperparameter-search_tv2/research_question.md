Fifty model fits on the largest benchmark, forty on the others — that
is the entire experimental career a strategy gets here, and this
variant removes the usual way of stretching it: every trial is taken
at full fidelity. No cheap screens, no partial training runs, no
fractional charges — each suggestion costs a whole unit and returns
one trustworthy number. The regime under study is HPO as a handful of
expensive, irreversible experiments.

With so few observations, the design question becomes information per
trial. A proposal is justified only by what the previous full-fidelity
scores imply about where improvement is still plausible; spending
three of forty trials in a region history already condemned is nearly
a tenth of the career gone. The strategy must decide, trial by trial,
between narrowing around the incumbent and testing a genuinely
different part of the space, knowing that either choice consumes an
irreplaceable unit. One policy must cover the mixed parameter types —
log-scaled continuous ranges, small integer grids, a categorical
kernel — with nothing tuned per benchmark.

best_val_score is the primary target here: with every observation
trustworthy, there is no excuse for a final incumbent that a
screening-based strategy would also have found. convergence_auc,
computed against spent cost, rewards ordering the few trials so the
good ones come early rather than being stumbled on at the end.

The claim to defend: at these budget sizes, in three to six
dimensions, deliberate sequential design at full fidelity matches or
beats fidelity juggling — the sophistication of multi-fidelity
scheduling was never the binding constraint here; trial-selection
discipline was.
