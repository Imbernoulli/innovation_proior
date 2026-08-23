At a twenty percent missing rate, damage is not an edge case but the
ambient condition: in a thirty-column table the typical row has several
holes at once, and an unlucky row loses most of its content. That regime,
not the average case, is the subject here. The failure mode of interest is
compounding: any method that predicts one column from the others must, on
a multi-hole row, stand its predictions on values that were themselves
imputed, so first-pass errors become second-pass inputs, and the
completion drifts furthest exactly on the rows that needed the most help.

The brief, then: take charge of the propagation instead of hoping it
stays benign. Two levers are set up in the scaffold. Ordering: columns are completed from
least-missing to most-missing, so the most reliable fills are in place
before the hardest columns consult them. Fallback: rows whose missing
fraction crosses a threshold are routed to an explicitly simple
degraded-row policy instead of a long inference chain, on the view that
chained inference over scant evidence compounds rather than helps. Both
levers currently wrap a plain per-column mean fill; what you must supply
is a chained scheme that decides, cell by cell, how much inference the
surrounding evidence can carry -- and stops before compounding outruns
accuracy.

Quality here must hold up on the heavily damaged rows specifically; an
average propped up by the lightly damaged ones earns no credit. A method
that shines on nearly complete rows while scattering the multi-hole ones
will be caught by both reported numbers. Argue from the structure of
your method why its error stays controlled as the number of simultaneous
holes in a row grows.
