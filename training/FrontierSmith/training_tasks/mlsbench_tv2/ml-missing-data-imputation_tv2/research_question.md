Every baseline in this environment is derived under the premise that holes
fall where chance puts them: if the mask is independent of the data,
statistics computed from observed rows transfer unbiased to the missing
ones. Drop that premise and the derivations quietly break -- when a
column's larger values are likelier to vanish, its observed mean
underestimates the truth, complete-case regressions inherit selection
bias, and iterative refinement polishes a systematically shifted
completion.

The ask: an imputer that nowhere in its derivation leans on the
random-mask premise. Treat the missingness pattern itself as data: which
columns' masks co-vary with which columns' observed values, whether
many-hole rows look systematically different from nearly complete rows,
and what those dependencies say about where the observed sample is
unrepresentative. The scaffold measures a mask-to-value dependence matrix
at fit time and anchors its fill on per-column medians -- a center chosen
because it degrades more gracefully than the mean under selective
observation -- but the dependence evidence is currently a diagnostic only;
promoting it into an explicit correction of the fill is the intended work.

The ground rules are these: any estimate feeding the completion must be
defensible without invoking randomness of the mask, so robust centers,
mask-conditional modeling, and shift corrections estimated from the
measured dependence all qualify. And nothing about how the work is judged
moves -- masked-entry reconstruction and the downstream model built on the
completed matrix remain the yardsticks -- so mechanism-agnosticism has to
be won without giving either of them away. The position to defend: this
completion would survive an adversary who chooses where the holes go.
