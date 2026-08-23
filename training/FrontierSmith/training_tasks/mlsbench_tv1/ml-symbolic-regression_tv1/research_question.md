A GP run can finish with a near-perfect fit on its training samples and still
be scored a failure: only predictions on withheld test inputs count, and a
bloated tree that memorizes the idiosyncrasies of a few hundred sampled points
has learned the sample, not the function. Because the task score is a
geometric mean over several hidden targets and runs repeat over multiple
seeds, a search that recovers two targets brilliantly but stalls on the third
— or succeeds only on lucky seeds — is scored close to its worst run. The
re-aimed question: engineer the evolutionary dynamics so that the expression
leaving each run is one whose training fit *transfers*, and so that no
target-seed combination is left behind.

Two failure regimes are the objects of study. First, deceptive fit: large
trees exploiting the protected operators (division near a vanishing
denominator, clipped exponentials, absolute-value logs) can interpolate the
training points while behaving pathologically between and beyond them.
Training-set error alone cannot see this, so the search needs internal signals
that correlate with transfer — complexity pressure, agreement on held-out
subsamples of the training data it already owns, behavioral diversity —
without being told the test inputs' responses. Second, stagnation: population
size, generation count, and depth caps are fixed by the driver, so a run that
converges prematurely has no budget left to recover; detecting stalls and
reacting (restarts seeded from the elite, adaptive operator rates, enforced
novelty) is worth more here than sharpening the average case.

Everything must live inside the provided five-function skeleton, over the
fixed operator set and tree representation, and the run must still end in a
single executable expression. The claim to defend is about reliability:
name the mechanism that kept the weakest run's held-out fit from collapsing,
and show it did not blunt the search on targets that were already easy.
