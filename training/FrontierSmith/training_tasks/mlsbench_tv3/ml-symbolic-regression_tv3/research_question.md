Treat every training sample as a witness that may be lying a little. Finite
sampling, clipped evaluation ranges, and the protected variants of division,
logarithm, and exponential all inject idiosyncrasies into the (X, y) pairs
the driver hands over, and a search that minimizes raw mean-squared error
will happily spend half of its tree explaining those idiosyncrasies. The
variant studied here is structure recovery under that suspicion: the search
should behave, throughout, as if a modest fraction of its residuals were not
signal — and still converge on the algebraic skeleton of the target.

Concretely, the objects of study are the robust ingredients themselves:
fitness that discounts the largest residuals instead of being dominated by
them (trimmed or otherwise robust losses); variation operators that make
measured local edits — exchanging an operator for a same-arity peer, nudging
a coefficient — rather than wholesale subtree churn that lets lucky spikes
chase stray points; and any consensus signal derived from re-scoring on
subsamples of the data the search already owns. The failure mode to
engineer away is the low-amplitude additive term that appears in a final
expression purely to absorb residual scatter: such terms do not transfer,
and the withheld test inputs are the only arbiter of whether they were
fitted.

Everything runs in the unchanged harness — same driver schedule, same
population and generation counts, same held-out R2 — so robustness cannot
be bought with extra evaluations. The claim to defend is comparative: on
targets where a raw-MSE search decorates its answer with junk terms, the
robust search returns a cleaner expression whose test-set fit is at least
as good; and on targets that were never noisy to begin with, the robustness
machinery costs essentially nothing.
