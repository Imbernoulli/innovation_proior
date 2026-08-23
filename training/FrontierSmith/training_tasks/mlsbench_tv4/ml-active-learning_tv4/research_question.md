Uncertainty sampling has a structural conflict of interest: the points it
prizes most — those sitting exactly on the decision boundary — are the points
whose labels are least reliable in any realistic annotation process, and
often least useful even when correct. Maximum-ambiguity examples include
genuine boundary cases, but also feature-space outliers, near-duplicates of
contradictory examples, and instances whose class is intrinsically contested;
a splice junction of debatable category or a hand-printed letter halfway
between two glyphs teaches a small network little and can actively distort
it. A strategy built as if the oracle were noisy therefore buys differently:
it wants examples that are informative and corroborable — points whose
neighborhood, cluster, or committee context would agree with the label the
oracle returns.

Concretely, this variant asks for an acquisition rule that treats extreme
ambiguity as a warning sign rather than a prize. The rule should measure
informativeness, but discount or exclude the pathological tail of that
ranking — the candidates that look least like anything already labeled, or on
which repeated stochastic predictions disagree wildly — and prefer the
reliable middle: examples the current model finds unresolved yet which sit in
dense, internally consistent regions of the pool. Which reliability signal to
use (neighborhood label agreement, embedding density, dropout-committee
variance, distance to the labeled set) and how hard to cut the tail are the
open design choices, with one rule required to serve all three datasets.

Nothing in the pipeline is altered and the oracle here is in fact clean, so
the burden of proof is sharp: the tail-discounting rule must match or beat
plain uncertainty sampling on final accuracy and on learning-curve area as
scored — demonstrating that the ambiguity tail it refuses to buy was not
carrying the value — while carrying its noise insurance for free.
