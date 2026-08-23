A depth-3 tree partitions its input space into at most eight axis-aligned
cells. Nothing a weighting scheme does can make one such tree strong, and
this variant treats that ceiling as the organizing constraint: every point
of held-out performance must be manufactured by how the two hundred rounds
relate to each other, never by the quality of any single round.

The object of study is complementarity. Two hundred weak trees fitted to
similar targets under similar distributions produce two hundred nearly
interchangeable opinions, and aggregating interchangeable opinions buys
almost nothing. The mechanism you design should make successive learners
disagree usefully: steer each new tree -- through the pseudo-targets it is
given and the sample distribution it is fitted under -- toward structure
the existing ensemble has not yet captured, and away from cells it already
handles. The scaffold keeps a per-sample record of how much attention the
run has spent on each point and deliberately leaves it unconsumed; turning
that record, or a better measure of redundancy between rounds, into a
steering signal is the intended contribution.

Three constraints define the variant. The base learner's capacity is fixed
by the pipeline and is off-limits as a lever. Whatever notion of coverage
or novelty you introduce must apply unchanged to the binary classification
set and to both regression sets. And the steering must remain profitable
under the fixed vote/accumulation rule, which is the only path from
individual trees to the reported test numbers. The claim to defend: the
finished ensemble is measurably better than what the unchanged quality of
its members would predict, because the sequence was engineered rather than
repeated.
