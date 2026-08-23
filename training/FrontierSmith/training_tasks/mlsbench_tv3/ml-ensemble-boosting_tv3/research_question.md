On each evaluation dataset most samples are easy: a few rounds fit them,
and further capacity spent on them is waste. The reported numbers are
decided by a minority -- boundary cases in the classification task,
high-residual points in the two regression tasks -- and this variant asks
for a boosting strategy built explicitly around finding and serving that
minority.

The central design question is what earns emphasis. A sample missed once
is not necessarily hard; a sample missed round after round is. Difficulty
evidence should therefore be accumulated across rounds -- persistent
misclassification, persistently large residuals -- and converted into
concentrated sample weight on a schedule you control, rather than emerging
as the accidental by-product of an exponential update. The failure mode
matters as much as the goal: unbounded emphasis lets a handful of atypical
points capture the whole distribution, which is exactly how classical
reweighting destroys held-out performance. A cap, budget, or saturation on
any single sample's weight share is a required element of the design, not
an optional refinement.

One emphasis machine must serve both task families. The difficulty
statistic itself may be task-appropriate -- margin-like for classification,
residual-magnitude-like for regression -- but the schedule mapping
difficulty to weight, and the ceiling bounding it, are shared. Scoring
stays as it was; what needs arguing is that the improvement traces to the
hard minority being served well without the easy majority coming
unsolved.
