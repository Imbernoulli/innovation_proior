Price the outcomes before designing the rule: in the deployment this variant
models, an incorrect prediction that slips through acceptance costs roughly
five times what a human review costs, while a deferral is merely the price of
one review. Under that ledger, the reported error columns — accepted-set
error overall and for the hardest subgroup — are the dominant cost terms,
and the policy's job is to drive them toward the floor attainable at the
operating budget.

Nothing else in the exercise moves: cost asymmetry has to enter through the
choice of acceptance rule alone. A budget-only quantile cutoff is
cost-blind: it will happily accept a case whose expected mis-acceptance cost
exceeds the review it saved, simply because that case cleared a rank
threshold. This variant asks for an expected-cost view instead — acceptance
justified case by case against the review alternative, using whatever
confidence evidence the calibration split supports — with the coverage
figure respected rather than gamed, and with the acceptance score still
functioning as a correctness ranking, since a cost gate built on a bad
ranking gates the wrong cases.

Worst-subgroup accepted error deserves particular attention: concentrated
expensive mistakes inside one population are the canonical way asymmetric
costs go wrong silently. The deferral-rate spread is reported and should not
blow out, but in this ledger it is a secondary term.

Defend the result as a cost argument: exhibit where the budget-only cutoff
accepts negative-expected-value cases, show the mechanism that screens them
out, and account for what that screen did to coverage and ranking quality on
each of the three datasets.
