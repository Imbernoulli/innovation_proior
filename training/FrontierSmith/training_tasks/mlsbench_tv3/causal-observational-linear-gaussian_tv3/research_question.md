False adjacencies are the tax every threshold pays, and in this benchmark the tax compounds:
a spurious edge damages adjacency precision immediately, creates fake orientation
opportunities that then poison the arrow metrics, and feeds SHD twice over when its arrowhead
also lands wrong. This variant treats the number of wrongly reported adjacencies — not any
single test's significance level — as the quantity under control.

The discipline demanded is multiplicity-aware admission. With up to fifty variables the
skeleton stage adjudicates over a thousand candidate pairs, and a per-pair alpha that looks
conservative in isolation still floods a sparse graph with false positives in aggregate.
Candidate edges must therefore pass a procedure whose guarantee is stated at the level of the
whole reported edge set — false discovery rate or a comparably explicit error budget — with
one configuration serving the sparse ten-node graph and the dense fifty-node one alike.
Sparsity is an asset to exploit, not an assumption to hard-code: the method may lean on the
expectation that true graphs are thin, but only through its error-budget arithmetic, never
through a fixed edge count or a constant chosen per scenario.

What must be defended at the end: adjacency precision that holds near its ceiling on every
regime, including the hidden noisy one, while recall is grown only as far as the certified
budget allows; SHD that beats laxer baselines because the errors it counts were never admitted
in the first place; and an explicit accounting, inside the method itself, of how many false
edges the reported graph is expected to contain. A high-recall skeleton that cannot state its
own error budget does not meet the bar here, however well it scores on any single regime.
