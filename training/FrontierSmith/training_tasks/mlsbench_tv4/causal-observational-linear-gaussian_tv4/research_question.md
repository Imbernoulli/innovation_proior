Two of the five graded settings draw their graphs from a Barabasi-Albert process, and one of
them is the regime that most often decides the aggregate: fifty nodes, denser attachment, only
a thousand samples. Preferential attachment concentrates degree on a few hubs, and hubs are
where degree-blind methods break — every pair of a hub's many children is marginally
correlated through the shared parent, so the estimated skeleton grows a clique where the true
graph has a star. This variant asks for a method built around that specific failure mode.

The specialization must be structural, not cosmetic. The method should estimate where the
degree mass sits and let that estimate reshape its decisions: which conditioning sets get
tried first, how aggressively correlated pairs near a suspected hub are pruned, how
orientation treats edges incident to high-degree nodes. What it may not do is detect the graph
family and branch — the two Erdos-Renyi regimes and the hidden noisy setting are scored by the
same run, so heavy-tailed-degree machinery has to be harmless when the degree profile turns
out flat. A hub-aware method that wrecks ER performance has merely relocated the failure.

The claim this variant defends is comparative: on the scale-free settings, explicitly modeling
degree heterogeneity must beat degree-blind thresholding on adjacency precision and SHD — the
star recovered as a star, not as a clique — while the Erdos-Renyi settings stay within noise
of what a competent generic method achieves. Progress is measured by the harness exactly as
before; what changes is where the method spends its statistical budget.
