# Answer

Reingold proves that undirected `s-t` connectivity is in deterministic log space, and therefore `SL = L`.

The proof derandomizes the structure, not merely the walk. Starting from an undirected graph on `[N]`, it builds a regular graph `G_reg` on `[N] x [N]`: labels `1` and `2` make a cycle inside each vertex cloud, label `3` encodes the original edge `(v,w) <-> (w,v)` when it exists and is otherwise a self-loop, and all remaining labels are self-loops. This preserves connected components and makes each component non-bipartite.

The main transform is local in rotation maps. For a `D^16`-regular graph `G` on `M` vertices and a fixed `D`-regular graph `H` on `[D^16]` with `lambda(H) <= 1/2`, set

`ell = 2 ceil(log(D M^2))`, `G_0 = G`, and `G_i = (G_{i-1} z H)^8`.

The zig-zag product lowers the degree to `D^2`, and the eighth power restores it to `D^16`. If `lambda = lambda(G_{i-1})`, the JACM zig-zag bound and powering give `lambda(G_i) < [1-(1-lambda)/3]^8`; this is below `1/2` when `lambda < 1/2`, and below `lambda^2` when `lambda >= 1/2`. Since a connected non-bipartite `D^16`-regular graph starts with `lambda <= 1 - 1/(D M^2)`, the chosen `ell` yields `lambda(G_ell) <= 1/2`.

Both powering and zig-zag act separately on connected components, so disconnected original components never merge. The final implicit graph has constant degree and logarithmic diameter inside each original connected component. Deterministically enumerating all logarithmically long labelled walks from `(s,1^{ell+1})` decides whether `(t,1^{ell+1})` is reached.

The implementation-level object is the Appendix A evaluator for `Rot_{G_ell}`. It stores shared variables `v,a_0,...,a_ell`, recursion height `I`, counters `j_1,...,j_ell`, and reusable scratch space; each level performs constant many `H` rotations and recursive lower-level rotation calls. This is why the logarithmically many phases do not turn into Savitch-style `O(log^2 N)` space.

Grounding: Reingold JACM 2008 Theorems 4.1/4.2, Definition 3.1, Lemmas 3.2-3.4, and Appendix A; ECCC 2004 preliminary version for lineage; RVW zig-zag paper for rotation maps and the product; Li-Lee and Trevisan expositions for independent reconstruction.
