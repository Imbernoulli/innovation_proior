I begin with the dense Johnson-Lindenstrauss proof because it tells me what I am not allowed to lose. For one fixed vector, a random map to `k = Theta(epsilon^-2 log(1/delta))` coordinates preserves the squared norm with failure probability at most `delta`. For a point set, I can union-bound over differences. The target dimension is already right. The unsolved part is the multiplication cost: one nonzero input coordinate still writes into essentially all `k` output coordinates.

My first instinct is to zero out most matrix entries independently. That feels like the shortest path from dense JL to sparse JL, but a basis vector immediately breaks the thought. If the only nonzero input coordinate lands in a random number of rows, then the projected norm of that basis vector fluctuates before any interaction with other coordinates appears. This is diagonal noise. No sign cancellation can repair it because there is no second coordinate to cancel against.

So I change what I demand from a sparse column. I do not want each entry to be independently present. I want exactly `s` active entries in every column, each with magnitude `1/sqrt{s}` and an independent sign. Now a basis vector is preserved exactly. The diagonal part of `||Sx||_2^2` is no longer random. That is the first real insight: the sparse transform has to spend its design budget removing self-error before it can ask a concentration proof to handle cross-error.

With that fixed-column design, I write the squared norm out. If `eta_{r,i}` records whether column `i` uses row `r`, and `sigma_{r,i}` is its sign, then for unit `x` the error is

`Z = ||Sx||_2^2 - 1 = (1/s) sum_r sum_{i != j} eta_{r,i} eta_{r,j} sigma_{r,i} sigma_{r,j} x_i x_j`.

Everything left is a collision between two different input coordinates. The problem becomes much sharper. I am no longer trying to prove that sparse rows look subgaussian in a generic sense. I am trying to show that this signed collision polynomial is small.

There is a clean route that almost solves the problem. Suppose I can arrange the column supports so every pair of columns collides in only about `s^2/k` rows. Then, after the locations are fixed, `Z` is a quadratic form in the signs. Hanson-Wright controls that quadratic form through two matrix norms. The Frobenius norm asks for `k` on the dense JL scale. The spectral norm asks for `s` on the `epsilon^-1 log(1/delta)` scale. This route explains the right answer before it gives the final method.

But I see the cost of that clean route. A worst-case pairwise collision promise is stronger than a fixed vector needs. If I ask a random support pattern to satisfy that promise for every pair of columns in a large ambient dimension, I pay extra logarithms or dimension dependence. The final proof cannot condition on a globally perfect collision code. It has to analyze the random collisions directly for the vector in front of it.

I turn the proof around. Instead of proving every pair of columns behaves well, I expand a high moment of `Z`. I choose an even moment `ell` on the order of `log(1/delta)`, because Markov's inequality will turn an `ell`-moment bound into a tail bound. A term in the expansion is a product of many collision factors. I can view it as a directed multigraph whose vertices are input coordinates and whose edges are collision events.

The signs do the first filtering. Since each edge contributes two signs, the expectation over signs vanishes unless every vertex has even degree in the multigraph. This is where limited independence becomes enough. I only expand up to degree `ell`, so I only need the signs to behave independently on the coordinates that can appear inside that expansion. Full independence is convenient, but it is not the mathematical reason the method works.

The support variables do the second filtering. In the graph version, a column chooses `s` distinct rows. In the block version, it chooses one row per block. Either way, the events that create collisions are not arbitrary dense interactions. The support indicators have the right upper bounds, and sampling without replacement gives helpful negative dependence. Each surviving graph pays collision probabilities tied to powers of `s/k`, while the even-degree constraint limits how many such graphs I have to count.

The moment calculation is the heart of the result. The primary source first proves a bound for the contribution from one row or block, then assembles the full `ell`-moment. The surviving terms are controlled well enough that, after constants, the choice

`s = Theta(epsilon^-1 ell)` and `k = Theta(s^2/ell) = Theta(epsilon^-2 ell)`

makes `E[|Z|^ell]` small on the scale needed for a failure probability `delta`. Since `ell = Theta(log(1/delta))`, this gives the same target dimension as dense JL and only `Theta(epsilon^-1 log(1/delta))` nonzeros per column.

This proof also clarifies why the construction is not just JL plus a hash table. Hashing alone tells me how to update a few counters. It does not by itself give a linear Euclidean embedding. CountSketch can estimate through several repetitions and a median, but the median is nonlinear. I need the norm of one linear image to concentrate. That is why the diagonal cancellation, the exact column sparsity, and the direct high-moment collision proof all matter.

The lower bounds confirm that the parameter is not a proof artifact. For these graph and block sparse embeddings, a two-coordinate vector can force `s = Omega(epsilon^-1 log(1/delta))`. For with-replacement sparse schemes, even a basis vector can expose repeated choices inside a single column. The construction succeeds because it removes the self-collision weakness and then pays exactly the amount of sparsity needed to control cross-collisions.

At the end I see the discovery path as a sequence of exclusions. Dense JL has the right dimension but wrong update time. Constant-sparse entry distributions still touch too many rows. Fast transforms spread mass but cost too much before sparse updates begin. Hash sketches are fast but their strongest estimators are not linear embeddings. The remaining object has to be a fixed-column-sparsity linear map whose only error is signed pair collisions. Once I focus on that polynomial, the limited-independence high-moment proof becomes the mechanism that lets sparsity and JL concentration coexist.
