Let me start from where the time actually goes. I train big Transformers, and almost all of the compute and memory sits in a handful of dense `n × n` matrices: the query/key/value/output projections in attention, and the two fat matrices of every feed-forward block. Each one is `O(n²)` parameters and `O(n²)` work per token. If I want training to be cheaper, those are the matrices I have to attack.

The textbook move is to replace each dense `W` with a *structured* matrix — something with `o(n²)` parameters and a fast multiply. Sparse, low-rank, a Fourier transform. So why hasn't anyone just done this and walked away with a 2× speedup? Let me be honest about each option and find the actual wall, because the wall is the thing I have to design around.

Take entrywise sparsity first. Prune `W` down to, say, 10% nonzeros. The FLOP count drops by 10×. But I keep measuring this and the wall-clock time on the GPU does *not* drop — it usually gets worse. And it has to: a GPU is a block machine. Its fast units multiply dense tiles; feed it scattered nonzeros and every multiply becomes an irregular gather, memory access stops coalescing, and the dense-tile hardware sits idle. FLOPs are a bookkeeping fiction here. The real currency is "did the work land on dense tiles." Scattered sparsity fails that test by construction. So whatever I build, its nonzeros cannot be scattered — they have to be packed into dense blocks the BMM units can chew on. That is constraint number one, and it is non-negotiable.

The obvious way to satisfy it is a single block-diagonal matrix: dense blocks down the diagonal, zeros elsewhere. That multiplies as a clean batched matmul, fast. But stare at what a block-diagonal matrix *does*: it is exactly block-separable. Coordinate `i` only ever talks to coordinates in its own block. There is no path for information to cross between blocks. A DFT couples every input to every output; a convolution couples neighbors that straddle block boundaries; even a single permutation that swaps two blocks is out of reach. So one block-diagonal matrix is hardware-perfect and expressively crippled. That is the tension in miniature: the thing that's fast (dense blocks) is local, and the thing I need (global mixing) wants to be dense.

Low-rank `UVᵀ` is hardware-fine and has a beautiful projection — Eckart–Young says the best rank-`r` Frobenius approximation of any matrix is just the top-`r` SVD. But a DFT matrix is full rank, a permutation is full rank; low-rank caps me below the transforms I actually want to represent. So low-rank fails the expressiveness demand for the same reason, from the other side.

And the hand-picked fast transforms — drop in an FFT layer? Fast (the FFT has a real GPU kernel) and exact, but it's fixed, not learnable, and choosing the right transform for a task is domain expertise I'd rather the model learn. A general orthogonal-polynomial transform doesn't even have a fast kernel.

So I have three demands and no class meets them: hardware-efficient (dense blocks), expressive (contains the real transforms), and — the one everyone forgets — projectable, meaning given a dense pretrained `A` I can efficiently find the closest member of my class, so I can convert a checkpoint instead of training from scratch. Closed-form projection exists only for sparse (magnitude pruning), low-rank (SVD), orthogonal (Procrustes). For anything richer it's iterative heuristics with no optimality guarantee. I'll keep that third demand in the back of my mind; it's going to turn out to be where the real surprise lives.

Where do I find both expressiveness and a dense-block flavor? The fast transforms themselves. Look at how the FFT actually factorizes. Cooley–Tukey writes the size-`N` DFT recursively:

```
F_N = [ I    Ω  ] [ F_{N/2}    0    ] P_N
      [ I   −Ω  ] [   0     F_{N/2} ]
```

with `Ω = diag(1, ω_N^{-1}, …)` and `P_N` the even/odd sort. Unroll the recursion all the way down and `F_N` becomes a product of `log₂ N` factors, each a block-diagonal matrix whose blocks are `2×2` arrangements of diagonal matrices, times a permutation. That structure — the product of `log n` of these special block-diagonal factors — is exactly the *butterfly matrix*. A butterfly factor is `[[D₁,D₂],[D₃,D₄]]` with each `Dᵢ` diagonal; a butterfly matrix is `B_n B_{n/2} … B_2`.

This is the right ancestor, and I should be precise about why. Butterflies aren't just the FFT; the kaleidoscope result (Dao et al. 2020) says products of butterflies represent *any* structured matrix — anything whose multiply is a low-depth arithmetic circuit — with near-optimal `O(n log n)` parameters and runtime. So if I want a class that contains convolution, Fourier, DCT, Hadamard, Toeplitz, I want a class that contains butterflies. Expressiveness, solved, in principle.

But now run a butterfly on a GPU and it's miserable, and again it has to be. A butterfly matrix is `log n` *sequential* factors, and each factor multiplies by a matrix whose blocks are `2×2` interleavings of diagonal entries — tiny, irregular, and there are `log n` of them in a chain. That's `log n` little kernel launches over awkwardly-strided data. The FLOPs are great, `O(n log n)`, but the work doesn't land on big dense tiles, it dribbles out in scraps. The expressive class is hardware-hostile, and the hostility comes precisely from the `log n` small factors. Pixelated butterfly tried to fix this by flattening the pattern into a fixed block sparsity, but flattening throws away the recursive structure and so throws away expressiveness — and it still can't project a dense matrix.

So the question sharpens: can I keep the butterfly's expressiveness but stop paying for `log n` tiny sequential factors? The factors are the disease. What if I don't multiply them one at a time?

Here's the thing I keep circling. The `log n` butterfly factors are a chain. Matrix multiplication is associative. Nothing forces me to apply them in `log n` steps — I can *precompute* a product of several adjacent factors into one matrix and apply that in a single step. If I multiply the first half of the factors together into one matrix and the second half into another, I've replaced `log n` small multiplies with two big ones. The catch is what those two condensed matrices look like. If condensing the factors just gives me two arbitrary dense matrices I've gained nothing — I'm back to `O(n²)`. The bet is that the condensed factors keep a *block* structure I can still run as batched matmul.

Is there any reason to believe two halves of an FFT condense into something block-structured? Yes — and it's an algorithm I already know, Bailey's four-step FFT. Bailey computes a length-`n` DFT, with `n = m²`, by reshaping the input into an `m × m` matrix and doing: FFT every column, multiply by twiddle factors, transpose, FFT every row. Now look at "FFT every column": that is `m` independent size-`m` transforms applied in parallel — which is exactly a multiply by a **block-diagonal** matrix with `m` blocks of size `m × m`. "FFT every row," after the transpose, is *another* block-diagonal multiply. Two block-diagonal passes, with a transpose (a permutation) between them. Bailey did this for the memory hierarchy, but what it's really telling me is: the entire `log n`-factor butterfly chain, the whole FFT, can be reorganized into **two block-diagonal multiplies separated by a permutation.** The `log n` collapses to 2.

Let me make this exact in butterfly terms, because I want it to hold for the whole butterfly class, not just the DFT. Take a butterfly matrix `B = B_n B_{n/2} … B_2` for `n` a power of 4. Split the factors in half. Let `R` be the product of the *last* `(log₂ n)/2` factors and `L'` be the product of the *first* `(log₂ n)/2`. Each butterfly factor matrix `B_k` is block-diagonal with blocks of size `k`. Multiplying together the small-block factors `B_b B_{b/2} … B_2` (block sizes up to `b`) keeps the result block-diagonal with block size `b` — small blocks compose into bigger blocks but stay block-diagonal. Set `b = m = √n`: then `R` is genuinely block-diagonal, `m` dense blocks of size `m × m`. Good — `R = diag(R₁, …, R_m)`.

What about the top half `L'`? Each top-half factor `B_k` (for `k` from `n` down to `2b`) is, by definition, a block *matrix* whose blocks are diagonal — its nonzeros sit on the diagonal of each block. Multiply the top half together and `L'` comes out as an `m × m` array of blocks where *each block is itself a diagonal matrix*:

```
        ⎡ D₁₁ … D₁ₘ ⎤
  L' =  ⎢  ⋮  ⋱  ⋮  ⎥
        ⎣ Dₘ₁ … Dₘₘ ⎦
```

That's *not* block-diagonal, so it doesn't directly run as a batched matmul. But it's one permutation away. Here's the picture: take the permutation `P` that reshapes a length-`n` vector into an `m × m` matrix, transposes it, and flattens it back — the *stride* or *transpose* permutation. It is its own inverse, `P = Pᵀ`. Conjugating `L'` by it, `L := P L' Pᵀ`, swaps the roles of "which block" and "where in the block": the array-of-diagonal-blocks turns into a genuine block-diagonal matrix, `L = diag(L₁, …, L_m)`, `m` dense blocks of size `m × m`. (This is exactly the transpose step sitting between Bailey's two FFT passes.) So the original top half is `L' = Pᵀ L P`, and

```
  B = L' R = Pᵀ L P R.
```

Since `P = Pᵀ`, that's the same as `B = P L Pᵀ R`. So: any butterfly `B = P L Pᵀ R`, where both `L` and `R` are honest block-diagonal matrices (`m` blocks of `m × m`) and `P` is the fixed transpose permutation. Two forward block-diagonal multiplies with the permutation explicit between them — the orientation Bailey hands me, where everything runs as batched matmul.

That's the object. Let me just *define* the class by that form and not require it to come from a butterfly at all:

> A matrix `M` of size `n × n`, `n = m²`, is in my class `𝓜` if `M = P L Pᵀ R`, where `L` and `R` are each block-diagonal with `m` blocks of size `m × m`, and `P` is the transpose permutation.

Two block-diagonal matrices, one fixed permutation. I'll call these Monarch matrices.

Now, does this buy me what I wanted, and at what price?

Expressiveness first, because that was the whole reason to come from butterflies. Every butterfly `B` can be written in this form — I just showed it, by condensing the `log n` factors into `L` and `R`. Condensing factors into a product can only *enlarge* the set of matrices I can express (a product of two free block-diagonal matrices has more freedom than the constrained product of `log n` butterfly factors), so `𝓑 ⊂ 𝓜`: Monarch contains all butterflies, strictly. Let me sanity-check the strictness by counting how many parameters a Monarch matrix genuinely needs, because that's where it bites. Even if I pin `R` to the identity, `L` is still a free block-diagonal matrix with `√n` blocks of `√n × √n` — that alone is `n^{3/2}` arbitrary entries, so describing any Monarch matrix takes at least `n^{3/2}` parameters. A butterfly has only `2n log₂ n`. And `2n log₂ n < n^{3/2}` once `n > 256` (it's the `log₂ n` versus `√n/2` race; `√n` wins for large `n`), so beyond that size a butterfly simply cannot have enough free parameters to reach every Monarch matrix — the inclusion is strict. (The count isn't tight; one can push the threshold lower with more care, but I just need *some* `n` where it bites.) More directly, a free product of two block-diagonal matrices has degrees of freedom the constrained product of `log n` butterfly factors does not, so there are matrices in `𝓜` that no butterfly can produce. Monarch is the *bigger*, more expressive class.

And the richer transforms? Butterflies alone give me the things that are single low-depth circuits, but to get the DFT itself, or DCT/DST, you need a *product* of butterflies — the `BB*` kaleidoscope classes. Since `𝓜 ⊃ 𝓑`, products of Monarch matrices contain products of butterflies: `𝓜𝓜* ⊃ 𝓑𝓑*` and `(𝓜𝓜*)² ⊃ (𝓑𝓑*)²`. Whatever the kaleidoscope hierarchy can represent, the Monarch hierarchy can too. The kaleidoscope result says `𝓑𝓑*` represents convolution, Hadamard, Toeplitz, AFDF; `(𝓑𝓑*)²` represents Fourier, DST/DCT, `(HD)³`, Fastfood, ACDC. So `𝓜𝓜*` and `(𝓜𝓜*)²` represent all of those. I inherit the entire expressiveness story for free, just by containing butterflies. (The only cost: the general Monarch-hierarchy representation of an arbitrary depth-`d`, `s`-gate circuit is suboptimal versus kaleidoscope by an `O(d√s)` factor, because a Monarch block is `O(n^{1.5})` where a butterfly is `O(n log n)`. I'm trading a polylog parameter overhead for hardware regularity. Given that scattered sparsity costs me *everything* on a GPU, a `√s` factor on paper is a fine price.)

Now efficiency, the whole point. Parameters `2n√n` — sub-quadratic, good. FLOPs: multiply by `R` (block-diagonal, `m` blocks of `m×m`, that's `m · m² = n√n` work), permute (free), multiply by `L` (another `n√n`), permute (free). Total `O(n√n)`. Notice that's *more* than the butterfly's `O(n log n)` in raw FLOP count. I should be uneasy about that for exactly half a second — and then remember the lesson from the top: FLOPs aren't the currency. Those `n√n` FLOPs are organized as **two batched matrix multiplies over dense `m × m` blocks**, the single operation a GPU is fastest at. The butterfly's `O(n log n)` FLOPs were organized as `log n` tiny irregular passes. Two big BMMs beat `log n` small irregular kernels in wall-clock by a wide margin even though the BMMs do more arithmetic. The asymptotic regression in FLOPs is the *price of regularity*, and regularity is what I'm buying. This is the inversion the whole design hinges on.

Let me make the multiply concrete as code-shaped operations, because it also sets up the next part. Don't materialize `P` or `M`. View the input `x`, length `n = m²`, as an `m × m` tensor `x_{ki}` (`k` indexes block, `i` indexes within). Then:

- Multiply by `R`: `R` is block-diagonal, so block `k` acts on row `k` of `x`. `y_{kj} = Σ_i R_{kji} x_{ki}`. That's a batched matmul over `k`.
- Apply `P L Pᵀ` to `y`: the conjugation by `P` makes `L` act along the *other* axis. `z_{ℓj} = Σ_k L_{jℓk} y_{kj}`. Another batched matmul, now over `j`.
- Flatten `z` back to a length-`n` vector.

So `z_{ℓj} = Σ_{k,i} L_{jℓk} R_{kji} x_{ki}`. Two BMMs with a transpose between them — Bailey's two passes, exactly. No permutation matrix ever built; it's just which axis the matmul runs along.

And now I want the third demand, projection: given a dense pretrained `A`, find the Monarch `M` minimizing `‖A − M‖²_F`. This should be hard. `M = PLPᵀR` is a *product* of two unknown matrices, so the objective is nonconvex in `(L, R)` — generically that means iterate-and-pray, local minima, no guarantee. Everyone before me on butterfly projection got stuck exactly here and fell back to first-order heuristics. Let me not assume it's hard, though; let me write `M` out in coordinates and just look.

From the multiply I have `z_{ℓj} = Σ_{k,i} L_{jℓk} R_{kji} x_{ki}`. The matrix `M` is whatever linear map sends `x_{ki}` to `z_{ℓj}`, so reading off the coefficient of `x_{ki}` in `z_{ℓj}`:

```
  M_{ℓjki} = L_{jℓk} · R_{kji}.
```

View `M` as a 4D tensor of size `m × m × m × m`, indexed `(ℓ, j, k, i)`. Stare at that product. For *fixed* `(j, k)`, what is `M` as a function of the remaining indices `(ℓ, i)`? It's `L_{jℓk} · R_{kji}` — the first factor depends only on `ℓ`, the second only on `i`. That is an outer product. A function of two indices that factors as (something in `ℓ`) times (something in `i`) is a **rank-1 matrix**. Define `p_{jk}` to be the length-`m` vector `(L_{jℓk})_ℓ` and `q_{jk}` to be `(R_{kji})_i`. Then the `(j,k)` slice of `M` is exactly `p_{jk} q_{jkᵀ}` — rank one.

Huh. So the 4D tensor `M`, sliced over `(j, k)`, is `m · m` independent **rank-1** matrices. The Monarch structure, which looked like a tangled product, is — after this one reshape — just "every block, in a particular reshaping, has rank 1." That's it. That's the whole constraint.

Now the projection collapses. The Frobenius norm is a sum of squared entries and doesn't care about the shape, so reshape `A` into the same 4D tensor `A_{ℓjki}` and write:

```
  ‖A − M‖²_F = Σ_{ℓjki} (A_{ℓjki} − L_{jℓk} R_{kji})²
             = Σ_{j,k} [ Σ_{ℓ,i} (A_{ℓjki} − L_{jℓk} R_{kji})² ].
```

The cross terms vanish — the outer sum over `(j,k)` separates completely, because the `(j,k)` slice of `M` only involves the `(j,k)` blocks of `L` and `R`. So the nonconvex global problem breaks into `m²` **independent** subproblems, one per `(j,k)`, and each subproblem is: find the best rank-1 approximation, in Frobenius norm, of the `m × m` slice `A_{:,j,k,:}`. That is *the Eckart–Young problem*. The best rank-1 Frobenius approximation of a matrix is the top singular component of its SVD — `σ₁ u₁ v₁ᵀ`. So I solve each slice by an SVD, take the rank-1 piece `u_{jk} v_{jkᵀ}`, set `L_{jℓk} = (u_{jk})_ℓ` and `R_{kji} = (v_{jk})_i`, and reshape back into block-diagonal `L` and `R`.

That's an *analytical optimum* for a nonconvex problem — the direct analogue of Eckart–Young, but for Monarch instead of low-rank. The nonconvexity was an illusion created by writing `M` as a product; once I reshape, the product is `m²` separate rank-1 facts, each of which has a closed form. Cost: `m²` SVDs of `m × m` matrices, each `O(m³)`, total `O(m⁵) = O(n^{5/2})`. And note: if `A` was a Monarch matrix to begin with, each slice is *exactly* rank 1, the SVD recovers it perfectly, and I get back the factors with `A = PLPᵀR`. So this same routine both projects *and* factorizes.

Let me push the factorization idea one notch further, because the transforms I most want to store cheaply — the FFT, the DCT — aren't single Monarch matrices, they're in `𝓜𝓜*`, a *product* of two Monarch matrices. If I have such a matrix `M` (say I built an exact FFT) and I want its Monarch factors so I can apply it as cheap BMMs, the per-slice-rank-1 trick doesn't directly apply, because a product of two Monarchs isn't rank-1-per-block. I need a different recovery.

Write `M ∈ 𝓜𝓜*`. Conjugate it the same way: `M̂ = P M Pᵀ`. Working it through, `M̂ = L₁ (P R Pᵀ) L₂`, where `L₁, L₂` are block-diagonal (the diagonal blocks `Aᵢ` and `Cⱼ` respectively) and the middle factor `P R Pᵀ` is the array-of-diagonal-blocks form — block `(i,j)` is a diagonal matrix `D_{ij}`. So if I cut `M̂` into `m × m` blocks of size `m × m`, block `(i,j)` is

```
  M̂_{ij} = Aᵢ D_{ij} Cⱼ,
```

`Aᵢ, Cⱼ` dense and invertible (assume `M` invertible), `D_{ij}` diagonal and invertible (assume `R` has no zeros in its blocks — that's exactly the condition that the `D_{ij}` have no zero on the diagonal). My job: recover `Âᵢ`, `Ĉⱼ`, diagonal `D̂_{ij}` with `M̂_{ij} = Âᵢ D̂_{ij} Ĉⱼ`.

Warm up with the toy case `D_{ij} = I` for all `i,j`, so `M̂_{ij} = Aᵢ Cⱼ`. There's an obvious gauge freedom: I can post-multiply each `Aᵢ` and pre-multiply each `Cⱼ` by inverse scalings/permutations and nothing changes. So I'm free to *pin* one factor: set `Ĉ₁ = I`. Then immediately `Âᵢ = M̂_{i1}` (since `M̂_{i1} = Aᵢ Ĉ₁ = Aᵢ`). And `Ĉⱼ = Â₁⁻¹ M̂_{1j}`. Does this reconstruct every block? Check `Âᵢ Ĉⱼ = M̂_{i1} M̂_{11}⁻¹ M̂_{1j} = (Aᵢ C₁)(A₁ C₁)⁻¹(A₁ Cⱼ) = Aᵢ C₁ C₁⁻¹ A₁⁻¹ A₁ Cⱼ = Aᵢ Cⱼ = M̂_{ij}`. It works. With identity middle, one pinned block determines everything.

The real case has the diagonal `D_{ij}` in the middle, so I can't just pin `Ĉ₁ = I` and read things off — the `D`'s pollute the products. But the gauge argument generalizes: as long as I find a `Ĉ₁` that's *correct up to a row permutation and scaling*, I can fold the leftover scalings into the `D̂`'s and recover the rest by the same `Âᵢ = M̂_{i1} Ĉ₁⁻¹`, `Ĉⱼ = Â₁⁻¹ M̂_{1j}` formulas. So the entire problem reduces to: find one good `Ĉ₁`.

How do I pin down `C₁` from the data when there are unknown diagonals everywhere? I want to form a combination of the `M̂_{ij}` in which the `Aᵢ` and `Cⱼ` cancel and only diagonal stuff (conjugated by `C₁`) survives. Try

```
  F(i,j) := M̂_{i1}⁻¹ M̂_{ij} M̂_{1j}⁻¹ M̂_{11}.
```

Substitute `M̂_{ij} = Aᵢ D_{ij} Cⱼ`:

```
  F(i,j) = (C₁⁻¹ D_{i1}⁻¹ Aᵢ⁻¹)(Aᵢ D_{ij} Cⱼ)(Cⱼ⁻¹ D_{1j}⁻¹ A₁⁻¹)(A₁ D₁₁ C₁)
         = C₁⁻¹ (D_{i1}⁻¹ D_{ij} D_{1j}⁻¹ D₁₁) C₁.
```

Every `Aᵢ` and `Cⱼ` telescopes away. What's left in the middle, `D_{i1}⁻¹ D_{ij} D_{1j}⁻¹ D₁₁`, is a product of diagonal matrices — so it's **diagonal**. Therefore `C₁ F(i,j) C₁⁻¹` is diagonal for *every* `(i,j)`: the single matrix `C₁` **simultaneously diagonalizes** all the `F(i,j)`. And the converse holds — *any* matrix that simultaneously diagonalizes all the `F(i,j)` is a valid `Ĉ₁` (it's `C₁` up to the permitted permutation+scaling gauge). So I don't need the "true" `C₁`; I just compute *some* simultaneous diagonalizer of the `F(i,j)` and call it `Ĉ₁`.

Simultaneous diagonalization is finding a shared eigenbasis. Diagonalize `F(1,2)`; if its eigenvalues are distinct the eigenbasis is unique up to permutation/scaling and I'm done — that basis is `Ĉ₁`. If there are repeated eigenvalues, the eigenbasis has freedom within each eigenspace, so I refine: take the current basis, look at the next `F(i,j)`, and within each degenerate block diagonalize it too; repeated eigenvalues of one matrix get split by another. (Any linear combination of eigenvectors sharing an eigenvalue is still an eigenvector, so refining one matrix never un-diagonalizes the ones already handled.) Process all the `F(i,j)`; the resulting basis simultaneously diagonalizes them all. Then `Âᵢ = M̂_{i1} Ĉ₁⁻¹`, `Ĉⱼ = Â₁⁻¹ M̂_{1j}`, and finally `D̂_{ij} = Âᵢ⁻¹ M̂_{ij} Ĉⱼ⁻¹` — and this last is guaranteed diagonal because `D̂_{ij} = Ĉ₁ F(i,j) Ĉ₁⁻¹`, which is diagonal by construction of `Ĉ₁`. Cost: `m²` matrices `F(i,j)`, each an eigen-style decomposition on `m × m` blocks, `O(m³)` each, total `O(m⁵) = O(n^{5/2})` again (more generally `O(n³/b)` for block size `b`). So products of Monarch matrices — the FFT, the DCT — can be cracked into Monarch factors and stored as cheap BMMs.

One more thing I need before this is usable on real weight matrices, which are rarely square and where I want a parameter knob. The square definition forces `m = √n` and a fixed parameter budget `2n√n`. In practice I want to dial parameters up or down (the analogue of the rank in low-rank, or the nonzero count in sparse) and I want rectangular `out × in`. The clean way: don't tie the block count to `√n` — make the **block size `b`** a free choice (with `b` dividing `n`). Then `R` is block-diagonal with blocks of size `b`, `L` is the array-of-`b×b`-diagonal-blocks form (and conjugating by the appropriate stride permutation `P_{(b,n)}` turns it block-diagonal with `n/b` blocks). Parameters become `n²/b + nb`. The permutation generalizes: `P_{(b,n)}` reshapes a length-`n` vector as `b × (n/b)`, transposes, flattens — in index form, if `i = i₁·b + i₀` then `σ_{(b,n)}(i) = i₀·(n/b) + i₁`. Everything I proved — that `L` permutes to block-diagonal, that butterflies sit inside (`𝓑 ⊂ 𝓜^{(b,n)}` for every valid `b`), that the projection decouples — goes through with `b` in place of `√n`. For rectangular `M`, the blocks just become rectangular and `L`'s blocks become "wrapped diagonal." In practice the right knob is the *number* of blocks: pick 2 to 4 blocks, which lands the parameter count at 25–50% of dense — the budget I'd reach for, since it leaves enough capacity to hope for dense-level quality while the two BMMs should comfortably beat a single dense GEMM in wall-clock. Fewer blocks → closer to dense; more blocks → cheaper but less expressive.

So the design is complete and I can see how each usage mode falls out. End-to-end from scratch: swap every dense `nn.Linear` in the attention projections and FFN blocks for a Monarch layer with `nblocks=4`, initialize the blocks like a dense layer (Kaiming-uniform on each block, fan-in = block input size, so it behaves like the dense baseline at init), and train with Adam through the two BMMs — the parameterization is differentiable, autograd handles it. Sparse-to-dense "reverse sparsification": train the Monarch model for ~90% of iterations (fast, because of the BMMs), then *densify* by literally multiplying out `L` and `R` with the permutation to get a dense matrix, and finish the last 10% dense — same total iterations, but most of them cheap, and you end with a full dense model where sparse training alone would have struggled to optimize or fit capacity. Dense-to-sparse fine-tuning: take a pretrained dense checkpoint, run the closed-form projection on each weight matrix to get its best Monarch approximation, and fine-tune from there — no retraining from scratch, because the projection transferred the pretrained knowledge in one SVD per block. And for the science/medical tasks where an FFT is the incumbent: project the (I)FFT matrix onto `𝓜` (exact, since the FFT lives in the class) to *initialize* a Monarch layer at the right transform, then let it learn — you start from the FFT and adapt away its aliasing, with far fewer parameters than a CNN, which is what you want in data-limited regimes.

Now let me write it as the real thing. The multiply never builds `P` or `M`; it's two batched matmuls with a reshape between, matching `z_{ℓj} = Σ_{k,i} L_{jℓk} R_{kji} x_{ki}`. The projection is the reshape-into-batched-slices, a batched rank-1 SVD, and reshape into the two block-diagonal factors. The layer wraps both as a drop-in for `nn.Linear`.

```python
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange


# === the Monarch matrix-vector multiply: M @ x as two batched dense matmuls ===
# x: (batch, n);  w1 = R: (k, q, p) with k*p = n;  w2 = L: (l, s, r) with l*r = k*q
# out: (batch, l*s).  Square Monarch: k=q=p=l=s=r=sqrt(n).
class MonarchMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1_bfly, w2_bfly):
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = int(np.prod(batch_shape))
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        assert k * p == n
        assert l * r == k * q
        # view x as (k, batch, p): one m-vector per block, then bmm by R block-wise
        x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
        out1 = torch.empty(batch_dim, k, q, device=x.device, dtype=x.dtype).transpose(0, 1)
        out1 = torch.bmm(x_reshaped, w1_bfly.transpose(-1, -2), out=out1)   # y_{kj}=sum_i R_{kji} x_{ki}
        # the permutation P: re-stride so the next bmm runs along the OTHER axis
        out1 = out1.transpose(0, 1).reshape(batch_dim, r, l).transpose(-1, -2).contiguous().transpose(0, 1)
        out2 = torch.empty(batch_dim, l, s, device=x.device, dtype=x.dtype).transpose(0, 1)
        out2 = torch.bmm(out1, w2_bfly.transpose(-1, -2), out=out2)         # z_{lj}=sum_k L_{jlk} y_{kj}
        out2 = out2.permute(1, 2, 0).reshape(*batch_shape, s * l)
        ctx.save_for_backward(x, w1_bfly, w2_bfly, out1)
        return out2

    @staticmethod
    def backward(ctx, dout):
        x, w1_bfly, w2_bfly, out1 = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = int(np.prod(batch_shape))
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        dx = dw1 = dw2 = None
        dout_reshaped = dout.reshape(batch_dim, s, l).transpose(-1, -2).contiguous().transpose(0, 1)
        if ctx.needs_input_grad[2]:
            dw2 = torch.bmm(dout_reshaped.transpose(-1, -2), out1.conj())
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[0]:
            dout1 = torch.bmm(dout_reshaped, w2_bfly.conj())
            dout1 = dout1.transpose(0, 1).transpose(-1, -2).contiguous().reshape(batch_dim, k, q).transpose(0, 1)
            if ctx.needs_input_grad[0]:
                dx = torch.bmm(dout1, w1_bfly.conj()).transpose(0, 1).reshape(*batch_shape, n)
            if ctx.needs_input_grad[1]:
                x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
                dw1 = torch.bmm(dout1.transpose(-1, -2), x_reshaped.conj())
        return dx, dw1, dw2

monarch_multiply = MonarchMultiply.apply


# === Eckart-Young: best rank-r Frobenius approximation of each matrix in a batch ===
def low_rank_project(M, rank):
    U, S, Vt = torch.linalg.svd(M)
    S_sqrt = S[..., :rank].sqrt()
    U = U[..., :rank] * rearrange(S_sqrt, '... rank -> ... 1 rank')
    Vt = rearrange(S_sqrt, '... rank -> ... rank 1') * Vt[..., :rank, :]
    return U, Vt


# === closed-form projection of a dense matrix A onto the Monarch class ===
# reshape A so each Monarch block is one batched matrix, take its rank-1 SVD.
def monarch_project(M, sizes=None):
    m, n = M.shape
    assert m == n, 'square only'
    if sizes is None:
        f = [(i, n // i) for i in range(1, int(math.isqrt(n)) + 1) if n % i == 0][-1]
        sizes = (f[1], f[0])                       # block sizes closest to sqrt(n)
    assert n == sizes[0] * sizes[1]
    # the (j,k) slice A_{:,j,k,:} of the 4D reshape becomes one matrix in the batch
    M_batched = rearrange(M, '(p k) (r s) -> k r p s', k=sizes[1], r=sizes[0])
    U, Vt = low_rank_project(M_batched, rank=1)     # each slice -> best rank-1 (Eckart-Young)
    w1_bfly = rearrange(Vt, 'k r 1 s -> r k s')     # the R factor
    w2_bfly = rearrange(U, 'k r s 1 -> k s r')      # the L factor
    return w1_bfly, w2_bfly


# === the Monarch linear layer: drop-in for nn.Linear ===
class MonarchLinear(nn.Module):
    def __init__(self, in_features, out_features, nblocks=4, bias=True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        in_blksz = math.ceil(in_features / nblocks)
        out_blksz = math.ceil(out_features / nblocks)
        self.in_features_extended = in_blksz * nblocks
        self.out_features_extended = out_blksz * nblocks
        if self.in_features_extended < self.out_features_extended:
            self.blkdiag1 = nn.Parameter(torch.empty(nblocks, in_blksz, in_blksz))
            self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz))
        else:
            self.blkdiag1 = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz))
            self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, out_blksz))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        # init each block like a dense layer's weight: fan_in = block input dim
        for blkdiag in [self.blkdiag1, self.blkdiag2]:
            fan_in = blkdiag.shape[-1]
            std = init.calculate_gain('leaky_relu', math.sqrt(5)) / math.sqrt(fan_in)
            bound = math.sqrt(3.0) * std
            with torch.no_grad():
                blkdiag.uniform_(-bound, bound)

    def forward(self, x):
        if x.shape[-1] < self.in_features_extended:           # pad to extended size
            x = F.pad(x, (0, self.in_features_extended - x.shape[-1]))
        out = monarch_multiply(x, self.blkdiag1, self.blkdiag2)
        out = out[..., :self.out_features]                    # crop to out_features
        return out + self.bias.to(out.dtype) if self.bias is not None else out
```

The causal chain, start to end: GPUs reward dense blocks and punish scattered nonzeros, so any usable structured matrix must be built from dense blocks; butterflies are the maximally expressive structured class but pay for it with `log n` tiny irregular factors; Bailey's four-step FFT shows that whole chain can be reorganized into two block-diagonal multiplies separated by a transpose, which I lift into the definition `M = P L Pᵀ R`; this class strictly contains butterflies, so it inherits their expressiveness (and via products, all the fast transforms) while running as exactly two batched matmuls; the seemingly-nonconvex projection of a dense matrix onto the class becomes, after reshaping `M` to a 4D tensor where each block is rank-1, a batch of independent Eckart–Young rank-1 problems solved in closed form by per-block SVD; and products of Monarch matrices can be factorized back by simultaneous diagonalization — together giving end-to-end sparse training, reverse-sparsification dense training, and dense-to-sparse fine-tuning from a single matrix family.
