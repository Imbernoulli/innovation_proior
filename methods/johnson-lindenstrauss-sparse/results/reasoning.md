Let me start from the thing that actually hurts. I have the Johnson–Lindenstrauss lemma in hand and I trust it completely: for any unit vector $x\in\mathbb{R}^d$ there is a distribution over $k\times d$ matrices $S$ with $k=\Theta(\varepsilon^{-2}\log(1/\delta))$ such that $\Pr[(1-\varepsilon)\le\|Sx\|_2\le(1+\varepsilon)]>1-\delta$, and that $k$ is optimal. Union-bound over the $\binom n2$ difference vectors and $n$ points flatten into $O(\varepsilon^{-2}\log n)$ dimensions. None of that is in question. What is in question is the cost of *using* it. The standard proof hands me a dense $S$ — i.i.d. Gaussian, or Achlioptas's rescaled Rademacher coins — and to apply a dense $k\times d$ matrix to $x$ I pay $\Theta(k\cdot\|x\|_0)$, which is $\Theta(kd)$ for a dense vector. In every pipeline where I reach for JL — nearest neighbor, closest pair, regression, low-rank — this multiply is the dominant term. I am introducing dimension reduction to *speed up* a search, and the reduction itself is now the slow step. That is the wound.

So the goal is sharp: keep the optimal $k$, keep a genuine $\ell_2\to\ell_2$ linear image (I'll come back to why I insist on "genuine"), and get the per-vector cost well under $kd$. Two regimes pull in different directions and I should hold them apart. One is a dense $x$ where I'd love $O(d\log d)$. The other is a sparse or streaming $x$, $\|x\|_0\ll d$, where a coordinate update $x\leftarrow x+v\,e_i$ should cost something like $\|x\|_0$, not $d$ — note that even the dumb dense product already does $O(k\cdot\|e_i\|_0)=O(k)$ on a single-coordinate update, so any "fast" transform that still pays $\Omega(d)$ per update is a step backward there. Different pains, probably different fixes.

Take the first regime. The obvious move is: make $S$ sparse. If each column has $s$ nonzeros then $Sx$ costs $\Theta(s\cdot\|x\|_0)$, and I'd be done if I could push $s$ way down. Let me just try it and see where it breaks. Suppose I zero out most entries of $S$ independently. Now feed it the nastiest input I can think of: $x=e_i$, a single spike. The estimator $\|Sx\|_p$ only ever touches column $i$ of $S$. If that column is mostly zero — and at high sparsity it is — then with substantial probability the few surviving entries badly mis-measure the unit norm. The estimator's variance is enormous on this input. A sparse matrix cannot preserve the norm of a sparse vector. That's the wall, and it's a hard one: it's exactly why one can only sparsify by a *constant* factor directly (Achlioptas gets to two-thirds zeros, a 3× win, and stops). The trouble isn't sparsity per se; it's the collision between a sparse matrix and a *spiky* input.

Stare at that. The matrix is fine for *spread-out* inputs — the danger is concentration in $x$. So what if I don't touch the matrix's sparsity at all, but first transform $x$ so it can't be spiky? I want a linear, norm-preserving map $R$ such that $Rx$ is "smooth" — small $\|Rx\|_\infty$ relative to $\|Rx\|_2$ — for every input, and then hit the smooth $Rx$ with a sparse projection. And there's a principle that says a transform *must* spread a spike: a signal and its spectrum can't both be concentrated, the Heisenberg uncertainty principle. A Fourier-type map applied to a delta will smear it out across all coordinates. That's the lever.

Which Fourier map? I want it real (no complex bookkeeping), orthogonal (so it preserves $\ell_2$ exactly and adds *zero* distortion of its own), and fast. The Walsh–Hadamard transform $H$ fits: it's the DFT over $\mathrm{GF}(2)^d$, $H_{ij}=d^{-1/2}(-1)^{\langle i-1,\,j-1\rangle}$, every entry $\pm d^{-1/2}$, $H$ orthogonal, and it factors recursively so I can apply it in $O(d\log d)$ by FFT. Good. Try $R=H$ and check it spreads a spike.

It doesn't, and I should have seen it: $H$ is a fixed deterministic matrix, so it has its own bad inputs. Hand it a column of $H$ itself and it maps to a coordinate vector — a spike. Determinism means an adversary picks the one input that re-concentrates. The patch is to randomize cheaply without losing orthogonality: insert a diagonal sign matrix $D$, $D_{ii}=\pm1$ i.i.d., and use $R=HD$. $D$ is orthogonal, $H$ is orthogonal, so $HD$ is orthogonal — $\|HDx\|_2=\|x\|_2$ exactly, still zero distortion. And now no fixed $x$ is adversarial: the randomness of $D$ smooths *whatever* came in.

Let me actually prove $HD$ smooths, because the whole construction rests on it. Fix a unit $x$ and let $u=HDx$. Look at one coordinate $u_1=\sum_i a_i x_i$. The first row of $H$ contributes $\pm d^{-1/2}$, and $D$ multiplies each $x_i$ by an independent sign, so effectively $a_i=\pm d^{-1/2}$ with the signs i.i.d. uniform. This is a signed sum; I want its tail. Since $d\,a_i=\pm\sqrt d$, the convenient quantity to take the moment generating function of is $d\,u_1=\sum_i d\,a_i x_i$:
$$\mathbb{E}\,e^{t\,d\,u_1}=\prod_i \mathbb{E}\,e^{t\,d\,a_i x_i}=\prod_i\cosh\!\big(t\sqrt d\,x_i\big)\le \prod_i e^{t^2 d\,x_i^2/2}=e^{t^2 d\,\|x\|_2^2/2}=e^{t^2 d/2},$$
using $\cosh z\le e^{z^2/2}$ and $\|x\|_2=1$. Markov on $e^{t d\,u_1}$ with the symmetric tail, set $t=s$:
$$\Pr[\,|u_1|\ge s\,]\le 2\,\mathbb{E}[e^{s d\,u_1}]/e^{s^2 d}\le 2 e^{s^2 d/2 - s^2 d}=2 e^{-s^2 d/2}.$$
I want this below $1/(20\,n d)$ so I can union-bound over all $n d$ coordinates of all $n$ vectors I'll embed; that wants $s=\Theta(d^{-1/2}\sqrt{\log n})$. Conclusion: with probability $\ge 1-1/20$,
$$\max_{x\in X}\|HDx\|_\infty = O\!\big(d^{-1/2}\sqrt{\log n}\big).$$
So after preconditioning, every vector is smooth: its mass is spread, $\|u\|_\infty$ is tiny, and the spike that killed the naïve sparse matrix can't occur.

Now the sparse projection $P$ onto the smooth $u$. Let $P$ be $k\times d$ with each entry independently $0$ with probability $1-q$ and otherwise $N(0,q^{-1})$. The variance $q^{-1}$ is deliberate — it keeps $\mathbb{E}\|Pu\|_2^2$ correctly normalized after thinning by $q$, and Gaussian entries give me $2$-stability to work with. What's the smallest $q$ I can get away with? Each output coordinate $y_1=\sum_j r_j b_j u_j$ (indicator $b_j$ for "entry present," $r_j\sim N(0,q^{-1})$), and conditioning on which entries survive, $y_1$ is Gaussian by $2$-stability of the normal: $(y_1\mid Z=z)\sim N(0,q^{-1}z)$ where $Z=\sum_j b_j u_j^2$. So $\mathbb{E}[Z]=\sum_j u_j^2\,\mathbb{E}[b_j]=q$, and the behavior of the estimator is governed by how tightly $Z$ — a sum of $u_j^2$ over the surviving coordinates — concentrates around $q$.

This is where smoothness earns its keep. $Z$ is a weighted sum of independent indicators with weights $u_j^2$. Its fluctuation is largest when the weight piles onto few coordinates; with $\|u\|_\infty\le s$ the weight vector $u^2$ lives in the polytope $\mathcal P=\{a:0\le a_j\le 1/m,\ \sum_j a_j=1\}$ with $m=s^{-2}$. The function $\mathbb{E}[Z^t]$ is convex in $u^2$, so it's maximized at a *vertex* of $\mathcal P$ — and a vertex is exactly the spread-out worst case $u^*=(m^{-1/2},\dots,m^{-1/2},0,\dots,0)$ with $m$ equal coordinates. There $Z^*\sim B(m,q)/m$, a scaled binomial, with $\mathrm{var}(Z^*)=q(1-q)/m$. Because smoothness forced $m$ to be large, that variance is small. Pushing the binomial moment bound through, with the smoothness premise in force, every row's $Z_i$ stays in $[q/2,2q]$ and $\sum_{i=1}^k Z_i$ concentrates, so
$$\Pr\!\Big[\big|\,\textstyle\sum_i y_i^2 - k\,\big| > \varepsilon k\Big]\le e^{-\Omega(\varepsilon^2 k)},$$
and at $k=c\,\varepsilon^{-2}\log n$ this is $1-1/\mathrm{poly}(n)$. For the $\ell_2$ target this fixes the density at $q=\min\{\Theta(\log^2 n / d),\,1\}$ — the smallest $q$ for which the per-row binomial still concentrates on the worst smooth input; any sparser and the variance on $u^*$ blows the bound.

Put the three pieces together: $\Phi=PHD$. $D$ flips signs, $H$ spreads, $P$ projects sparsely. The merits read off as a single statement — with probability $\ge 2/3$, for all $x$ in my set, $(1-\varepsilon)\alpha_p\|x\|_2^p\le\|\Phi x\|_p^p\le(1+\varepsilon)\alpha_p\|x\|_2^p$ with $\alpha_2=k$ (and $\alpha_1=k\sqrt{2\pi^{-1}}$ for the $\ell_1$ version, where $|y_1|$ has the half-normal mean). If I want the usual norm-preserving $\ell_2$ image, I rescale the output by $k^{-1/2}$. Repeat $O(\log(1/\delta))$ times to drive $2/3$ up to $1-\delta$.

Now the payoff, the runtime. $Dx$ is $O(d)$. $H(Dx)$ is the Walsh–Hadamard FFT, $O(d\log d)$. $P(HDx)$ costs $O(|P|)$ where $|P|$ is the number of nonzeros of $P$, distributed $B(dk,q)$, so $\mathbb{E}|P|=O(\varepsilon^{p-4}\log^{p+1}n)$ (taking $q=\Theta(\varepsilon^{p-2}\log^p n/d)$ in general). Total:
$$O\!\big(d\log d + \min\{\,d\varepsilon^{-2}\log n,\ \varepsilon^{p-4}\log^{p+1}n\,\}\big).$$
For $\ell_2$ that's $O(d\log d+\min\{d\varepsilon^{-2}\log n,\ \varepsilon^{-2}\log^3 n\})$; for $\ell_1$, $p=1$, it collapses to $O(d\log d + \varepsilon^{-3}\log^2 n)$. Either way it beats the $O(d\varepsilon^{-2}\log n)$ of the dense product whenever $k$ is not too close to $d$. The dense matrix-vector multiply — the wound I started from — is gone, replaced by an FFT plus a touch of sparse arithmetic. I'll call this the Fast Johnson–Lindenstrauss Transform.

But it doesn't touch my second regime, and I should be honest about that: $\Phi x$ pays $\Omega(d\log d)$ no matter how sparse $x$ is, because $HD$ touches every coordinate. For a streaming update $x\leftarrow x+v\,e_i$ I'd recompute $\Phi e_i$ at $\Omega(d\log d)$, which is worse than the dumb $O(k)$. Preconditioning bought me speed on dense vectors at the cost of destroying input sparsity. So for sparse and streaming inputs I need a *genuinely sparse matrix*, no preconditioner — back to the strand I abandoned, but armed with what I learned about why it failed.

Let me reconsider the sparse-matrix line carefully. I want a $k\times d$ matrix $S$ with exactly $s$ nonzeros per column and a *linear* $\ell_2$ estimator $\|Sx\|_2$. I have to say "linear" because there's a tempting cheat I must refuse: CountSketch sets $s=1$ and recovers the norm by a *median* of independent estimates. The median is gorgeous and sparse, but it's nonlinear — it is not an embedding into $\ell_2$ at all. The moment I want nearest-neighbor search in the reduced space, or approximate regression, or to train a classifier by SGD (which needs a differentiable estimator), the median is useless. So I'm committed to an actual linear image, which means I must *pay for collisions* rather than median them away.

The state of the art on this is the DKS construction and its sharper analyses: replicate each coordinate $s$ times preserving the norm, hash the copies to $s$ targets *with replacement* with random signs, and read off the linear $\|Sx\|_2$. It gets down to $s=\tilde O(\varepsilon^{-1}\log^2(1/\delta))$ — the first $o(k)$ column sparsity that's a real $\ell_2$ embedding — but it needs the input nice, $\|x\|_\infty=O(\sqrt\varepsilon)$, via a block-Hadamard preconditioner, and the original sampling view burns $O(ds\log k)$ random bits, fatal for streaming. And that extra $\log(1/\delta)$ in the sparsity nags at me. Where does the $\log^2$ come from when the target feels like $\varepsilon^{-1}\log(1/\delta)$?

Let me hunt for the waste by writing the error cleanly. Take $S_{i,j}=\eta_{i,j}\sigma_{i,j}/\sqrt s$ with $\sigma$ Rademacher signs and $\eta_{i,j}$ the indicator that entry $(i,j)$ is nonzero (exactly $s$ per column). For unit $x$, since the map is linear I can assume $\|x\|_2=1$ and it suffices to control $|\,\|Sx\|_2^2-1\,|>2\varepsilon-\varepsilon^2$. Expand:
$$Z:=\|Sx\|_2^2-1=\frac1s\sum_{r=1}^k\sum_{i\ne j}\eta_{r,i}\eta_{r,j}\,\sigma_{r,i}\sigma_{r,j}\,x_i x_j.$$
The diagonal terms sum to $\|x\|_2^2=1$ and cancel; what's left is exactly the cross terms — the *collisions*. So error $=$ collisions, precisely. Now look at DKS's "with replacement." Replication plus hashing with replacement means a single coordinate's own $s$ copies can land in the same target — a coordinate colliding *with itself*, within its own column. Picture the hard input $x=(1,0,\dots,0)$: if $q$ of the $s$ copies of $x_1$ pile into one target coordinate with agreeing signs, that coordinate contributes $q^2/s$ to the squared norm while those copies only account for $q/s$ of the true mass — an error $\sim q^2/s$ from a single column's self-collisions. Working that out, the DKS scheme *requires* $s=\Omega(\varepsilon^{-1}\lceil\log^2(1/\delta)/\log^2(1/\varepsilon)\rceil)=\tilde\Omega(\varepsilon^{-1}\log^2(1/\delta))$; the $\log^2$ is forced by its own within-column collisions. So I can't tune DKS down — I have to change the construction.

The diagnosis hands me the fix. The wasteful collisions are a coordinate against *itself*. Forbid them: hash each column's $s$ nonzeros *without replacement*, so every column has exactly $s$ distinct target rows and no self-collision is possible. I can pick the $s$ targets without replacement, which is a bipartite graph with $d$ left vertices, $k$ right vertices, left-degree $s$, and $S$ as its signed incidence matrix over $\sqrt s$; that is the graph construction. Or I can split $[k]$ into $s$ contiguous blocks of size $k/s$ and drop one nonzero per block; that is the block construction, structurally CountSketch, except I keep the linear $\ell_2$ estimator instead of a median and I only need $O(\log(1/\delta))$-wise independent hashes, which are cheap to build.

In both, $\sum_i\eta_{i,j}=s$ exactly per column, and within a row the $\eta$ are *negatively correlated* across columns — knowing one nonzero sits in a row makes others less likely. That's the whole asymptotic gain in one structural change; now let me prove it actually buys the better $s$.

The cleanest route uses the fact that $Z$ is a quadratic form in the signs. Write $Z=\sigma^T T\sigma$, where $T$ is block-diagonal with $k$ blocks $T_r$, $(T_r)_{i,j}=\eta_{r,i}\eta_{r,j}x_i x_j/s$ for $i\ne j$ and zero on the diagonal, so $\mathrm{tr}\,T=0$. A quadratic form in Rademachers around its trace is exactly what Hanson–Wright controls:
$$\mathbb{E}\,|\sigma^T T\sigma|^\ell\le C^\ell\max\{\sqrt\ell\,\|T\|_F,\ \ell\,\|T\|_2\}^\ell.$$
I just need the two norms. For Frobenius,
$$\|T\|_F^2=\frac1{s^2}\sum_{i\ne j}x_i^2 x_j^2\Big(\sum_{r=1}^k\eta_{r,i}\eta_{r,j}\Big).$$
The inner sum $\sum_r\eta_{r,i}\eta_{r,j}$ counts the rows where columns $i$ and $j$ both have a nonzero — the number of places they can collide. If I can guarantee this is $O(s^2/k)$ for every pair, then $\|T\|_F^2\le O(1/k)\sum_{i\ne j}x_i^2 x_j^2\le O(1/k)\|x\|_2^4=O(1/k)$. For the operator norm, each block $T_r=(1/s)(S_r-D_r)$ with $S_r=uu^T$ ($u_i=\eta_{r,i}x_i$) rank-one PSD of norm $\|u\|_2^2\le\|x\|_2^2=1$, and $D_r$ diagonal with $\|D_r\|_2\le\|x\|_\infty^2\le1$; both PSD, so $\|T\|_2\le(1/s)\max\{1,1\}=1/s$. Feed these in with $\ell=\log(1/\delta)$:
$$\Pr[\,|Z|>2\varepsilon-\varepsilon^2\,]\le(2\varepsilon-\varepsilon^2)^{-\ell}\,\mathbb{E}\,Z^\ell\le C^\ell\max\Big\{O(\varepsilon^{-1})\sqrt{\tfrac\ell k},\ (2\varepsilon-\varepsilon^2)^{-1}\tfrac\ell s\Big\}^\ell,$$
and this is $\le\delta$ once $k=\Theta(\varepsilon^{-2}\log(1/\delta))$ and $s=\Theta(\varepsilon^{-1}\log(1/\delta))$. There it is — the target sparsity, with the optimal $k$.

Notice what the proof actually used: it never needed the $\eta$ to be random — only the deterministic condition $\sum_r\eta_{r,i}\eta_{r,j}=O(s^2/k)$ for every pair $i\ne j$. That is a statement about the *columns of $S$*: no two columns share nonzero rows in more than $O(s^2/k)$ places. For the block construction, writing $C_i\in[k/s]^s$ for column $i$'s target-per-block, it says $C_i,C_j$ agree in at most $O(s^2/k)$ blocks — i.e. $\{C_i\}$ is an error-correcting code of relative distance $1-O(s/k)$. For the graph construction, the columns are codewords of a constant-weight binary code with minimum distance $2s-O(s^2/k)$. And this code condition is not a convenience — it's *necessary*: if two columns collided in many rows, an adversary spreads $x$'s mass equally on those two coordinates and forces a large error with good probability over the signs. So a good code is exactly necessary and exactly sufficient. The catch: to *get* such a code by the probabilistic method I need $s=\Omega(\varepsilon^{-1}\sqrt{\log(d/\delta)\log(1/\delta)})$, slightly above my target by a $\sqrt{\log_{1/\delta}(1/\varepsilon)}$ factor. The bottleneck is procuring the code, not the embedding.

To shave that last factor I drop the deterministic code and let the hashes be random of high enough independence, then bound $\mathbb{E}\,Z^\ell$ from first principles — and here Hanson–Wright doesn't simplify anything, so I go combinatorial. Let $Z_r=\sum_{i\ne j}\eta_{r,i}\eta_{r,j}\sigma_{r,i}\sigma_{r,j}x_i x_j$, so $Z=\frac1s\sum_r Z_r$, and first bound the $t$-th moment of one row. Expand
$$Z_r^t=\sum_{\substack{i_1\ne j_1,\dots,i_t\ne j_t}}\ \prod_{u=1}^t\eta_{r,i_u}\eta_{r,j_u}\,x_{i_u}x_{j_u}\,\sigma_{r,i_u}\sigma_{r,j_u},$$
and associate each monomial with a directed multigraph: a vertex for each distinct index, and for each $u$ an edge labeled $u$ from the vertex of $i_u$ to that of $j_u$. Take expectations. Over the signs, any vertex of *odd* degree carries some $\sigma$ an odd number of times and kills the term — so only graphs with all degrees even survive, $t$ edges, $v\in[2,t]$ vertices. Over the $\eta$, within a fixed row $r$ the $\eta_{r,\cdot}$ are independent across columns (negative correlation only hurts me, which I'll use later for the *product over rows*), so $\mathbb{E}_\eta\prod\eta=(s/k)^v$ — one factor $s/k$ per distinct vertex. Thus
$$\mathbb{E}\,Z_r^t=\sum_{G}\Big(\frac sk\Big)^{v}\!\!\sum_{\text{monomials}\to G}\ \prod_u x_{i_u}x_{j_u}.$$
Now the counting. For a graph $G$ with degrees $d_1,\dots,d_v$, the coefficient of $\prod_u x_{a_u}^{d_u}$ in $(\|x\|_2^2)^t=1$ is the multinomial $\binom{t}{d_1/2,\dots,d_v/2}$, while the number of monomials mapping to that pattern for a fixed $G$ is at most $v!$. Pass to vertex-labeled graphs $\mathcal G'_t$ (a canonical labeling by first-visit order makes the $v!$ relabelings distinct), and use $t!\ge t^t/e^t$ together with $\prod(d_u/2)!\le 2^{-t}\prod\sqrt{d_u}^{\,d_u}$ to reach
$$\mathbb{E}\,Z_r^t\le (e/2)^t\sum_{v=2}^t\Big(\frac sk\Big)^v\frac1{t^t}\sum_{G\in\mathcal G'_t}\prod_u\sqrt{d_u}^{\,d_u}.$$
The remaining graph sum I bound by a clean induction. Define $S_i(a_1,\dots,a_v)=\sum_{G}\prod_u\sqrt{a_u}^{\,d'_u}$ over all $i$-edge labeled graphs on $v$ vertices; adding one edge from $u$ to $w$ multiplies by at most $\sum_{u\ne w}\sqrt{a_u}\sqrt{a_w}\le(\sum_u\sqrt{a_u})^2\le(\sum_u a_u)\,v$ by Cauchy–Schwarz, so $S_t\le(\sum_u a_u)^t v^t$. With $\sum_u d_u=2t$ this gives $S_t\le(2tv)^t$, and summing over the $<2^t$ degree sequences,
$$\mathbb{E}\,Z_r^t\le (2e)^t\sum_{v=2}^t\Big(\frac sk\Big)^v v^t.$$
The summand $(s/k)^v v^t$ peaks at $v=\max\{2,\,t/\ln(k/s)\}$, so
$$\mathbb{E}\,Z_r^t\le t(2e^2)^t\cdot\begin{cases}(s/k)^2 & t<2\ln(k/s)\\[2pt](t/\ln(k/s))^t & \text{otherwise,}\end{cases}$$
which I can loosen uniformly to $\mathbb{E}\,Z_r^t\le t(2e^3)^t(s/k)^2 t^t$.

Now assemble the full moment $\mathbb{E}\,Z^\ell$. Expand $Z^\ell=s^{-\ell}(\sum_r Z_r)^\ell$ into products over distinct rows $r_1<\dots<r_q$ with multiplicities $\ell_i\ge2$ (single powers vanish since $\mathbb{E}\,Z_r=0$) summing to $\ell$. Across distinct rows the $\eta$ are *negatively correlated*, so $\mathbb{E}\prod_i Z_{r_i}^{\ell_i}\le\prod_i\mathbb{E}\,Z_{r_i}^{\ell_i}$ term by term — exactly where the without-replacement structure pays off, letting me factor the expectation. Insert the per-row bound, then Stirling and AM–GM ($\prod_i\ell_i\le(\ell/q)^q\le\binom\ell q<2^\ell$, $\ell_i!\ge e(\ell_i/e)^{\ell_i}$, $\ell!\le(\ell+1)((\ell+1)/e)^\ell$), and count $\binom kq$ ways to choose the rows and $<2^\ell$ partitions:
$$\mathbb{E}\,Z^\ell\le\Big(\frac{8e^3(\ell+1)}s\Big)^\ell(\ell+1)\sum_{q=1}^{\ell/2}\Big(\frac{s^2}{qk}\Big)^q.$$
The tail sum peaks at $q=\max\{1,s^2/(ek)\}$, contributing $e^{q}\le e^{\ell/2}$. Choose $\ell=\Theta(\log(1/\delta))$ even, $s=8e^4\sqrt e(\ell+1)/(2\varepsilon-\varepsilon^2)=\Theta(\varepsilon^{-1}\log(1/\delta))$, and $k=2s^2/(e\ell)=\Theta(\varepsilon^{-2}\log(1/\delta))$ — that last choice exactly makes $s^2/(ek)\le\ell/2$ so the peak sits inside the range. Then $\mathbb{E}\,Z^\ell\le(2\varepsilon-\varepsilon^2)^\ell\delta$, and Markov on $Z^\ell$ gives $\Pr[\,|Z|>2\varepsilon-\varepsilon^2\,]<\delta$. The without-replacement constructions hit $s=\Theta(\varepsilon^{-1}\log(1/\delta))$ at the optimal $k$, no code needed, and the block version samples from an $O(\log(1/\delta)\log d)$-bit seed using $O(\log(1/\delta))$-wise hashing — which finally answers the streaming complaint that buried DKS.

And $s=\Theta(\varepsilon^{-1}\log(1/\delta))$ is not an artifact of the proof for these two without-replacement schemes. Run the same hard inputs the other direction: when $s\le1/(2\varepsilon)$, a vector with $t=\lfloor1/(s\varepsilon)\rfloor$ equal entries gets a single collision of error $2/(ts)\ge2\varepsilon$ with probability $\Omega(1/\log(1/\delta))\gg\delta$; when $s>1/(2\varepsilon)$, the vector $(1/\sqrt2,1/\sqrt2,0,\dots)$ produces $2s\varepsilon$ sign-agreeing collisions, hence error $2\varepsilon$, with probability above $\delta$ unless $s=\Omega(\varepsilon^{-1}\log(1/\delta))$. So the graph and block constructions cannot beat this sparsity.

Let me close the loop on the whole thing. The dense JL multiply was the bottleneck of every pipeline that used it, and the two ways it hurt called for two different cures. For dense vectors, a sparse matrix alone fails because it distorts a spike, so I precondition with a randomized Walsh–Hadamard $HD$ — orthogonal, hence distortion-free, and by the uncertainty principle it provably smooths every input in $X$ to $\|HDx\|_\infty=O(d^{-1/2}\sqrt{\log n})$ — and *then* a sparse Gaussian $P$ concentrates on the now-smooth vector, giving $\Phi=PHD$ in $O(d\log d+{\rm sparse})$ time. For sparse and streaming vectors, where a preconditioner would destroy the very sparsity I'm exploiting, I keep a genuinely sparse matrix and notice that error equals collision; DKS's $\log^2$ is forced by self-collisions from hashing with replacement, so I hash *without* replacement — graph or block — turning the columns into an error-correcting code, and a Hanson–Wright bound (with the code condition) plus a sharper monomial-to-graph moment bound (with random hashes) both certify the JL guarantee at the optimal $k=\Theta(\varepsilon^{-2}\log(1/\delta))$ with column sparsity $s=\Theta(\varepsilon^{-1}\log(1/\delta))$, which the same hard inputs show is tight for these constructions.

```python
import numpy as np

# ---- Fast-JL: Phi = P H D, for dense inputs (precondition then sparse-project) ----
def fwht(a):
    # In-place Walsh-Hadamard transform, O(d log d); d must be a power of two.
    a = a.astype(float).copy()
    h = 1
    n = len(a)
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x, y = a[j], a[j + h]
                a[j], a[j + h] = x + y, x - y
        h *= 2
    return a / np.sqrt(n)            # normalized so H is orthogonal

class FastJL:
    def __init__(self, d, k, n, eps=1.0, p=2, seed=0):
        assert d & (d - 1) == 0, "d must be a power of two for the Hadamard FFT"
        assert p in (1, 2)
        rng = np.random.default_rng(seed)
        self.d, self.k = d, k
        self.D = rng.choice([-1.0, 1.0], size=d)            # random signs: randomize H
        logn = np.log(max(n, 2))
        q = min((eps ** (p - 2)) * (logn ** p) / d, 1.0)    # l2: log^2 n / d
        alpha = k if p == 2 else k * np.sqrt(2.0 / np.pi)
        self.output_scale = alpha ** (-1.0 / p)             # normalize the p-th-power theorem
        # P: k x d, each entry 0 w.p. 1-q else N(0, 1/q); store only nonzeros.
        mask = rng.random((k, d)) < q
        self.rows, self.cols = np.nonzero(mask)
        self.vals = rng.standard_normal(len(self.rows)) / np.sqrt(q)

    def apply(self, x):
        u = fwht(self.D * x)        # D then H: orthogonal, smooths ||u||_inf ~ d^-1/2 sqrt(log n)
        y = np.zeros(self.k)
        np.add.at(y, self.rows, self.vals * u[self.cols])
        return self.output_scale * y

# ---- Sparse-JL block construction: exactly s nonzeros/column, no within-column collision ----
class SparseJLBlock:
    def __init__(self, d, k, s, seed=0):
        assert k % s == 0
        rng = np.random.default_rng(seed)
        self.d, self.k, self.s, self.block = d, k, s, k // s
        # one target per block (without replacement across blocks) + a random sign, per column
        self.rows = np.empty((d, s), dtype=int)
        self.signs = rng.choice([-1.0, 1.0], size=(d, s))
        for j in range(d):                                  # limited-independent hashes can replace this
            for b in range(s):
                self.rows[j, b] = b * self.block + rng.integers(self.block)

    def apply(self, x):
        y = np.zeros(self.k)
        nz = np.nonzero(x)[0]
        for j in nz:                                        # cost scales with nnz(x): streaming-friendly
            y[self.rows[j]] += self.signs[j] * x[j] / np.sqrt(self.s)
        return y
```
