The problem I set out to solve is the gap between what coding theory promises and what I can actually build. Shannon's 1948 theorem guarantees that for any rate $R < C$ there exist codes whose probability of error vanishes as the block length $n$ grows, and Elias's 1955 refinement is sharper still: he pinned down matching upper and lower bounds on the best achievable error probability on the binary symmetric channel, showed they fall off exponentially in $n$ for every rate up to capacity, and proved that almost *every* long random linear code is essentially as good as the very best. Most codes are good codes. And yet I cannot use a single one of them. The proofs are averages over an ensemble; they hand me no code I can write down, and worse, even if they did, I would have no way to decode it. Optimal decoding means finding, for a received vector $y$, the codeword maximizing $P(y\mid c)$ — and a long random code of dimension $k$ has on the order of $2^k$ codewords, so the nearest-codeword search is exponential in the dimension. The codes I *can* decode go the other way: the short algebraic families — Hamming, BCH, Reed–Solomon — admit bounded-distance algebraic decoders, but for every explicit construction I know the ratio of minimum distance to block length drifts to zero as $n$ grows, so they fall well short of capacity at the long block lengths capacity demands. Convolutional codes with sequential decoding are the dominant practical scheme, but their throughput collapses above the computational cut-off rate $R_0 < C$; threshold decoding is linear-cost but weak. Structure that lets me decode seems to cost distance; distance seems to demand a randomness that destroys any handle for decoding. That is the knot I have to cut.

The way out comes from looking at *why* the good random code is hard to decode. A codeword is any $c$ with $H c^{\mathsf T} = 0$, where each row of the parity-check matrix $H$ is one parity equation naming a subset of positions whose modulo-2 sum must vanish. If $H$ is the dense ensemble Elias blesses — entries independent fair coin flips — a single check involves roughly $n/2$ bits, so knowing one check is satisfied tells me almost nothing about any individual bit; every constraint is entangled with every bit and there is no locality to exploit. So I invert the wish, and I propose Low-Density Parity-Check codes decoded by belief propagation (the sum-product algorithm). I insist that $H$ be *sparse*: every column carries exactly $j$ ones (each bit sits in $j$ checks) and every row exactly $k$ ones (each check ties $k$ bits), with $j,k$ small fixed constants. Then the total number of ones is $jn = km$, growing linearly in $n$, the matrix is almost all zeros, and each check is a small, almost-isolated piece of evidence about its $k$ bits. Any decoder that sweeps the nonzeros a constant number of times runs in $O(n)$ time.

The immediate objection is the one that sank every other structured construction: does forcing sparsity make the code weak? I answer it the way Shannon and Elias did, by averaging over an ensemble I can reason about. Take $j$ submatrices stacked vertically, each $\tfrac{n}{k}\times n$. In the first, put the ones in consecutive runs — row $i$ holds its $k$ ones in columns $(i-1)k+1$ through $ik$ — so every column has one one and every row has $k$. Form the other $j-1$ blocks by independent uniformly-random column permutations of that base block and stack them; now every column has exactly $j$ ones, every row exactly $k$, and the randomness lives only in the permutations. The rate is $R \approx 1 - j/k$. Counting candidate nonzero words of weight $w$ and bounding the probability that the random permutations make every check see even parity on them, the union-bound exponent is negative for all small enough $\delta > 0$ when $j \ge 3$ and $k > j$, so the expected number of nonzero codewords below $\delta_{jk}\, n$ vanishes; by Markov, the fraction of codes whose minimum distance falls below that fixed fraction of $n$ collapses as $n \to \infty$. The minimum-distance-to-length ratio settles on a *positive* constant — the codes are asymptotically good, linear minimum distance, exactly the property the algebraic families lost. And $j \ge 3$ is essential, not cosmetic: at $j = 2$ each bit joins exactly two checks, so drawing an edge between the two checks a bit touches turns the code into a graph whose low-weight codewords are cycles; a random graph with as many edges as vertices teems with short cycles, so $j = 2$ gives only logarithmic distance. I keep $j \ge 3$ and set $k$ for the rate — $j = 3$, $k = 6$ gives rate one-half.

The whole gamble was that sparsity buys a decoder, and it does, through a single combinatorial identity. Fix a bit $d$ and lay out the checks and bits around it as a tree rooted at $d$: the $j$ checks containing it branch first, each carrying $k-1$ other bits, those bits sit in further checks, and so on; as long as no bit is revisited, every node is a distinct position with its own independent channel observation, and that independence is the entire engine. Consider one check above $d$ with $k-1$ other bits, bit $l$ equal to one with probability $P_l$. The check is satisfied iff an even number of its bits are one, and the probability of an even number of ones is
$$\frac{1 + \prod_l (1 - 2P_l)}{2}.$$
This drops out of the generating polynomial $\prod_l(1 - P_l + P_l t)$, whose $t^i$ coefficient is the probability of $i$ ones: at $t = 1$ the product is $1$ (all coefficients summed), at $t = -1$ it is $\prod_l(1-2P_l)$ (even coefficients minus odd); adding the two cancels every odd term and doubles every even one. The object I carry around is $\prod_l(1-2P_l)$, and $1 - 2P_l$ is just the expected sign of bit $l$. Conditioning on $d$: if $d = 0$ the other $k-1$ bits must be even, probability $(1+\prod)/2$; if $d=1$ they must be odd, probability $(1-\prod)/2$. The $j$ checks on $d$ are conditionally independent given $d$, so multiplying their per-check factors and Bayes-combining with $d$'s own channel reading yields the a-posteriori probability of $d$ given the channel data one tier up.

Because the rule is a product of products, I move to log-likelihood ratios $L = \log\!\big(P(0)/P(1)\big)$, where the sign is the tentative bit and the magnitude the confidence. Two things become clean. Conditionally independent evidence multiplies in probability and so *adds* in the log domain, turning the bit's combination step into a sum. And $1 - 2P_l$ written in terms of $L_l$ is exactly $\tanh(L_l/2)$, so the check step is a product of $\tanh$'s run back through an inverse $\tanh$. The decoder is then two alternating local sweeps over the sparse graph. The horizontal (check $\to$ bit) message is
$$L_{r,\,m\to n} = 2\,\operatorname{atanh}\!\Big(\!\prod_{n' \in N(m)\setminus n} \tanh\!\tfrac{L_{q,\,n'\to m}}{2}\Big) = \log\frac{1+\prod}{1-\prod},$$
and the vertical (bit $\to$ check) message is
$$L_{q,\,n\to m} = L_c(n) + \sum_{m' \in M(n)\setminus m} L_{r,\,m'\to n},$$
with the posterior $L_n = L_c(n) + \sum_{m\in M(n)} L_{r,\,m\to n}$ and hard decision $x_n = [\,L_n < 0\,]$. The one subtlety I must get exactly right is that every message is *extrinsic*: the message a check sends to bit $d$ uses the other $k-1$ bits but never $d$'s own belief, and the message $d$ sends back up a check sums the channel reading and the verdicts of its *other* $j-1$ checks. If $d$'s belief flowed into a check and straight back, I would feed $d$ its own opinion as fresh external evidence — double-counting, and the conditional-independence the lemma assumes would be a lie. With the exclusion enforced, each local computation is exactly the tree calculation I derived.

There is one wall I knew was coming: the tree is a fiction. After about $\log n$ tiers the branching touches more bits than exist, so the tree closes on itself and the independence behind multiplying the per-check probabilities is gone. But this is precisely where sparsity earns its keep a second time. With $j$ and $k$ tiny the shortest cycle through any bit is long, so each bit's local neighborhood really does look like a tree out to many tiers; the dependencies, when they finally arrive, come through long loops, are weak, and tend to cancel. So I keep iterating past the point where the tree assumption is exact and treat the residual correlation as small — safe also because after several honest sweeps every bit's equivocation has already dropped, so later sweeps operate on an easier, partly-resolved problem and still pull beliefs toward the truth. Sparsity was never only for cheap arithmetic; it is what makes the local approximation converge, and the two motives turn out to be one. I stop not at a fixed count but on a certificate the structure hands me for free: after each sweep I take the sign of every posterior log-ratio as a tentative decoding and test the syndrome — if $Hx = 0$, $x$ is a genuine codeword and I am done; otherwise I sweep again up to a cap and declare failure rather than risk returning a wrong codeword. The decoder runs straight off the channel's soft log-ratios — on a binary-input AWGN channel with BPSK mapping ($0\to+1$, $1\to-1$) and noise variance $\sigma^2$, the channel input is $L_c = 2y/\sigma^2$ — so I never threshold and never throw away the soft information a hard-decision decoder discards. The globally intractable maximum-likelihood problem has become local probability propagation, $O(jn) = O(n)$ per iteration, with the number of useful iterations growing like $\log n$.

```python
import numpy as np

def parity_check_matrix(n, d_v, d_c, rng):
    """Sparse regular H: stack d_v column-permutations of a consecutive-ones block.
    Column weight d_v (bit in d_v checks), row weight d_c (check on d_c bits)."""
    assert n % d_c == 0 and d_c > d_v >= 2
    m = n * d_v // d_c
    rows = m // d_v
    block = np.zeros((rows, n), dtype=int)
    for i in range(rows):
        block[i, i * d_c:(i + 1) * d_c] = 1
    H = [block]
    for _ in range(1, d_v):
        H.append(rng.permutation(block.T).T)
    return np.vstack(H)

def systematic_generator(H):
    """G (k x n) with H G^T = 0 over GF(2), via Gauss-Jordan."""
    m, n = H.shape
    A = H.copy() % 2
    r = 0
    for c in range(n):
        piv = np.where(A[r:, c] == 1)[0]
        if not len(piv):
            continue
        A[[r, piv[0] + r]] = A[[piv[0] + r, r]]
        for rr in range(m):
            if rr != r and A[rr, c]:
                A[rr] ^= A[r]
        r += 1
        if r == m:
            break
    rank = r
    pivot_cols = [int(np.where(A[i] == 1)[0][0]) for i in range(rank)]
    nonpivot = [c for c in range(n) if c not in pivot_cols]
    order = pivot_cols + nonpivot
    P = A[:rank, order][:, rank:]
    k = n - rank
    G = np.zeros((k, n), dtype=int)
    G[:, :rank] = P.T
    G[:, rank:] = np.eye(k, dtype=int)
    G = G[:, np.argsort(order)]
    assert not ((G @ H.T) % 2).any()
    return G

def bp_decode(H, Lc, maxiter=100):
    """Log-domain sum-product. Lc = per-bit channel log-likelihood ratios."""
    m, n = H.shape
    bits_in_check = [np.where(H[i] == 1)[0] for i in range(m)]
    checks_on_bit = [np.where(H[:, j] == 1)[0] for j in range(n)]
    Lq = np.zeros((m, n)); Lr = np.zeros((m, n))
    for i in range(m):
        Lq[i, bits_in_check[i]] = Lc[bits_in_check[i]]
    for _ in range(maxiter):
        for i in range(m):                                   # horizontal (check->bit)
            ni = bits_in_check[i]
            t = np.tanh(0.5 * Lq[i, ni])
            for j in ni:
                prod = np.clip(np.prod(t[ni != j]), -1 + 1e-12, 1 - 1e-12)
                Lr[i, j] = np.log((1 + prod) / (1 - prod))
        for j in range(n):                                   # vertical (bit->check)
            mj = checks_on_bit[j]; s = Lr[mj, j].sum()
            for i in mj:
                Lq[i, j] = Lc[j] + s - Lr[i, j]
        L = Lc + np.array([Lr[checks_on_bit[j], j].sum() for j in range(n)])
        x = (L < 0).astype(int)
        if not ((H @ x) % 2).any():
            return x, True
    return x, False

if __name__ == "__main__":
    rng = np.random.RandomState(0)
    n, d_v, d_c = 96, 3, 6
    H = parity_check_matrix(n, d_v, d_c, rng)
    G = systematic_generator(H); k = G.shape[0]
    for snr_db in [1.0, 2.0, 3.0, 4.0]:
        raw = dec = fail = 0
        for s in range(40):
            r = np.random.RandomState(s)
            u = r.randint(0, 2, k); c = (u @ G) % 2
            sigma2 = 10 ** (-snr_db / 10) / (2 * (k / n))
            y = (1 - 2 * c) + np.sqrt(sigma2) * r.randn(n)
            Lc = 2 * y / sigma2
            raw += int(((y < 0).astype(int) != c).sum())
            xhat, ok = bp_decode(H, Lc); dec += int((xhat != c).sum()); fail += (0 if ok else 1)
        print(f"rate={k/n:.3f} snr={snr_db:.0f}dB raw_bit_err={raw} decoded_bit_err={dec} fail={fail}")
```
