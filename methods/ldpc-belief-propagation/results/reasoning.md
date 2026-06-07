Let me start from where it actually hurts. Shannon told us, back in 1948, that for any rate below capacity there exist codes of growing block length whose probability of error goes to zero. Elias sharpened it in 1955: he pinned down matching upper and lower bounds on the best achievable error probability for length-n codes on the binary symmetric channel, showed they fall off exponentially in n for every rate up to capacity, and — this is the part that keeps me up — showed that almost *every* long random linear code is essentially as good as the very best. Most codes are good codes. And yet I cannot use a single one of them. The proof is an average over an ensemble; it hands me no code I can write down, and worse, even if it did, I would have no way to decode it. To decode optimally I must, given the received vector, find the codeword that maximizes the likelihood — and a long random code has on the order of 2^k codewords. Comparing against all of them, or equivalently solving the nearest-codeword problem for an unstructured linear code, costs me something exponential in the dimension. So the situation is precisely this: the codes the theory promises are abundant and excellent and undecodable, and the codes I can actually decode — the short algebraic ones, Hamming, BCH, Reed–Solomon — fall short of capacity, because for every explicit construction I know the ratio of minimum distance to block length drifts to zero as n grows. Structure that lets me decode seems to cost me distance; distance seems to demand a randomness that destroys any handle for decoding. That is the knot I have to cut.

Let me look hard at what decoding a linear code even *is*, because I think the generator-matrix picture is hiding the structure I need. A codeword is any c with H c^T = 0, where H is the parity-check matrix, m rows by n columns. Each row of H is one parity-check equation: it names a subset of the n positions and asserts their modulo-2 sum is zero. So a codeword is a vector that satisfies all m of these little local constraints simultaneously. Decoding is then: given a noisy y, find the assignment closest to y that satisfies every parity check. Now here is what I never paid attention to. If H is a typical dense matrix — say its entries are independent fair coin flips, the very ensemble Elias blesses — then a single parity check involves roughly n/2 of the bits. Knowing that one such check is satisfied or violated tells me almost nothing about any individual bit; the information is smeared across half the block. Every constraint is entangled with every bit. There is no locality anywhere to exploit. That is *why* decoding the good random code is hard: not in spite of its richness but because of it.

So invert the wish. What if a parity check touched only a handful of bits? Suppose I insist that every column of H has just j ones — each bit sits in only j checks — and every row has just k ones — each check ties together only k bits, with j and k small fixed constants, not growing with n. Then the total number of ones in H is jn, which is also km, and it grows *linearly* in n, not as n². The matrix is almost all zeros. A single parity check now constrains k bits, a tiny local neighborhood, and a single bit feels only j checks. Suddenly there is locality: each check is a small, almost-isolated piece of evidence about its k bits, and most of the matrix is empty. The cost of even looking at every constraint once is proportional to the number of ones, which is O(n). Whatever decoder I build, if its work per bit is bounded and it sweeps the nonzeros a constant number of times, it runs in linear time in the block length. That alone is worth chasing.

But I have to confront the obvious objection immediately, because it's the same objection that sank every other structured construction: does forcing H to be this sparse make the code *weak*? If a sparse parity-check matrix gives me a code whose minimum distance grows only like log n, or worse stays constant, I've bought cheap decoding and thrown away the thing that made the code good. I cannot answer this for one matrix — a single long code is intractable to analyze, there are too many codewords. But I can do what Shannon and Elias did: define an ensemble of these sparse matrices and average. Let me build one I can actually reason about. Take j submatrices stacked vertically, each of size (n/k) by n. In the first submatrix, put the ones in consecutive runs: row i carries its k ones in columns (i−1)k+1 through ik, so every column has exactly one one and every row has exactly k. That first block is rigid and trivial. Now form the other j−1 blocks by taking that first block and permuting its columns at random, independently, each permutation equally likely. Stack them. Every column of the whole matrix has exactly j ones (one per block), every row exactly k. The randomness lives only in the permutations, but it's enough to make the code random-like while keeping the degrees pinned at j and k. The rate is at least 1 − m/n = 1 − j/k, a hair higher actually because the blocks are not all independent, but call it 1 − j/k.

Now the distance question, over this ensemble. I want to know the typical minimum distance — the smallest weight of a nonzero codeword — as a function of n. The way to get a handle on it is to count candidate nonzero words of weight w, estimate the probability that a random choice of the j−1 permutations makes every check see even parity on those w positions, and sum that expected count over w up to δn. For fixed j ≥ 3 and k > j, the exponent in that union bound is negative for all sufficiently small positive δ, so the expected number of nonzero codewords below δ_jk n goes to zero. Markov's inequality then says the fraction of codes in the ensemble whose minimum distance is below that fixed fraction δ_jk of the block length collapses to zero as n grows. The minimum-distance-to-length ratio settles on a *positive* constant. The codes are asymptotically good — distance growing linearly with n — which is exactly the property every explicit algebraic construction failed to keep. So sparse does not mean weak. That is the first thing that surprised me: I expected to pay for sparsity in distance, and I don't, as long as j ≥ 3.

And j ≥ 3 is not cosmetic; I should see why. Picture j = 2: every bit sits in exactly two checks. Think of each check as a vertex and lay an edge for each bit between the two checks it touches — the code becomes a graph, and low-weight codewords correspond to cycles in that graph. A random graph with as many edges as vertices is riddled with short cycles, so j = 2 floods the code with low-weight codewords and the distance grows only logarithmically. Bumping to j = 3 is what tips the ensemble over into linear minimum distance. So I'll keep j ≥ 3, and rate 1 − j/k, picking k to set the rate — j = 3, k = 6 gives me rate one-half.

Good. I have a sparse, random-like, asymptotically-good code that costs O(n) just to write down its constraints. The whole gamble was that sparsity buys a decoder. Time to build it.

Let me try the crudest possible idea first. Compute all m parity checks on the received hard bits. A bit that's wrong will tend to violate the checks it sits in. Because each check has only k bits and errors are rare, most checks contain at most one error, so a violated check is a fairly clean accusation against its members. If a particular bit is sitting in an unusually large number of violated checks — more than some threshold — flip it, recompute the checks, repeat until nothing is violated. This actually works on the BSC at low enough error rates, and it shows me the engine I want: unsatisfied checks are *evidence*, and a bit gathers evidence from the few checks it lives in. But it's crude. It throws away how *strong* each piece of evidence is, it makes hard flips, and on a soft channel like the Gaussian one I'd have to threshold to bits first and lose information — and I already know thresholding a Gaussian channel into a BSC throws away capacity; the per-symbol likelihood ratio carries strictly more than the sign of the received value. I want a decoder that works in probabilities, not flips.

So let me decode one bit properly and see if the locality makes it tractable. Fix a bit, call it d. I want the probability that the transmitted bit at position d is a 1, given everything the channel told me. The channel gives me, for d and for every other bit, an a-posteriori probability from its received symbol. The constraints linking them are the parity checks. Lay them out as a tree rooted at d: d is the root; the j checks containing d are the first branches; each such check has k−1 other bits, which form the first tier of nodes; each of *those* bits sits in further checks, whose other bits form the second tier; and so on. As long as I never revisit a bit, every node in this tree is a distinct position with its own independent channel observation. That independence is the whole game, so I hold onto it.

Consider one check sitting just above d, with d and k−1 other bits in it. The check is satisfied exactly when an even number of its bits are 1. I need: given the channel probabilities of those k−1 other bits, what does this check say about d? So I first need the probability that those k−1 bits contain an even number of ones. Let bit l be 1 with probability P_l, independently. I claim the probability that an even number of them are 1 is (1 + ∏(1 − 2P_l))/2. Let me actually get this, because it's the heart of the whole thing. Form the generating polynomial ∏(1 − P_l + P_l t). Expand it: the coefficient of t^i is exactly the probability of i ones. Now form the same product with t replaced by −1 inside, ∏(1 − P_l − P_l) … no, more precisely evaluate the product at t = 1 and at t = −1. At t = 1 the product is ∏(1 − P_l + P_l) = 1, the total probability, sum of all coefficients. At t = −1 the product is ∏(1 − 2P_l), and that's the alternating sum of the coefficients — even-index terms with a plus, odd with a minus. Add the two: every odd coefficient cancels, every even one doubles. So the sum of even-index coefficients is (1 + ∏(1 − 2P_l))/2. That is precisely the probability of an even number of ones. Clean. That single product ∏(1 − 2P_l) is the object I'll carry around; note 1 − 2P_l is just the expected sign of bit l, the difference between its probability of being 0 and being 1.

Now use it. Condition on d. If d = 0, the check is satisfied iff the other k−1 bits have an even number of ones, probability (1 + ∏(1 − 2P_l))/2 over the k−1 of them. If d = 1, the check needs the other k−1 to be odd, probability (1 − ∏(1 − 2P_l))/2. The j checks on d are, in the tree, conditionally independent given d, so the probability that *all* j checks are satisfied is the product of these per-check factors. Bayes-combine that with d's own channel probability and the j checks' verdicts, and out drops the a-posteriori probability that d = 1 given the channel reading at d and the readings of all the bits one tier up. That's the single-tier rule. I can write it as a ratio and it's a product over the j checks, each check itself a product over its k−1 members of those (1 − 2P) factors — manageable, and entirely local to d's neighborhood.

The expression is a product of products, and it's begging to be linearized, so let me move to log-likelihood ratios. Write each bit's belief as a log-ratio L = log(P(0)/P(1)); the sign is the tentative bit, the magnitude is the confidence. Two things become pleasant. First, the bit's combination step — fold the channel reading together with the j check verdicts — turns into a *sum* of log-ratios, because conditionally independent evidence multiplies in probability and therefore adds in log. Second, the check step. That per-check product of (1 − 2P_l) terms: 1 − 2P_l, written in terms of the log-ratio of bit l, is exactly tanh(L_l/2). So the check's outgoing log-ratio is governed by a *product of tanh's* of the incoming half-log-ratios, run back through an inverse-tanh. Concretely the check sends out 2·atanh(∏ tanh(L_in/2)), or equivalently log((1 + ∏ tanh(L_in/2))/(1 − ∏ tanh(L_in/2))). The whole decoder is then two alternating local operations on the sparse graph: at each check, take the bits' current log-ratios, tanh them, multiply, atanh — a "horizontal" step across the k members of a row; at each bit, add the channel log-ratio to the incoming check log-ratios — a "vertical" step down the j entries of a column. Both touch only nonzeros of H, so a full sweep is O(jn) = O(n).

There is one subtlety I must get exactly right or the tree-independence I leaned on silently breaks. When a check computes the message it sends *to* bit d, it must use the other k−1 bits but *not* d's own current belief; and when bit d computes the message it sends back *up* a particular check, it must sum the channel reading and the verdicts of its *other* j−1 checks, excluding the one it's answering. If I let d's belief flow into a check and then straight back to d, I'd be feeding d its own opinion as if it were fresh external evidence — double-counting, and the independence used in the lemma would be a lie. So every message is extrinsic: each outgoing message along an edge omits the information that came in along that same edge. With that exclusion, the single-tier computation is exactly the conditional-independence calculation I derived, no fudge.

Now extend past one tier. The single-tier rule gave me, for each first-tier bit, an improved belief computed from the second tier. Feed those improved beliefs back as the priors for the first-tier bits and run the rule again to update d. By induction this propagates information down from tier m to the root, and after m sweeps the root's belief is its a-posteriori probability given all received symbols out to tier m of the tree. The cost per bit per sweep is independent of block length — it depends only on j and k — so the total per sweep is linear in n. And the number of *independent* tiers I can climb grows like the logarithm of n, because each tier multiplies the node count by (j−1)(k−1). Logarithmically many sweeps, each linear: this is the cheap decoder I was hunting for.

And here is the wall I knew was coming. The tree is a fiction. After about log n tiers, the branching has touched more bits than exist in the block, so the tree must close on itself — the same bit reappears in two places, and the independence that justified multiplying the per-check probabilities is gone. For any code long enough to be worth using, the dependencies set in while m is still small, long before the beliefs have converged. So strictly, my derivation is only valid for a handful of sweeps. Do I throw the method out?

No — and the reason is exactly the property I built into H. The graph is *sparse*, with j and k tiny, which means the shortest cycle through any bit is long; the local neighborhood of any bit really does look like a tree out to many tiers. So the dependencies, when they finally arrive, come in through long loops, they're weak, and they tend to cancel rather than reinforce. The honest move is to just keep iterating past the point where the tree assumption holds and treat the residual correlation as small. There's a second reason this is safe: after m−1 honest sweeps, every bit's equivocation has already dropped — the partly-resolved beliefs are like a cleaner received sequence, easier to decode than the original — so even if sweep m is no longer exact, it's operating on an easier problem and still pulls beliefs toward the truth. The global, intractable maximum-likelihood inference has been replaced by local probability propagation that is *exact on the tree the sparse graph locally resembles*, and approximately right because sparsity keeps that resemblance good for many tiers, and it only works because I made H sparse in the first place. Sparsity wasn't only for linear-time bookkeeping; it's what makes the local approximation converge. The two motives — cheap arithmetic and accurate local inference — turn out to be the same motive.

So I stop iterating not at a fixed count but on a signal the structure hands me for free: after each sweep, take the sign of every bit's current log-ratio as a tentative decoding, and check the syndrome. If H x = 0, x is a genuine codeword and I'm done — I don't need to know it's the right one in any deeper sense, a satisfied syndrome is the certificate. If not, sweep again, up to some cap, and if the cap is hit I declare a failure rather than risk a wrong codeword. Crucially the decoder works straight off the channel's soft log-ratios — on a Gaussian channel the input log-ratio is just proportional to the received value, so I never threshold and never throw away the soft information that hard-decision decoders discard.

Let me assemble it concretely and pin down the channel arithmetic. Build H as the stack of j permuted blocks. Get a systematic generator G from H by Gauss-Jordan over GF(2) so I can actually produce codewords. Map a codeword to ±1 by 0 → +1, 1 → −1, send over additive Gaussian noise of variance σ². The per-bit channel log-ratio is L_c = 2y/σ² — that's log(P(bit=0|y)/P(bit=1|y)) for this mapping, the soft input the decoder runs on. Initialize each edge's bit-to-check message to the bit's channel log-ratio. Then loop: the horizontal check step computes each check-to-bit message as the atanh-of-product-of-tanh over the *other* bits in that check; the vertical bit step sets each bit-to-check message to the channel log-ratio plus the sum of incoming check messages *except* the one on that edge; form the posterior log-ratio as channel plus the sum of *all* incoming check messages, take its sign for the hard bit, and test the syndrome; stop when it's zero.

```python
import numpy as np

def parity_check_matrix(n, d_v, d_c, rng):
    """Sparse H: stack d_v column-permutations of a consecutive-ones block.
    Each column has d_v ones (bit in d_v checks), each row d_c ones (check ties
    d_c bits); ones = d_v*n = O(n). This is the random-like, asymptotically-good
    sparse construction (need d_v >= 3 for linearly growing minimum distance)."""
    assert n % d_c == 0 and d_c > d_v >= 2
    m = n * d_v // d_c
    rows = m // d_v
    block = np.zeros((rows, n), dtype=int)
    for i in range(rows):
        block[i, i * d_c:(i + 1) * d_c] = 1        # consecutive runs of d_c ones
    H = [block]
    for _ in range(1, d_v):
        H.append(rng.permutation(block.T).T)        # random column permutation
    return np.vstack(H)

def systematic_generator(H):
    """G (k x n) with H G^T = 0 over GF(2), by Gauss-Jordan. So c = uG is a
    codeword, syndrome H c^T = 0; nothing about the code beyond linearity here."""
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
    P = A[:rank, order][:, rank:]                   # H' = [I | P]
    k = n - rank
    G = np.zeros((k, n), dtype=int)
    G[:, :rank] = P.T                               # G' = [P^T | I]
    G[:, rank:] = np.eye(k, dtype=int)
    G = G[:, np.argsort(order)]                     # undo the column permutation
    assert not ((G @ H.T) % 2).any()
    return G

def bp_decode(H, Lc, maxiter=100):
    """Log-domain sum-product on the sparse graph. Lc = channel log-ratios.
    Two alternating local sweeps, each touching only nonzeros of H => O(n)."""
    m, n = H.shape
    bits_in_check = [np.where(H[i] == 1)[0] for i in range(m)]   # row supports
    checks_on_bit = [np.where(H[:, j] == 1)[0] for j in range(n)]  # column supports
    Lq = np.zeros((m, n))                            # bit -> check messages
    Lr = np.zeros((m, n))                            # check -> bit messages
    for i in range(m):
        Lq[i, bits_in_check[i]] = Lc[bits_in_check[i]]  # init with channel log-ratios
    for _ in range(maxiter):
        # horizontal: each check answers each bit using the OTHER bits (extrinsic).
        # 1 - 2P = tanh(L/2); a check is satisfied iff an even number of 1s
        # (Lemma 1), so the message is atanh of the product of the others' tanh.
        for i in range(m):
            ni = bits_in_check[i]
            t = np.tanh(0.5 * Lq[i, ni])
            for j in ni:
                prod = np.clip(np.prod(t[ni != j]), -1 + 1e-12, 1 - 1e-12)
                Lr[i, j] = np.log((1 + prod) / (1 - prod))
        # vertical: each bit adds its channel log-ratio and the OTHER checks'
        # verdicts (extrinsic) -- independent evidence adds in the log domain.
        for j in range(n):
            mj = checks_on_bit[j]
            s = Lr[mj, j].sum()
            for i in mj:
                Lq[i, j] = Lc[j] + s - Lr[i, j]
        # posterior = channel + ALL incoming check messages; sign is the bit.
        L = Lc + np.array([Lr[checks_on_bit[j], j].sum() for j in range(n)])
        x = (L < 0).astype(int)
        if not ((H @ x) % 2).any():             # satisfied syndrome = codeword: done
            return x, True
    return x, False

if __name__ == "__main__":
    rng = np.random.RandomState(0)
    n, d_v, d_c = 96, 3, 6                       # rate ~ 1 - 3/6 = 1/2
    H = parity_check_matrix(n, d_v, d_c, rng)
    G = systematic_generator(H)
    k = G.shape[0]
    print("rate  snr(dB)  raw_bit_err  decoded_bit_err  fail")
    for snr_db in [1.0, 2.0, 3.0, 4.0]:
        raw = dec = fail = 0
        for s in range(40):
            r = np.random.RandomState(s)
            u = r.randint(0, 2, k)
            c = (u @ G) % 2                      # H c^T = 0
            sigma2 = 10 ** (-snr_db / 10) / (2 * (k / n))
            y = (1 - 2 * c) + np.sqrt(sigma2) * r.randn(n)   # BPSK over AWGN
            Lc = 2 * y / sigma2                  # channel log-ratio, soft input
            raw += int(((y < 0).astype(int) != c).sum())
            xhat, ok = bp_decode(H, Lc)
            dec += int((xhat != c).sum())
            fail += (0 if ok else 1)
        print(f"{k/n:.3f}   {snr_db:>4.0f}     {raw:>7d}        {dec:>8d}     {fail:>3d}")
```

Walking the whole chain back: capacity-achieving codes have to be long and random-like, but a long random code is intractable to decode because its dense parity checks entangle every bit with every constraint. Make the parity checks *sparse* — each bit in a few checks, each check on a few bits — and three things fall out at once: the matrix is O(n) to handle, the ensemble's minimum distance still grows linearly with n so the code stays good (as long as each bit is in at least three checks), and, because sparsity means long cycles, every bit's local neighborhood looks like a tree. On a tree the a-posteriori bit probabilities are computed exactly by a local rule — a parity check is satisfied iff an even number of its bits are 1, which in log-likelihood form is a product of tanh's at each check and a sum at each bit, with every message kept extrinsic so the tree-independence stays honest. Iterate that local message passing on the real (loopy but locally-tree-like) graph, accept the weak residual correlations because the loops are long, and stop when the syndrome is satisfied; a globally intractable decoding problem has become O(n) per iteration.
