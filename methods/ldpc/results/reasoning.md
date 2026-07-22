Let me start from the thing that actually bothers me. Shannon settled the existence question two decades ago: pick a code by drawing 2^{nR} length-n binary strings at random, decode each received word to the most likely codeword, and as n grows the average error probability over that random ensemble drops to zero, exponentially, for every rate R below the channel capacity C. So good long codes exist — in fact almost all random codes are good. The problem is that "decode to the most likely of 2^{nR} codewords" means scanning an exponentially large list. A random code is structureless by construction; there is nothing to exploit, so maximum-likelihood decoding is the only option and it costs 2^{nR} comparisons. Precisely in the regime where the theorem promises the codes get good — n large — the decoder becomes unrunnable. So I don't actually want a random code. I want a code with just enough structure that I can decode it cheaply, while keeping the error behavior of the random ensemble. The whole task is to find that structure.

What kind of structure makes decoding cheap? The most flexible handle I have is linearity over GF(2): define the code as all x with Hx = 0, where H is a parity-check matrix and each row is one parity constraint — a subset of bit positions forced to sum to zero mod 2. Linear is good because membership is a single matrix product and encoding is linear algebra. But linearity alone doesn't help decoding: a generic H still defines a code whose maximum-likelihood decoding is exactly the intractable search. I haven't bought anything yet. Decoding, written honestly, is an inference problem — given the channel's noisy readout y, find the x with Hx = 0 that maximizes the posterior P(x | y). The channel hands me, per bit, a soft likelihood: on the binary symmetric channel a crossover probability, on the Gaussian channel a density around ±1. Optimal use of that soft information over the constraint Hx = 0 is the thing that's expensive. So the question sharpens: what property of H would let me do this inference without the exponential search?

Inference gets cheap only if everything does not depend on everything else. When each constraint touches many variables and each variable sits in many constraints, the dependencies tangle and I can't decompose the problem. So make the dependencies *sparse*. Let H have very few 1s: each column carries a small fixed number j of ones (each bit appears in exactly j parity checks) and each row carries a small fixed number k of ones (each parity check involves exactly k bits), with j and k small constants independent of n. Call this an (n, j, k) low-density parity-check code. Most of H is zeros; the 1s are a thin scattering. The number of checks is the number of rows; counting 1s two ways, jn = k·(#rows), so there are jn/k checks and the rate is R ≥ 1 − j/k (≥ because some of those rows can turn out linearly dependent, which only raises the rate). The bet I'm making is that sparsity is exactly what will make the inference local and therefore cheap. I have to check that bet doesn't cost me the error performance.

First the distance, because that's the classical worry. The error-correcting power of a code is summarized by its minimum distance D, the fewest bit positions in which two codewords differ; a nearest-codeword decoder corrects up to ⌊(D−1)/2⌋ flips. I can't compute D for one enormous sparse code, but I can average over an ensemble of random parity-check matrices and ask for the *typical* distance. Fix a nonzero binary pattern of weight ℓ. For a generic random parity-check code of rate R there are n(1−R) checks, and the pattern satisfies each independent check with probability exactly ½ — because flip the last 1-position in the pattern and the check's parity flips, so satisfied/unsatisfied are equiprobable regardless of the other positions. So the expected number of weight-ℓ codewords is N̄(ℓ) = (n choose ℓ) · 2^{−n(1−R)}. Put λ = ℓ/n and run Stirling on the binomial: (n choose λn) ≈ exp[n H(λ)] / sqrt(2πnλ(1−λ)) with H(λ) = −λ ln λ − (1−λ) ln(1−λ). So N̄(λn) ≈ exp{ n[H(λ) − (1−R) ln 2] } up to the polynomial prefactor. The exponent is negative for small λ and crosses zero at the δ_0 solving H(δ_0) = (1−R) ln 2 — that's the Gilbert bound. For any fixed δ < δ_0, summing over ℓ ≤ δn still leaves an exponentially small expectation, so the probability that the minimum distance is below δn vanishes. The typical code has minimum distance growing *linearly* with n. Distance was never the bottleneck; it's the decoder.

Now does the *low-density* restriction wreck that? Redo the count for the (n, j, k) ensemble — same flavor of calculation but with the column weight pinned to j — and the conclusion holds for j ≥ 3: minimum distance still grows linearly with n. But j = 2 is special and bad. With column weight 2 each bit sits in exactly two checks, so the bits-and-checks graph is just a collection of cycles, and a chain of bits around a cycle can form a low-weight codeword; the minimum distance grows only *logarithmically* with n. So I need j ≥ 3 to get linear distance growth. Good — that pins down the smallest useful column weight, and it's a real constraint, not a convenience. The honest cost of going low-density is that these codes are not distance-optimal for a given block length, and the maximum rate at which they're usable is bounded a little below capacity. But if the payoff is a decoder that runs in time linear in n, that small rate sacrifice is a bargain. Let me go find the decoder, because that's the whole point of sparsity.

Start with the crudest thing that could possibly work, to build intuition. Make a hard decision on every bit from the channel, compute all the parity checks, and look at which fail. Here's where sparsity already pays. Suppose exactly one transmitted bit is wrong. It sits in j checks, and since H is sparse and the 1s barely overlap, every one of those j checks now fails, while any *other* bit shares at most one check with the wrong bit, so no other bit is in more than one failed check. The culprit lights up: it's the bit that participates in an unusually large number of failed checks. So the rule writes itself — flip any bit that is contained in more than some fixed number of unsatisfied checks, recompute the checks with the flipped values, and repeat until all checks are satisfied. With small parity-check sets most checks see zero or one error, so a bit drowning in failed checks is overwhelmingly likely to be the error. This bit-flipping decoder is dead simple and its cost per bit per iteration doesn't grow with n. But it only uses hard decisions and only really works on the BSC at rates well below capacity — it throws away the soft a-posteriori information the channel gave me, and that information is exactly what the algebraic codes were also forced to discard. I want the soft version.

So: instead of flipping bits, I want to *propagate probabilities*. For a given bit d, I want the probability that the transmitted value was 1, conditioned not just on d's own channel readout but on all the other received bits reachable through the parity checks — because those bits, through their shared checks, carry information about d. Picture the structure as a tree rooted at d (this is exactly what the sparsity buys me, and I'll have to come back and check the "tree" claim). Tier 1: the j checks that contain d, and hanging off each check the other k−1 bits in that check. Tier 2: the *other* checks those tier-1 bits sit in, and the bits in those, and so on. Each check is a parity constraint, so it ties d's value to the values of the other bits under it. If I can compute, within an ensemble where the bits hanging in the tree are independent, the probability that d is 1 given all the received symbols out to some tier, I'll have a soft decoder. Let me try to derive that probability for one tier and see if it iterates.

I need one combinatorial fact first, about a single parity check. A check on bit d is satisfied (given d's value) exactly when the *other* bits under it have the right parity. So I need: given a bunch of independent bits where bit ℓ is 1 with probability P_ℓ, what is the probability that an *even* number of them are 1? Let me build the generating function ∏_ℓ (1 − P_ℓ + P_ℓ t). Expand it: the coefficient of t^i is precisely the probability that exactly i of the bits are 1, because each factor contributes either "0 with prob 1−P_ℓ" (the constant term) or "1 with prob P_ℓ" (the t term). Now form the twin product ∏_ℓ (1 − P_ℓ − P_ℓ t) — identical except every t carries a minus sign, so in the expansion all the *odd* powers of t flip sign and the even powers are unchanged. Add the two products: odd terms cancel, even terms double. Set t = 1 and divide by 2 and I've isolated exactly the even-power sum, which is the probability of an even number of 1s:

  Pr[even number of ones] = [ ∏_ℓ (1 − P_ℓ + P_ℓ) + ∏_ℓ (1 − P_ℓ − P_ℓ) ] / 2 = [ 1 + ∏_ℓ (1 − 2P_ℓ) ] / 2.

The first product is just 1 (each factor is 1), and the second is ∏_ℓ (1 − 2P_ℓ). So the even-parity probability is [1 + ∏(1−2P_ℓ)]/2, and by the same token the odd-parity probability is [1 − ∏(1−2P_ℓ)]/2. The whole content of a parity check, probabilistically, is carried by that single product ∏(1 − 2P_ℓ). That's a beautiful collapse — k−1 messy bits reduce to one product, and the quantity that matters per incoming bit is (1 − 2P_ℓ), the "bias" of that bit toward 0.

Now the one-tier posterior for d. Let P_d be the prior that d is 1 from its own channel readout, and P_{iℓ} the probability that the ℓ-th of the other bits in the i-th check on d is 1, from *its* channel readout. Let S be the event that all j parity checks on d are satisfied. I want the ratio Pr[x_d=0 | {y}, S] / Pr[x_d=1 | {y}, S], because a ratio lets the awkward normalization drop out. By Bayes, this equals (1−P_d)/P_d times Pr[S | x_d=0, {y}] / Pr[S | x_d=1, {y}]. If x_d = 0, then check i is satisfied exactly when an even number of its *other* k−1 bits are 1, so by the lemma Pr[check i satisfied | x_d=0] = [1 + ∏_{ℓ}(1−2P_{iℓ})]/2, and the checks are independent in this ensemble so Pr[S | x_d=0] = ∏_{i=1}^{j} ([1 + ∏_{ℓ=1}^{k−1}(1−2P_{iℓ})]/2). If x_d = 1, each check needs an *odd* number of other ones, giving ∏_{i=1}^{j} ([1 − ∏_{ℓ=1}^{k−1}(1−2P_{iℓ})]/2). The j factors of ½ cancel in the ratio, and I get

  Pr[x_d=0 | {y},S] / Pr[x_d=1 | {y},S] = (1−P_d)/P_d · ∏_{i=1}^{j} [ 1 + ∏_{ℓ=1}^{k−1}(1−2P_{iℓ}) ] / [ 1 − ∏_{ℓ=1}^{k−1}(1−2P_{iℓ}) ].

So one tier of soft decoding is: for each check, form the product of the (1−2P) biases of the k−1 *other* bits; that gives the check's verdict on d as a ratio; multiply those j verdicts together with d's own channel prior. The structure is exactly variable-side and check-side: the check turns biases into a parity verdict, the variable multiplies its incoming verdicts and its own prior.

How do I go past one tier? The P_{iℓ} I just used were raw channel priors. But each of those tier-1 bits is itself the root of its *own* tree of deeper checks, so I should replace its raw prior with the *refined* posterior I'd get by running the same one-tier formula on it — using its j checks, except I must *omit the check through which it is currently talking to d*. That omission is the crucial part, and let me make sure I see why. If I let the message a bit sends up into a check depend on what that same check just told it, I'd be feeding a node its own past belief back to itself and double-counting the evidence — the estimate would feed on itself and run away. By excluding the receiving check from the product, the message a bit sends along an edge depends only on *other*, independent evidence — the extrinsic information. So the rule is: along each edge, send the belief computed from everything *except* that edge. With that omission, tier by tier, the one-tier formula iterates: compute each bit's belief from its other checks, feed those up, compute the next tier's beliefs, and so on. After m passes, d's belief is conditioned on all received symbols out to tier m, and if it's converging, the belief on each bit drives toward 0 or 1.

Now I owe myself the check I deferred: is this really a tree, and for how long? The neighborhood of d branches by (k−1) bits per check and (j−1) further checks per bit as I go outward, so tier m contains on the order of [(j−1)(k−1)]^m distinct nodes. As long as that count stays below n, the outward branches reach *fresh* bits and checks — no node repeats — so the neighborhood genuinely is a tree and the independence assumption behind the lemma and Bayes step holds exactly. That stays true for m up to about log n / log[(j−1)(k−1)], i.e. logarithmically many tiers. This is precisely what sparsity bought: with j, k small constants the branching is slow, so the local graph looks like a tree out to a depth that grows with the block length, and on a tree this message passing computes the *exact* posterior marginals. The same sparsity makes each pass cheap — each of the jn/k checks does O(k) work and each of the n bits does O(j) work, so a pass is O(n), linear in the block length, with the per-bit cost independent of n. Linear-time, near-exact soft decoding — that is the thing I was hunting for when I decided to make H sparse.

But the tree can't go on forever. Once m exceeds that logarithmic depth the branches start colliding — the same bit reappears at two places in the "tree," which means the graph actually has a cycle and the two paths I treated as independent share a node. The independence assumption breaks; the beliefs I pass are no longer guaranteed exact. This is real, and it's the price of a finite graph: the shorter the cycles (the smaller the girth), the sooner the breakdown and the more biased the messages. The saving grace is that the dependencies appear only after many tiers, by which point each bit's belief has usually already been sharpened a great deal, and the residual dependencies tend to partly cancel; so I keep iterating past the tree depth and treat the result as a very good approximation rather than the exact marginal. The construction lesson is also clear: when I build H I should avoid short cycles, keep the girth large, so the tree approximation holds as long as possible.

Let me get the arithmetic into a form I'd actually run, because products of probabilities underflow and the (1±∏)/(1∓∏) ratios are clumsy. Work with the log-likelihood ratio L = ln(P(0)/P(1)) of each quantity. A ratio of posteriors becomes a *sum* in log space, which is exactly what I want for the variable node: its belief is the sum of its channel LLR and the LLRs coming in from its checks. Concretely, taking the log of the one-tier ratio, the variable-node update for d's outgoing message on a given edge is L_d (channel) plus the sum of the incoming check LLRs *except* the one on that edge — the extrinsic sum. Independent pieces of evidence about the same bit just add their LLRs; that's the whole variable-node rule.

The check node is the interesting one, because the product ∏(1−2P) has to become an LLR operation. Watch what 1−2P is in LLR terms. If L = ln((1−P)/P) for a bit (so P is its probability of being 1), then P = 1/(1+e^L), and

  1 − 2P = 1 − 2/(1+e^L) = (e^L − 1)/(e^L + 1) = tanh(L/2).

So each bit's bias 1−2P is just tanh of half its LLR. The check's product ∏(1−2P_ℓ) becomes ∏ tanh(L_ℓ/2). And the check's outgoing LLR is the log of [1 + ∏(1−2P)]/[1 − ∏(1−2P)], i.e.

  L_check = ln( (1 + ∏_ℓ tanh(L_ℓ/2)) / (1 − ∏_ℓ tanh(L_ℓ/2)) ) = 2 · artanh( ∏_ℓ tanh(L_ℓ/2) ),

with the product running over the k−1 *other* bits in the check. So the check-node rule is: push each incoming LLR through tanh(·/2), multiply, and pull the product back through 2·artanh. There's the soft analogue of a parity constraint — a smooth combiner that says "the least-confident incoming bit dominates the check's verdict," since tanh saturates and the product is small whenever any factor is small. (Equivalently, one can write the combiner with the self-inverse function f(β) = ln((e^β+1)/(e^β−1)) acting on the sum of f's of the magnitudes, carrying the signs separately; it's the same operation, since f is exactly the log of the tanh expression and applying it twice returns the argument.) The variable node adds LLRs; the check node combines them through tanh/artanh; both always omitting the edge they're sending on. That pair *is* the decoder.

Let me pin down the channel LLRs that seed it. On the BSC with crossover p, a received 0 gives L = ln((1−p)/p) and a received 1 gives the negative of that. On the Gaussian channel I send each bit antipodally as x = +1 for a 0 and x = −1 for a 1, receive y = x + noise of variance σ², and the channel LLR is ln[ N(y; +1, σ²) / N(y; −1, σ²) ] = (2/σ²) y, which is just a scaled copy of the matched-filter output. So the seed is L_c = 2y/σ², linear in the received sample — cheap.

Now the loop, assembled. Seed every bit with its channel LLR L_c. Horizontal (check) step: for each check and each bit in it, compute the outgoing LLR 2·artanh(∏ tanh(L/2)) over the *other* bits, where the L's are the messages those bits sent in (on the first pass, just their channel LLRs). Vertical (variable) step: for each bit and each check it's in, set the outgoing message to L_c plus the sum of incoming check LLRs from its *other* checks. After each round, form the full posterior LLR of each bit — its channel LLR plus *all* its incoming check LLRs (no omission now, since this is the final belief, not a message) — threshold it (positive LLR means bit 0, since L = ln P(0)/P(1)), and test whether the hard-decided word satisfies Hx = 0. If it does, I've landed on a valid codeword and I stop; if not, iterate, up to a cap. The Hx = 0 test is a real gift: a codeword is a fixed point of the constraints, so a valid decode is self-certifying, and if the cap is hit without one, the decoder *knows* it failed rather than silently returning garbage. Each round is O(n); the number of rounds needed grows only logarithmically with n. That's the linear-per-iteration, near-optimal soft decoder the whole construction was aimed at.

The matrix has to be regular enough that the analysis and the implementation stay simple: every column has exactly j ones, every row has exactly k ones, and the same small adder/combiner pattern repeats everywhere. I also cannot choose the cheapest possible j. Column weight 2 would lower the work, but it gave only logarithmic distance, so I keep j ≥ 3. The rate follows from the shape of H as R ≥ 1 − j/k.

Let me write it. First build a regular low-density H by Gallager's construction — stack j submatrices, each a permutation of a base block whose rows hold k consecutive 1s, so every column ends up with weight j and every row with weight k; the row weight has to divide n, and it must be larger than the column weight in this regular construction. Turn H into a generator G by Gaussian elimination over GF(2) so that x = Gv satisfies Hx = 0. Then the BPSK-over-AWGN channel, then the log-domain belief-propagation decoder, with the tanh check step, the summing variable step, and the Hx = 0 stopping test.

```python
import warnings
import numpy as np

def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, np.random.RandomState):
        return seed
    return np.random.RandomState(seed)

def binaryproduct(X, Y):
    return X.dot(Y) % 2

def gaussjordan(X, change=False):
    A = np.copy(X).astype(int)
    m, n = A.shape
    if change:
        P = np.identity(m, dtype=int)
    pivot_old = -1
    for j in range(n):
        pivot = np.argmax(A[pivot_old + 1:m, j]) + pivot_old + 1
        if A[pivot, j]:
            pivot_old += 1
            if pivot_old != pivot:
                A[[pivot_old, pivot]] = A[[pivot, pivot_old]]
                if change:
                    P[[pivot_old, pivot]] = P[[pivot, pivot_old]]
            for i in range(m):
                if i != pivot_old and A[i, j]:
                    A[i] = np.abs(A[i] - A[pivot_old])
                    if change:
                        P[i] = np.abs(P[i] - P[pivot_old])
        if pivot_old == m - 1:
            break
    return (A, P) if change else A

# ---- build a regular (n, j, k) low-density parity-check matrix -------------
def parity_check_matrix(n, d_v, d_c, seed=None):
    if d_v <= 1:
        raise ValueError("d_v must be at least 2.")
    if d_c <= d_v:
        raise ValueError("d_c must be greater than d_v.")
    if n % d_c:
        raise ValueError("d_c must divide n for a regular matrix.")
    rng = check_random_state(seed)
    n_equations = (n * d_v) // d_c          # jn/k checks
    block = np.zeros((n_equations // d_v, n), dtype=int)
    block_size = n_equations // d_v
    for i in range(block_size):             # base block: k consecutive 1s per row
        block[i, i*d_c:(i+1)*d_c] = 1
    H = np.empty((n_equations, n), dtype=int)
    H[:block_size] = block
    for i in range(1, d_v):                  # remaining j-1 blocks: column perms
        H[i*block_size:(i+1)*block_size] = rng.permutation(block.T).T
    return H                                  # weight j per column, k per row

def coding_matrix(H):
    n_equations, n = H.shape
    H_columns, tQ = gaussjordan(H.T, change=True)
    H_reduced = gaussjordan(H_columns.T)
    n_bits = n - H_reduced.sum()
    Y = np.zeros((n, n_bits), dtype=int)
    Y[n - n_bits:, :] = np.identity(n_bits, dtype=int)
    return binaryproduct(tQ.T, Y)

# ---- channel: send x = (-1)^bit over AWGN ---------------------------------
def encode(G, v, snr, seed=None):
    rng = check_random_state(seed)
    d = binaryproduct(G, v)
    x = (-1) ** d                            # bit 0 -> +1, bit 1 -> -1
    sigma = 10 ** (-snr / 20)
    return x + rng.randn(*x.shape) * sigma

def incode(H, x):
    return (binaryproduct(H, x) == 0).all()

# ---- log-domain belief propagation (sum-product) --------------------------
def decode(H, y, snr, maxiter=1000):
    y = np.asarray(y, dtype=float)
    m, n = H.shape
    var = 10 ** (-snr / 10)                   # noise variance sigma^2
    Lc = 2 * y / var                          # channel LLR, L = log P(0)/P(1)
    Lq = np.zeros((m, n))                      # variable -> check messages
    Lr = np.zeros((m, n))                      # check -> variable messages
    checks = [np.where(H[i])[0] for i in range(m)]   # bits in each check
    bits   = [np.where(H[:, j])[0] for j in range(n)]# checks on each bit

    for it in range(maxiter):
        # horizontal (check) step: tanh rule, omit the receiving bit
        for i in range(m):
            for j in checks[i]:
                src = Lq[i] if it > 0 else Lc          # messages in (or channel on pass 0)
                X = 1.0
                for jp in checks[i]:
                    if jp != j:
                        X *= np.tanh(0.5 * src[jp])
                num, denom = 1 + X, 1 - X
                if num == 0:
                    Lr[i, j] = -1.0
                elif denom == 0:
                    Lr[i, j] = 1.0
                else:
                    Lr[i, j] = np.log(num / denom)   # = 2*artanh(X)

        # vertical (variable) step: sum of channel + other checks, omit this check
        for j in range(n):
            for i in bits[j]:
                Lq[i, j] = Lc[j] + sum(Lr[ip, j] for ip in bits[j] if ip != i)

        # posterior belief and hard decision
        L_post = Lc + np.array([Lr[bits[j], j].sum() for j in range(n)])
        x = (L_post <= 0).astype(int)          # positive LLR -> bit 0
        if incode(H, x):                       # valid codeword: self-certified stop
            break
    else:
        warnings.warn("Decoding stopped before convergence.")
    return x
```
