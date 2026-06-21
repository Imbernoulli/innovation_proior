We want to send $k$ information bits as a longer block of $n$ bits over a noisy channel and recover them with vanishing error probability, at the highest possible rate $R = k/n$. The existence half of this is settled: Shannon showed that for any rate $R < C$ there are length-$n$ codes whose error probability falls to zero — exponentially in $n$ — and his proof simply draws the code at random, picking $2^{nR}$ codewords uniformly from the $2^n$ binary strings and decoding each received word to the most likely codeword. The trouble is entirely on the decoding side. A code drawn this way has no internal structure, so the only decoder is maximum likelihood: compare $y$ against all $2^{nR}$ codewords and keep the most probable. That cost grows exponentially in $n$, precisely the regime in which the theorem promises the codes get good. Capacity is achievable in principle by codes nobody can run.

So the real question is not whether good long codes exist but whether there is a family with just enough structure to decode cheaply — ideally linearly in $n$ per pass — while keeping near-capacity error performance and using the channel's soft per-bit information. The existing structured families each leave a gap. Algebraic bounded-distance codes (Hamming, BCH, Reed–Solomon) decode efficiently but operate on hard decisions — they compute a syndrome and correct any pattern of weight up to $t = \lfloor (D-1)/2\rfloor$, throwing away the channel's a-posteriori probabilities and abandoning every error past $t$, so at long lengths their performance sits well short of capacity. Sequential decoding of convolutional codes does use soft information but pays with a per-digit computation that is a random variable which occasionally explodes, and a practical rate ceiling $R_{\mathrm{comp}} < C$ above which the average work is unbounded. Threshold decoding is cheap but weak. None is a long block code that is simultaneously cheap to decode and able to exploit the full soft information near capacity.

I propose low-density parity-check (LDPC) codes with iterative belief-propagation (sum-product) decoding. Keep the flexible handle of linearity over GF(2) — define the code as all $x$ with $Hx = 0$, each row of $H$ one parity constraint — but make $H$ *sparse*: every column has a small fixed weight $j$ (each bit lies in $j$ checks) and every row a small fixed weight $k$ (each check binds $k$ bits), with $j,k$ small constants independent of $n$. Counting the 1s two ways gives $jn = k \cdot(\text{\#checks})$, so there are $jn/k$ checks and the rate is $R \ge 1 - j/k$. Sparsity is the entire bet: it is what makes the decoding inference local, hence cheap, and I have to check it does not cost the error performance.

Distance is the classical worry, and it survives. View $H$ as a bipartite graph of bit nodes and check nodes with an edge per 1. Averaging over a random parity-check ensemble, a fixed nonzero weight-$\ell$ pattern satisfies each of the $n(1-R)$ independent checks with probability exactly $\tfrac12$ — flip the pattern's last 1 and the check parity flips — so the expected number of weight-$\ell$ codewords is $\bar N(\ell) = \binom{n}{\ell}\,2^{-n(1-R)}$. Writing $\lambda = \ell/n$ and applying Stirling, $\bar N(\lambda n) \approx \exp\{n[H(\lambda) - (1-R)\ln 2]\}$ with $H(\lambda) = -\lambda\ln\lambda - (1-\lambda)\ln(1-\lambda)$; the exponent is negative until $\lambda$ reaches the $\delta_0$ solving $H(\delta_0) = (1-R)\ln 2$ (the Gilbert bound), so the typical minimum distance grows linearly in $n$. Redoing this with the column weight pinned to $j$, the same conclusion holds for $j \ge 3$. But $j = 2$ is special and bad: with column weight 2 the graph is a union of cycles, a chain around a cycle is a low-weight codeword, and the minimum distance grows only logarithmically. So $j \ge 3$ is a real constraint — the smallest useful column weight — not a convenience.

The decoder is where sparsity pays off. The crudest version already shows why: make hard decisions, compute all checks, and note that a single wrong bit makes all $j$ of its checks fail while any other bit shares at most one check with it, so the culprit lights up as the bit in an unusually large number of failed checks — flip it and repeat. That bit-flipping rule costs $O(1)$ per bit per iteration but uses only hard decisions. The soft version propagates probabilities instead. Root a tree at a bit $d$: tier 1 is its $j$ checks and the other $k-1$ bits under each, tier 2 the further checks those bits sit in, and so on. The key combinatorial fact about one check is the even-parity probability among independent bits where bit $\ell$ is 1 with probability $P_\ell$. Forming $\prod_\ell(1 - P_\ell + P_\ell t)$ the coefficient of $t^i$ is the probability of exactly $i$ ones; forming the twin $\prod_\ell(1 - P_\ell - P_\ell t)$ flips the sign of every odd power; adding them cancels the odd terms and doubles the even ones, and at $t=1$,
$$\Pr[\text{even \# of ones}] = \frac{1 + \prod_\ell(1 - 2P_\ell)}{2},$$
so a whole parity check collapses, probabilistically, into the single product $\prod_\ell(1 - 2P_\ell)$ of the bits' biases toward 0.

This gives the one-tier posterior for $d$. Let $P_d$ be $d$'s channel prior of being 1, $P_{i\ell}$ the prior of the $\ell$-th other bit in the $i$-th check, and $S$ the event that all $j$ checks on $d$ are satisfied. Taking the ratio $\Pr[x_d{=}0\mid\cdot]/\Pr[x_d{=}1\mid\cdot]$ so the normalization drops out, Bayes gives $(1-P_d)/P_d$ times the ratio of $\Pr[S\mid x_d]$. If $x_d = 0$ each check needs an even number of other ones, if $x_d = 1$ an odd number, the $j$ factors of $\tfrac12$ cancel, and
$$\frac{\Pr[x_d{=}0\mid y,S]}{\Pr[x_d{=}1\mid y,S]} = \frac{1-P_d}{P_d}\prod_{i=1}^{j}\frac{1 + \prod_{\ell=1}^{k-1}(1-2P_{i\ell})}{1 - \prod_{\ell=1}^{k-1}(1-2P_{i\ell})}.$$
To go past one tier, replace each raw prior $P_{i\ell}$ by the refined belief that bit would compute from its *own* other checks — but omitting the check through which it is currently talking to $d$. That omission is load-bearing: if a bit's outgoing message depended on what the receiving check just told it, the node would feed on its own past belief and the estimate would run away. By excluding the receiving edge, every message carries only *extrinsic*, independent evidence. With that, the one-tier formula iterates tier by tier. The tree assumption is exactly what sparsity buys: the neighborhood branches by $(k-1)(j-1)$ per step, so out to depth $\sim \log n / \log[(j-1)(k-1)]$ the branches reach fresh nodes, the graph is genuinely a tree, and on a tree this message passing computes the exact posterior marginals. Beyond that depth the branches collide — the same bit reappears, meaning a real cycle — and the independence breaks; the messages are then a very good approximation rather than exact, which is why one keeps iterating and why $H$ should be built with large girth so short cycles do not bias the messages early.

The arithmetic is run in log-likelihood ratios $L = \ln(P(0)/P(1))$, which turns products into sums and prevents underflow. A bit's bias is $1 - 2P = (e^L-1)/(e^L+1) = \tanh(L/2)$, so the check product becomes $\prod_\ell \tanh(L_\ell/2)$ and the check's outgoing LLR is the log of $(1+X)/(1-X)$ with $X$ that product, i.e.
$$L_{\text{check}} = 2\,\mathrm{artanh}\!\Big(\prod_\ell \tanh(L_\ell/2)\Big),$$
over the $k-1$ *other* bits. This is a smooth parity constraint in which the least-confident incoming bit dominates, since $\tanh$ saturates and the product is small whenever any factor is. The variable node is the trivial dual: independent evidence about the same bit just adds, so $d$'s outgoing message on an edge is its channel LLR plus the sum of the incoming check LLRs *except* the one on that edge. The seed LLRs are the channel readouts — on the BSC with crossover $p$, $\pm\ln((1-p)/p)$; on the AWGN channel with antipodal signalling $x = (-1)^{\text{bit}}$ and received $y = x + \text{noise}$ of variance $\sigma^2$, $L_c = 2y/\sigma^2$, a scaled matched-filter output. The loop alternates a horizontal (check) step and a vertical (variable) step, both always omitting the edge they send on; after each round the full posterior $L_{\text{post}}[j] = L_c[j] + \sum_{i}L_{i\to j}$ (no omission now) is thresholded — positive means bit 0 — and the hard word is tested against $Hx = 0$. A codeword is a fixed point of the constraints, so a valid decode is self-certifying and the decoder stops; if the cap is hit without one, it reports a *detected* failure rather than silent garbage. Each round is $O(n)$ and the number of rounds grows only logarithmically in $n$ — the linear-time, near-optimal soft decoder the whole construction was aimed at. The matrix itself is built by Gallager's regular construction: stack $j$ column-permuted copies of a base block whose rows hold $k$ consecutive 1s, giving every column weight $j$ and every row weight $k$, with $k$ dividing $n$ and $k > j$; Gaussian elimination over GF(2) then turns $H$ into a generator $G$ with $x = Gv$.

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

def parity_check_matrix(n, d_v, d_c, seed=None):
    """Regular (n, d_v, d_c) LDPC matrix by Gallager's construction:
    stack d_v column-permuted blocks of a base block with d_c consecutive 1s/row."""
    if d_v <= 1:
        raise ValueError("d_v must be at least 2.")
    if d_c <= d_v:
        raise ValueError("d_c must be greater than d_v.")
    if n % d_c:
        raise ValueError("d_c must divide n for a regular matrix.")
    rng = check_random_state(seed)
    n_equations = (n * d_v) // d_c               # = d_v * n / d_c checks
    block_size = n_equations // d_v
    block = np.zeros((block_size, n), dtype=int)
    for i in range(block_size):
        block[i, i*d_c:(i+1)*d_c] = 1            # d_c consecutive 1s
    H = np.empty((n_equations, n), dtype=int)
    H[:block_size] = block
    for i in range(1, d_v):
        H[i*block_size:(i+1)*block_size] = rng.permutation(block.T).T
    return H                                      # weight d_v per col, d_c per row

def coding_matrix(H):
    n_equations, n = H.shape
    H_columns, tQ = gaussjordan(H.T, change=True)
    H_reduced = gaussjordan(H_columns.T)
    n_bits = n - H_reduced.sum()
    Y = np.zeros((n, n_bits), dtype=int)
    Y[n - n_bits:, :] = np.identity(n_bits, dtype=int)
    return binaryproduct(tQ.T, Y)

def encode(G, v, snr, seed=None):
    """G v over GF(2) -> BPSK -> AWGN.  snr = 10 log10(1/sigma^2) dB."""
    rng = check_random_state(seed)
    d = binaryproduct(G, v)
    x = (-1) ** d                                 # bit 0 -> +1, bit 1 -> -1
    sigma = 10 ** (-snr / 20)
    return x + rng.randn(*x.shape) * sigma

def incode(H, x):
    return (binaryproduct(H, x) == 0).all()

def decode(H, y, snr, maxiter=1000):
    """Log-domain belief propagation (sum-product)."""
    y = np.asarray(y, dtype=float)
    m, n = H.shape
    var = 10 ** (-snr / 10)                        # sigma^2
    Lc = 2 * y / var                              # channel LLR, log P(0)/P(1)
    Lq = np.zeros((m, n))                          # variable -> check
    Lr = np.zeros((m, n))                          # check -> variable
    checks = [np.where(H[i])[0] for i in range(m)]   # bits in each check
    bits   = [np.where(H[:, j])[0] for j in range(n)]# checks on each bit

    for it in range(maxiter):
        # horizontal: tanh check rule, omit the receiving bit
        for i in range(m):
            for j in checks[i]:
                src = Lq[i] if it > 0 else Lc
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
                    Lr[i, j] = np.log(num / denom)   # 2*artanh(X)
        # vertical: sum of channel + other checks, omit this check
        for j in range(n):
            for i in bits[j]:
                Lq[i, j] = Lc[j] + sum(Lr[ip, j] for ip in bits[j] if ip != i)
        # posterior + hard decision + stopping test
        L_post = Lc + np.array([Lr[bits[j], j].sum() for j in range(n)])
        x = (L_post <= 0).astype(int)              # positive LLR -> bit 0
        if incode(H, x):
            break
    else:
        warnings.warn("Decoding stopped before convergence.")
    return x
```
