I would summarize the canonical result as Shannon's channel coding theorem. It answers the question of how much information can be reliably pushed through a noisy communication channel if we are allowed to encode messages into long blocks and decode the received blocks with a probability of error that can be made arbitrarily small.

The setting is a discrete memoryless channel. It takes an input symbol x from an input alphabet and produces an output symbol y from an output alphabet according to a fixed conditional probability law W(y|x), and each channel use is independent of the others. Before Shannon, one might have guessed that noise forces a fundamental degradation: the more reliable you demand the communication, the more redundancy you must add, and perhaps the useful rate must shrink toward zero. Shannon showed the opposite. There is a sharp threshold, the channel capacity C, such that every rate below C is achievable with vanishing error, and no rate above C is achievable no matter how clever the encoding.

The capacity is defined as the maximum mutual information between the channel input and output over all possible input distributions. If p(x) is an input distribution and p(x, y) = p(x) W(y|x), then the mutual information is I(X;Y) = H(X) - H(X|Y), where H(X) is the entropy of the input and H(X|Y) is the conditional entropy of the input given the output. Intuitively, H(X) measures how much uncertainty there was at the sender, and H(X|Y) measures how much uncertainty remains after observing the channel output. Their difference is how much ambiguity the channel has resolved. Because we are free to choose how often we use each input symbol, we maximize over p(x), giving C = max_{p(x)} I(X;Y).

The achievability half of the theorem is surprising and beautiful. It uses a random coding argument rather than an explicit construction. Fix a good input distribution p(x) achieving capacity or close to it. Draw 2^{nR} codewords independently at random, each of length n, with each symbol generated i.i.d. from p(x). This is the random codebook. The encoder maps a message to one of these random codewords and sends it through the channel. The decoder, upon receiving the output block, searches the codebook for a codeword that is jointly typical with the received block. If exactly one such codeword exists, the decoder declares that message; otherwise it declares an error.

Why does this work? The transmitted codeword is correlated with the received block through the channel, so with high probability the pair is jointly typical. Every other codeword, however, was generated independently of the transmitted one and independently of the received block. The probability that any particular wrong codeword looks jointly typical with the output is approximately 2^{-n I(X;Y)}. There are 2^{nR} - 1 wrong codewords, so a union bound bounds the total error probability by roughly 2^{nR} 2^{-n I(X;Y)} = 2^{-n (I(X;Y) - R)}. If R < I(X;Y), this bound goes to zero as n grows. Therefore the average error probability over the random codebook ensemble is small, which means at least one deterministic codebook in the ensemble must have small error. A standard expurgation step, discarding the worst half of the codewords, upgrades this to small maximal error with negligible loss in rate. Maximizing over p(x) gives that every rate R < C is reliably achievable.

The converse half shows that no rate above C is possible. Suppose a code with 2^{nR} messages and vanishing error probability exists. Then the message M is almost determined by the received block Y^n, so the conditional entropy H(M|Y^n) is o(n). By the chain rule for entropy, nR = H(M) = I(M;Y^n) + H(M|Y^n) <= I(M;Y^n) + o(n). But the message can influence the output only through the channel inputs X^n, so I(M;Y^n) <= I(X^n;Y^n). Because the channel is memoryless, the mutual information across n independent uses cannot exceed nC, giving I(X^n;Y^n) <= nC. Combining these inequalities yields nR <= nC + o(n), so R <= C in the limit. This proves the impossibility direction.

A particularly important special case is the additive white Gaussian noise channel with bandwidth W, signal power P, and noise power N. Its capacity is C = W log_2(1 + P/N) bits per second. This formula is so famous because it connects a physical signal-to-noise ratio P/N to an exact information rate limit.

The theorem is an existence result, not a recipe. It tells us that good long codes exist, but it does not tell us how to find them or how to decode them efficiently. Decoding by exhaustive search over an exponentially large codebook is computationally infeasible. Modern coding theory, from Hamming and Reed-Solomon codes to turbo codes, low-density parity-check codes, and polar codes, can be seen as the ongoing project of finding codes that approach capacity while admitting efficient encoding and decoding algorithms.

```python
import numpy as np

def mutual_information_bsc(p0, eps):
    """I(X;Y) for a binary symmetric channel with P(X=0)=p0."""
    W = np.array([[1 - eps, eps], [eps, 1 - eps]])  # W[y, x]
    px = np.array([p0, 1 - p0])
    py = W @ px
    mi = 0.0
    for x in range(2):
        for y in range(2):
            if W[y, x] > 0:
                mi += px[x] * W[y, x] * np.log2(W[y, x] / py[y])
    return mi

eps = 0.1
ps = np.linspace(0.001, 0.999, 999)
mis = [mutual_information_bsc(p0, eps) for p0 in ps]
capacity = max(mis)
best_p = ps[mis.index(capacity)]
print(f"Capacity C ≈ {capacity:.4f} bits/channel use")
print(f"Capacity-achieving input: P(X=0) ≈ {best_p:.4f}")

def simulate_random_code(n, R, eps, num_trials=200):
    """Estimate block-error rate of a random code with ML decoding."""
    M = max(2, int(round(2 ** (n * R))))
    px0 = 0.5
    errors = 0
    for _ in range(num_trials):
        codebook = (np.random.rand(M, n) > px0).astype(int)
        message = np.random.randint(M)
        sent = codebook[message]
        received = (sent + (np.random.rand(n) < eps).astype(int)) % 2
        # Minimum Hamming-distance (ML for BSC) decoder.
        distances = np.sum((codebook - received) % 2, axis=1)
        decoded = np.argmin(distances)
        if decoded != message:
            errors += 1
    return errors / num_trials

np.random.seed(0)

# Rate safely below capacity: error probability should be small.
n = 15
R_below = 0.30
err_below = simulate_random_code(n, R_below, eps)
print(f"Below capacity: R={R_below:.2f}, n={n}, block-error rate ≈ {err_below:.3f}")

# Rate above capacity: error probability should be large.
R_above = 0.60
err_above = simulate_random_code(n, R_above, eps)
print(f"Above capacity: R={R_above:.2f}, n={n}, block-error rate ≈ {err_above:.3f}")

print(f"Capacity threshold: C ≈ {capacity:.4f}")
```

I would conclude by emphasizing the conceptual status of the theorem. Shannon's channel coding theorem identifies capacity as the fundamental figure of merit for a noisy channel. It separates what is information-theoretically possible from what is practically convenient. Below capacity, the world of long random-like codes contains reliable communication schemes even though we may not know how to implement them efficiently. Above capacity, the data-processing inequality and the memoryless structure of the channel make reliable communication impossible. This is the foundation of modern information theory, and it is why the canonical method is called Shannon's channel coding theorem.
