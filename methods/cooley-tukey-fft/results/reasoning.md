OK, let me think this through from scratch. I have $N$ samples $A(0), \dots, A(N-1)$ and I want all $N$ Fourier coefficients

$$X(j) = \sum_{k=0}^{N-1} A(k)\, W^{jk}, \qquad W = e^{-2\pi i/N}.$$

Written out, this is just a matrix times a vector: the matrix $F$ has $(j,k)$ entry $W^{jk}$, and $X = F A$. A dense $N \times N$ matrix-vector product is $N^2$ complex multiplications. For a few hundred points that's fine; for $N$ in the tens of thousands it's already painful, and at $N = 10^6$ it's $10^{12}$ multiplies — that's the difference between a coffee break and two weeks of machine time. So the whole game is: can I beat $N^2$? The naive method treats every entry of $F$ as if it were an arbitrary number. But it isn't. Every entry is a power of one number, $W$. That has to be the lever. Let me stare at the structure.

What do I actually know about $W$? Two things, and they're cheap. First, $W^N = 1$ — the exponents only matter mod $N$, so the matrix entries $W^{jk}$ wrap around; there are at most $N$ distinct values among the $N^2$ entries, not $N^2$. Second, and this is the one that smells useful: if $N$ is even, $W^{N/2} = e^{-2\pi i \cdot (N/2)/N} = e^{-\pi i} = -1$. And $W^2 = e^{-2\pi i \cdot 2/N} = e^{-2\pi i/(N/2)}$, which is *itself* a primitive root of unity of order $N/2$. So squaring $W$ gives me the root of unity for a half-length problem. That can't be a coincidence I should ignore — it says a length-$N$ transform has a length-$N/2$ transform hiding inside it.

So the redundancy is real. The question is how to convert "the same value appears many times" into "I do the arithmetic once and reuse it." Grouping equal entries the way the old hand-computers did — collect all the terms that multiply the same $\cos$ value, add them first, multiply once — saves a constant factor, sure. But I just convinced myself the savings there are bounded: you remove duplicate *multiplications by equal values*, but you never actually build a shorter transform, so the count stays $\Theta(N^2)$. That's a dead end if I want the *order* to drop. To change the order I need to express the length-$N$ DFT in terms of genuinely shorter DFTs. The squaring observation is pointing right at that. Let me chase it.

How do I get a length-$N/2$ transform to fall out of the sum? The sum runs over all $k$. If I split the index $k$ into two classes and the exponent simplifies on each class, I might get two half-sums each of which is a clean shorter DFT. The squaring fact involved $W^2$, i.e. *even* powers, so let me split $k$ by parity: $k = 2m$ for the even-indexed samples and $k = 2m+1$ for the odd-indexed ones, each with $m = 0, \dots, N/2 - 1$.

$$X(j) = \sum_{m=0}^{N/2-1} A(2m)\, W^{j(2m)} + \sum_{m=0}^{N/2-1} A(2m+1)\, W^{j(2m+1)}.$$

Now pull the structure out of the exponents. $W^{j(2m)} = (W^2)^{jm}$, and I just noted $W^2 = e^{-2\pi i/(N/2)}$ — call it $W_{N/2}$, the half-length root. So the first sum is $\sum_m A(2m)\, W_{N/2}^{\,jm}$ — that is *exactly* a length-$N/2$ DFT, of the even subsequence, evaluated at index $j$. For the odd sum, $W^{j(2m+1)} = W^{j}\,(W^2)^{jm} = W^j\, W_{N/2}^{\,jm}$, so I can factor the single $W^j$ out front:

$$X(j) = \underbrace{\sum_{m} A(2m)\, W_{N/2}^{\,jm}}_{E(j)} \;+\; W^{j}\underbrace{\sum_{m} A(2m+1)\, W_{N/2}^{\,jm}}_{O(j)}.$$

There it is. $E$ is the length-$N/2$ DFT of the even-indexed inputs, $O$ is the length-$N/2$ DFT of the odd-indexed inputs, and $X(j) = E(j) + W^j O(j)$. The single factor $W^j$ that splices the two halves together — I'll keep calling that the twiddle factor, the thing that "corrects" the odd half for being shifted one sample over.

But wait — something is off if I read this too fast. $E(j)$ and $O(j)$ are defined for a length-$N/2$ transform, so they only naturally make sense for $j = 0, \dots, N/2 - 1$. Yet $X(j)$ runs over the full range $j = 0, \dots, N-1$. How do I get $N$ outputs out of two half-length transforms that each only have $N/2$ values? For a moment this looks like it can't close — I'm short by half the outputs. Stare at it. $E$ and $O$ are DFTs of length $N/2$, so they are *periodic* in $j$ with period $N/2$: $E(j + N/2) = E(j)$, same for $O$. So the formula $X(j) = E(j) + W^j O(j)$ does extend to $j \ge N/2$ — I just reuse $E(j \bmod N/2)$ and $O(j \bmod N/2)$. The only thing that changes between $j$ and $j + N/2$ is the twiddle. And here the halving fact pays off: $W^{j + N/2} = W^j \cdot W^{N/2} = W^j \cdot (-1) = -W^j$. So

$$X(j) = E(j) + W^{j} O(j), \qquad X(j + N/2) = E(j) - W^{j} O(j), \qquad j = 0, \dots, N/2-1.$$

This is much better than I even hoped. I don't recompute anything for the upper half. I compute $E(j)$ and $O(j)$ once, form the product $W^j O(j)$ *once* — that's a single complex multiply — and then I get *two* outputs, $X(j)$ and $X(j+N/2)$, by adding and subtracting. One multiply and two adds, yielding two of the $N$ answers. That little add/subtract pair sharing one twiddle multiply — the butterfly — is the unit of work, and the sign flip is just $W^{N/2} = -1$ doing its job. So a full level of recombination over all $j < N/2$ costs $N/2$ complex multiplies and $N$ adds, i.e. $O(N)$, to merge two half-transforms into the whole.

Now the recursion, because $E$ and $O$ are the same kind of object I started with, only half as long. Computing each of them naively would be $(N/2)^2$, two of those is $N^2/2$ plus the $O(N)$ merge — already half the work, but I haven't recursed yet. The point is I don't compute $E$ and $O$ naively: I apply the *same* split to each. $E$, the even subsequence's transform, splits into its even-even and even-odd quarter-length transforms; $O$ likewise. Let me just write the cost recurrence. If $T(N)$ is the cost of a length-$N$ transform, then

$$T(N) = 2\,T(N/2) + cN,$$

two subproblems of half the size plus a linear merge. Unrolling: at the top level I pay $cN$; that spawns two problems each paying $c(N/2)$, total $cN$ again; four problems each $c(N/4)$, total $cN$ again; and so on. Every level costs $cN$, and the number of levels is the number of times I can halve $N$ down to $1$, which is $\log_2 N$. So $T(N) = cN \log_2 N$. That's the whole prize: $N \log_2 N$ instead of $N^2$. At $N = 2^{20} \approx 10^6$, that's $\sim 20 \cdot 10^6$ versus $10^{12}$ — a factor of fifty thousand. The recursion bottoms out at $N = 1$, and the transform of a single sample is just that sample copied to the output — the one-point DFT is the identity.

I should make sure the recursion is clean, which means I want $N$ to halve exactly every time, all the way down. That's true precisely when $N$ is a power of two. If $N$ isn't, I can pad with zeros up to the next power of two, or — I'll come back to this — factor it differently. Power-of-two first; it makes every split exact and every level uniform.

Now there's a bookkeeping question that I'd better nail before writing code, because the recursion shuffles the data in a specific way. At each level I separate even-indexed from odd-indexed entries. Even index means the least-significant bit of $k$ is $0$; odd means it's $1$. So the first split sorts by bit 0 of $k$. Within the even half, the next split sorts by the next bit; within the odd half, likewise. After $\log_2 N$ splits, each input has been routed by all its bits, *least-significant first*. Follow any leaf down the tree — a path of "even/odd, even/odd, …" choices — and the sample sitting there is the one whose bits, read in the order the splits tested them (LSB first), match that path. Read the path most-significant-first instead and you get the reversed bit pattern. So the sample that ends up at leaf position $p$ is the input $A(k)$ where $k$ is the bit-reversal of $p$. Concretely for $N=8$, position $001$ holds input $100$, position $011$ holds $110$, and so on — reverse the bits.

That's a gift. It means I don't need the recursion's call stack at all. If I first permute the input array into bit-reversed order, then the leaves are already laid out as adjacent one-point transforms, and I can build the answer bottom-up with plain loops. Adjacent pairs combine into 2-point transforms; adjacent pairs of pairs into 4-point transforms; and so on, doubling the block length each pass until the two halves of the whole array combine into the length-$N$ result. There are $\log_2 N$ passes, each touching all $N$ elements once — the same $N \log_2 N$, now with no recursion and, better, *in place*: each butterfly reads two slots and writes the two sums back into those same two slots, so I need no scratch array. And the bit-reversal permutation is just swapping element $p$ with element $\mathrm{rev}(p)$; since reversal is an involution ($\mathrm{rev}(\mathrm{rev}(p)) = p$), the swaps are disjoint and free of extra storage, and there are $N$ of them — cheap relative to the transform itself.

Let me make the iterative passes precise so the twiddles come out right. Pass $s$ builds transforms of length $L = 2^s$, for $s = 1, \dots, \log_2 N$. Within a length-$L$ block, the relevant root of unity is $W_L = e^{-2\pi i/L}$, because a length-$L$ transform's twiddles are powers of *its own* root, and the butterfly at offset $r$ inside the block uses twiddle $W_L^{\,r}$ for $r = 0, \dots, L/2 - 1$, pairing element $r$ with element $r + L/2$. So within each block I march a running twiddle $w$ starting at $1$ and multiplying by $W_L$ each step: $u = a[\text{start}+r]$, $t = w \cdot a[\text{start}+r+L/2]$, then $a[\text{start}+r] = u + t$ and $a[\text{start}+r+L/2] = u - t$, then $w \leftarrow w\,W_L$. That add/subtract pair is exactly the $E(j) \pm W^j O(j)$ I derived, just localized to one block at one pass. The only subtlety worth flagging: I compute the root $W_L$ once per pass (a single transcendental evaluation) and propagate $w$ by multiplication through the inner loop, rather than calling sine/cosine inside the loop — $\log_2 N$ transcendental calls total instead of $O(N \log N)$. (Repeated complex multiplication does drift in floating point over a long block; the practical fix is the trigonometric recurrence that updates $(\cos, \sin)$ stably, but the algorithm is the same.)

Let me sanity-check the whole thing on the smallest non-trivial case, $N = 2$. Bit reversal of two elements is the identity. One pass, $L = 2$, $W_2 = e^{-\pi i} = -1$, one butterfly with $w = 1$: $X(0) = A(0) + A(1)$, $X(1) = A(0) - A(1)$. That's the correct 2-point DFT. Good. And for $N=4$ it composes two such passes with twiddles $1$ and $W_4 = -i$ in the second pass; expanding it by hand reproduces the $4 \times 4$ DFT matrix. The pieces lock together.

Now — is power-of-two really essential, or did I just pick it for convenience? Go back to the split. Nothing about it needed the factor to be $2$. Suppose $N = r_1 r_2$. I can reindex both the input and the output through the two factors. Write the input index as $k = k_1 r_2 + k_0$ and the output index as $j = j_1 r_1 + j_0$, with $k_0, j_1 \in \{0,\dots,r_2-1\}$ and $k_1, j_0 \in \{0,\dots,r_1-1\}$ — every $k$ and $j$ in $[0,N)$ is hit exactly once. Substitute into $W^{jk}$ and expand the exponent $jk = (j_1 r_1 + j_0)(k_1 r_2 + k_0)$:

$$jk = j_1 k_1\, r_1 r_2 + j_1 k_0\, r_1 + j_0 k_1\, r_2 + j_0 k_0.$$

The first term has the factor $r_1 r_2 = N$, so $W^{j_1 k_1 N} = (W^N)^{j_1 k_1} = 1$ — it vanishes, exactly the way $W^{jN/2}$ collapsing was the radix-2 special case. The term $W^{j_1 k_0 r_1}$: since $W^{r_1} = e^{-2\pi i r_1/N} = e^{-2\pi i/r_2}$ is the $r_2$-th root, this is $W_{r_2}^{\,j_1 k_0}$, depending only on the "$r_2$-world" indices. The term $W^{j_0 k_1 r_2} = W_{r_1}^{\,j_0 k_1}$, the $r_1$-th root, in the "$r_1$-world." And the leftover cross term $W^{j_0 k_0}$ couples the two worlds — there's the twiddle again, now in general form. So, grouping the sum,

$$X(j_1 r_1 + j_0) = \sum_{k_0=0}^{r_2-1} W_{r_2}^{\,j_1 k_0}\; W^{\,j_0 k_0} \Big[ \sum_{k_1=0}^{r_1-1} A(k_1 r_2 + k_0)\, W_{r_1}^{\,j_0 k_1} \Big].$$

Read inside out: the bracketed inner sum, for each fixed $k_0$, is a length-$r_1$ DFT (over $k_1$) — there are $r_2$ of them. Multiply each result by the twiddle $W^{j_0 k_0}$. Then the outer sum over $k_0$ is a length-$r_2$ DFT — there are $r_1$ of them. Counting: $r_2$ transforms of length $r_1$ plus $r_1$ transforms of length $r_2$, each inner transform $\sim r_1^2$ or $r_2^2$, gives roughly $r_2 r_1^2 + r_1 r_2^2 = N(r_1 + r_2)$ operations instead of $N^2$. And the radix-2 butterfly is just this with $r_1 = 2$: two half-length transforms plus the twiddle splice.

The recursion now generalizes: if $r_1$ or $r_2$ is itself composite, factor it again. Carried to the prime factorization $N = p_1 p_2 \cdots p_L$, the cost is $N(p_1 + p_2 + \dots + p_L)$ — $N$ times the sum of the prime factors. When all the primes are $2$ ($N = 2^L$) that sum is $2L = 2\log_2 N$, recovering $2N\log_2 N$. The lesson is that the method loves *highly composite* $N$, and conversely is helpless when $N$ is prime: there's no nontrivial factorization, the sum-of-prime-factors is just $N$ itself, and I'm back to $N^2$. Padding to the next power of two is the clean escape.

I want to place this against the doubling rules I knew going in. Runge's method builds a $2N$-point transform from two $N$-point ones with about $N$ extra operations — that's exactly my $r_1 = 2$ merge, but stated only in the doubling direction, $N \to 2N$, so it's tied to repeated doubling rather than a single recursion. Stumpff added a tripling rule ($r=3$) and hinted at a general multiple. The Danielson–Lanczos halving lemma is the cleanest of these — a length-$N$ DFT is the sum of two length-$N/2$ DFTs on the even and odd samples, applied recursively — which is precisely the radix-2 case I rederived, prized in their setting because the redundant recomputation doubles the length for barely more than the half-length labor and doubles as an accuracy check. What none of them isolated is the general composite-$N$ statement with the explicit twiddle $W^{j_0 k_0}$ between the two stages and the index reindexing that makes it a uniform, in-place machine procedure for any factorization.

There's a different factorization worth distinguishing, because it looks similar and isn't. If $r_1$ and $r_2$ are *coprime*, the Chinese Remainder Theorem gives an index map under which the cross term disappears entirely — no twiddle at all, just a genuine two-dimensional $r_1 \times r_2$ transform. That's a real, multiplication-saving scheme, but the coprimality is essential to it, and that's exactly why it can't do what I want here: $N = 2^L$ factors as $2 \cdot 2 \cdots 2$, and those factors are *not* coprime, so the CRT route can't recurse on a power of two at all. The twiddle-factor version I derived has no such restriction — it works for *any* factorization, coprime or not, including the repeated factor of $2$ — and that generality, plus the in-place butterfly bookkeeping, is what turns the centuries-old splitting idea into a single algorithm I can just run.

Let me write it. First the honest baseline and the recursive form that mirrors the derivation directly, then the iterative in-place version that's what you actually run.

```python
import cmath

def dft_direct(A, sign=-1):
    # baseline X(j) = sum_k A(k) W^{jk}, W = exp(sign*2pi i/N): the O(N^2) matrix multiply
    N = len(A)
    W = cmath.exp(sign * 2j * cmath.pi / N)
    return [sum(A[k] * W**(j * k) for k in range(N)) for j in range(N)]

def fft_recursive(A, sign=-1):
    # decimation in time: split by parity, recurse, splice with one twiddle per pair
    N = len(A)
    if N == 1:                     # one-point DFT is the identity
        return [A[0]]
    E = fft_recursive(A[0::2], sign)   # transform of the even-indexed samples
    O = fft_recursive(A[1::2], sign)   # transform of the odd-indexed samples
    X = [0j] * N
    for j in range(N // 2):
        t = cmath.exp(sign * 2j * cmath.pi * j / N) * O[j]   # twiddle W^j times O(j)
        X[j]          = E[j] + t       # X(j)        = E(j) + W^j O(j)
        X[j + N // 2] = E[j] - t       # X(j+N/2)    = E(j) - W^j O(j),  since W^{N/2}=-1
    return X

def _bit_reverse_in_place(A):
    # route input to leaf order: sample at position rev(p) goes to position p
    n = len(A)
    bits = n.bit_length() - 1
    for p in range(n):
        q = int(format(p, '0{}b'.format(bits))[::-1], 2)
        if q > p:                  # reversal is an involution -> disjoint swaps, in place
            A[p], A[q] = A[q], A[p]

def fft_iterative(A, sign=-1):
    # the runnable version: bit-reverse, then log2 N doubling passes of in-place butterflies
    A = list(A)
    n = len(A)
    _bit_reverse_in_place(A)
    L = 2
    while L <= n:                  # pass s builds length-L = 2^s transforms
        wL = cmath.exp(sign * 2j * cmath.pi / L)   # root of unity for this block length
        for start in range(0, n, L):
            w = 1 + 0j             # running twiddle, propagated by multiplication
            for r in range(L // 2):
                u = A[start + r]
                t = w * A[start + r + L // 2]       # twiddle times the upper half
                A[start + r]          = u + t       # butterfly: E + W^r O
                A[start + r + L // 2] = u - t       # and        E - W^r O
                w *= wL
        L <<= 1
    return A
```

The chain, end to end: the DFT is a matrix-vector product whose matrix is all powers of one root of unity, so it is drowning in redundancy; the redundancy is exposed by splitting the sum on the index, where $W^2$ being the half-length root makes each half a genuine shorter DFT and the cross factor $W^j$ — the twiddle — splices them; the periodicity of the half-transforms plus $W^{N/2} = -1$ lets one twiddle multiply produce two outputs, the butterfly; recursing gives $T(N) = 2T(N/2) + O(N) = O(N\log N)$; and the leaf ordering being bit-reversal turns the recursion into in-place doubling passes — which generalizes from radix 2 to any composite $N = r_1 r_2$ via the same index reindexing and the twiddle $W^{j_0 k_0}$, at cost $N$ times the sum of the prime factors.
