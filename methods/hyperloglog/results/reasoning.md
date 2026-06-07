OK, let me start from what actually hurts. I have a stream — packets on a wire, lines in a file too big to hold, keys flowing past — and I want one number: how many *distinct* values went by. Not how many records, how many different ones. And I want it cheap: one pass, because the data is read-once and I can't sort it or revisit it, and tiny memory, because the whole point is that the data is enormous and the count itself might be a billion.

The honest method is to remember everything I've seen — a hash set, or sort and dedup. But that's exactly the wall: to certify an element as new I must have stored every distinct element so far, so the memory is Θ(n), linear in the very quantity I'm trying to learn. If n is 10^9 that's gigabytes of table for a statistic I wanted to be a footnote. Linear memory is a non-starter. So exactness is off the table; I'll have to estimate, accept a few percent error, and buy sub-linear memory with that slack.

Can I sample? Keep a small random subset, count distinct in it, scale up? No — and it's worth seeing exactly why, because it tells me what cardinality *is*. Distinct-counting is brutally sensitive to how often things repeat. If one value appears a million times and ten others appear once each, a small sample is almost all the popular value and sees maybe two distinct values; scale that up and I'm wildly wrong. Sampling estimates the number of *occurrences*, weighted by frequency, but cardinality must be blind to frequency — a value seen a million times and a value seen once must both count as exactly one. So whatever I build has to be *insensitive to replication*: its answer must depend only on the set of values present, not on their multiplicities.

That word "set" is the lever. How do I make something depend only on the set, automatically, while reading a stream with repeats? Hash. Push every record through a fixed hash function h, and treat the output bits as if they were independent fair coin flips — uniform over {0,1}^L, equivalently a uniform real in [0,1]. Two things fall out at once. First, identical records hash to identical bits, so any function I compute of the *collection of hash values* is really a function of the *set* of distinct hash values — repeats land on top of themselves and change nothing. Replication-insensitivity comes for free. Second, hashing hands me exactly the uniform randomness I'll need to do probability on. So from here on I don't think about the data; I think about n uniform random binary strings and what cheap, set-only summary of them betrays n.

What summary? Rare patterns are evidence of large n. Look at the leading bits of a hashed string. The probability that a uniform string starts with k zeros and then a one — the pattern 0^k 1 — is 2^{-(k+1)}. So a run of k leading zeros happens about once in 2^{k+1} strings. If I've drawn n strings and I've seen a leading-zero run of length k somewhere, that's mild evidence that n is at least around 2^{k}. Long leading-zero runs are a thermometer for log₂ n. For the bitmap version, let r be the zero-based position of the first 1-bit, which is exactly the length of the leading-zero run. Among n distinct values, the largest r I observe should sit near log₂ n. If I later keep a single rank instead of a bitmap position, the one-based value ρ = r + 1 is cleaner: it is the position of the leftmost 1-bit.

Let me make that concrete the way the Flajolet–Martin probabilistic-counting idea does. Keep a bit-vector BITMAP[0..L-1], all zero. For each element, set BITMAP[r(hash(x))] = 1. Bit i gets turned on the first time some hashed value has exactly i leading zeros before its first 1, which happens with per-element probability 2^{-(i+1)}. So after n distinct elements, bit 0 has been hit about n/2 times, bit 1 about n/4 times, and so on — the low bits are almost surely 1, the high bits almost surely 0, and there's a fuzzy transition around i ≈ log₂ n. The position of that transition encodes the cardinality. The clean thing to read off is R = the position of the *leftmost zero* in the bitmap: below R everything's filled, above R it's empty, and R ≈ log₂ n.

But "≈ log₂ n" hides a constant and a lot of noise, and I have to pin both down or this is useless. Run the real analysis: under the uniform model, what is E[R]? You set up the probability that R ≥ k — that bits 0..k-1 are all set — and asymptotically, via a Mellin transform and summing residues, the expectation comes out to

  E[R] ≈ log₂(φ·n),  with φ = 0.77351…,

plus a tiny periodic wobble of amplitude below 10^{-5} that I can ignore. That φ is not cosmetic — it's a genuine multiplicative bias. The leftmost zero systematically sits a bit below log₂ n because of the fringe of holes-and-ones around the transition, and φ corrects for it: 2^{R} estimates n only after I divide by φ. So my single-bitmap estimate is 2^{R}/φ. Good — but now the dispersion: σ(R) ≈ 1.12 bits. A standard deviation of more than a full bit means 2^{R} is off by a factor of two routinely. One bitmap pins n only to within a binary order of magnitude. Useless as a point estimate. Wall.

The cure for "one noisy measurement" is "average many." If I had m independent estimators R^{(1)},…,R^{(m)}, each with σ ≈ 1.12, their mean has σ ≈ 1.12/√m, and I can drive the error down by spending memory on m. The naive way to get m independent estimators is m independent hash functions — run each element through all m, update m bitmaps. But that multiplies the per-element CPU by m, and worse, it needs a family of m provably-independent hashes, which I don't actually know how to construct cheaply. Dead end as stated.

I don't need m *hash functions*; I need m *independent streams*. So split the one hash. Use a few of its bits to choose which of m buckets this element belongs to — say bucket a = h(x) mod m — and use the *remaining* bits to compute r and update only that bucket's bitmap. Now each element touches exactly one bucket, one hash, constant work. About n/m distinct elements land in each bucket, and the buckets are (near enough) independent because the splitting bits are independent of the observation bits. Call this stochastic averaging — I'm emulating m parallel experiments with a single hash by letting the hash itself randomize the assignment. Each bucket's leftmost-zero R_j now estimates log₂(φ·n/m). Average them: with S = Σ_j R_j, the estimate is

  Z = (m/φ)·2^{S/m}.

Note what averaging in the exponent does — 2^{S/m} = (∏_j 2^{R_j})^{1/m} is the geometric mean of the per-bucket 2^{R_j}, and (1/φ)·that estimates n/m, so m·it estimates n. Standard error now improves like 0.78/√m. With m = 64 bitmaps of 32 bits, about 256 bytes, I get ~10% error and can count past a billion. That's real, that works.

But stare at the memory: m bitmaps, each L bits. A whole 32-bit vector per bucket — and what do I actually extract from each bitmap? One number, R_j, of size about log₂(n/m). I'm storing L bits to recover a single log₂ n-sized quantity. The bitmap is not literally the same as a maximum; its leftmost zero depends on which earlier positions have been filled, and holes matter. But the evidence it uses is still the same bit-pattern scale: a position around log₂(n/m) is where the action is. If I am willing to change the statistic and re-analyze its bias, I can ask for just one transition-sized integer per bucket.

What if a bucket stores only the *maximum* ρ it has ever seen, now using the one-based position ρ = r + 1 — a single register M[j] = max over the bucket of the leftmost-1-bit position? The maximum leading-zero run among about n/m uniform strings is close to log₂(n/m). So M[j] is a small integer near log₂(n/m), shifted by one and a fixed distributional offset that the bias constant will absorb, and to store an integer of that size I need only about log₂ log₂ n bits — five bits comfortably covers cardinalities past 10^9. With a finite L-bit hash and p bucket bits, the suffix has only K = L - p bits; then a nonempty bucket records ranks ρ in 1..K+1, assigning the all-zero suffix the capped rank K+1 rather than pretending it has an infinite run. An untouched practical bucket can stay at 0. That's the collapse I wanted: from an L-bit bitmap per bucket down to a log-log-sized register per bucket. Memory goes from "log n per bucket" to "log log n per bucket." A register, not a vector.

Now I have to re-derive the estimator around the max, because I changed the observable. M[j] ≈ log₂(n/m), so 2^{M[j]} is of the order of n/m per bucket, and I average across buckets the same way — arithmetic mean of M[j] in the exponent, which is the geometric mean of the 2^{M[j]}:

  E = α_m · m · 2^{(1/m)·Σ_j M[j]}.

The constant α_m is again a bias correction: the max of n/m geometric-ish variables has its own systematic offset from log₂(n/m), exactly as φ corrected the leftmost-zero, and α_m absorbs it (it works out to something near 0.39701 in the limit). This is the LogLog estimator — log log n memory, and it counts billions in a kilobyte or two. The standard error here is 1.30/√m.

Except 1.30/√m bugs me. The order-statistics estimators — track the smallest hash values, use E[min] = 1/(n+1) — get down to about 1.00/√m with the same m registers. And there's a lower bound floating around (Chassaing–Gérin) saying that a wide class of order-statistics estimators can't beat something close to 1/√m. So 1.30 is leaving a real gap below the ~1.0 benchmark, and I want to know *where the gap comes from*, because the observable is fine — the max-ρ register is a perfectly good thermometer. So it has to be the *averaging*.

Let me look hard at the distribution of the per-bucket quantity 2^{M[j]}. M[j] is roughly geometrically distributed around log₂(n/m): it's usually close to the mean, but it has a slowly decaying right tail — every so often a bucket gets a freakishly long leading-zero run and M[j] jumps by 2 or 3, which *doubles or quadruples* 2^{M[j]}. The geometric mean (arithmetic mean of M[j] in the exponent) is gentler than a plain arithmetic mean of the 2^{M[j]}, but it still lets those occasional huge values pull the estimate up — a single bucket that lucks into a long run shifts S/m and inflates E. That fat right tail is the variance I'm paying for. I need a mean that *down-weights* the large outliers instead of letting them dominate.

Which mean kills a slowly-decaying right tail? Stare at it. The arithmetic mean of values v_j weights each v_j by 1 — a giant v_j dominates. The geometric mean softens that but a giant log v_j still drifts the average. The *harmonic* mean, (1/m · Σ 1/v_j)^{-1}, weights each v_j by 1/v_j — so a freakishly large v_j contributes almost *nothing* to the sum of reciprocals; it can't pull the mean up, only fail to pull it down. A harmonic mean is exactly the variance-reducer for distributions with heavy right tails: the very buckets that wreck the arithmetic and geometric means are the ones the harmonic mean ignores. So let me replace the geometric mean with the harmonic mean of the 2^{M[j]}.

Write it out. The harmonic mean of {2^{M[j]}} is m / (Σ_j 2^{-M[j]}). Each 2^{M[j]} is of the order of n/m, so the harmonic mean targets that same scale, and I multiply by m to get n — but I want to be careful, because I'm going to fold the factors together. Define the indicator

  Z = (Σ_{j=1}^{m} 2^{-M[j]})^{-1}.

The harmonic mean is m·Z, which targets n/m, so m·(m·Z) = m²·Z targets n. Up to a multiplicative bias I haven't fixed yet, the cardinality estimate is

  E = α_m · m² · Z = α_m · m² / Σ_{j=1}^{m} 2^{-M[j]}.

Same observable as before — max leading-zero run per bucket — only the evaluation function changed, from geometric to harmonic mean. And the harmonic form is gorgeous to compute: just accumulate Σ 2^{-M[j]} over the registers and invert.

Now nail the bias constant α_m, because m²·Z has a fixed multiplicative bias, exactly as 2^R and the LogLog raw estimate did. I can at least see where the constant must come from. To isolate that constant, I first remove finite-hash boundary effects and think with infinite suffixes and untouched buckets initialized to -∞; the finite program will cap ρ at K+1 and use the zero-initialized correction only at the end. In one bucket with ν hashed suffixes, the rank Y = ρ(w) satisfies P(Y ≥ k) = 2^{1-k}, so the maximum M has

  P(M = k) = (1 - 2^{-k})^ν - (1 - 2^{-(k-1)})^ν.

Fixed n makes the bucket occupancies multinomial, which tangles the registers together. Poissonize: let the total cardinality be Poisson with mean λ, so each bucket is an independent Poisson flow of rate x = λ/m. Then the probability that a bucket's maximum equals k becomes

  g(x/2^k),  where g(u) = e^{-u} - e^{-2u}.

The annoying part is the reciprocal in Z = 1/Σ2^{-M[j]}; it couples all the registers. But the identity 1/a = ∫_0^∞ e^{-at} dt separates it. After substituting the register probabilities, the expectation becomes an integral of a product:

  E_P(λ)[Z] = ∫_0^∞ G(x,t)^m dt,  G(x,t) = Σ_{k≥1} g(x/2^k)e^{-t/2^k},  x = λ/m.

Change variables t = xu. Now the object to understand is G(x,xu). The Mellin calculation says that, as x grows, this harmonic sum settles to

  f(u) = log₂((2+u)/(1+u)),

up to tiny periodic fluctuations. So the mean is asymptotically

  E_P(λ)[Z] ≈ x ∫_0^∞ f(u)^m du = (λ/m) J_0(m),

where J_0(m) = ∫_0^∞ f(u)^m du. Depoissonization then says the same leading term holds again for fixed n. To make α_m m² Z have mean n, I need α_m m² · (n/m)J_0(m) = n, hence

  α_m = 1/(mJ_0(m)) = ( m ∫_0^∞ ( log₂( (2+u)/(1+u) ) )^m du )^{-1},

and by Laplace's method its large-m limit is the clean number α_∞ = 1/(2 ln 2) ≈ 0.72134. For small m the integral isn't quite there, so I just evaluate it: α_16 = 0.673, α_32 = 0.697, α_64 = 0.709, and for m ≥ 128 the approximation α_m = 0.7213/(1 + 1.079/m) is faithful. With α_m in front, E is asymptotically unbiased.

And the payoff I was chasing — the variance. The second moment uses the sibling identity 1/a² = ∫_0^∞ t e^{-at} dt, so the same separated integral appears with one extra factor of u:

  J_1(m) = ∫_0^∞ u f(u)^m du,  Var(Z) ≈ (n/m)^2 (J_1(m) - J_0(m)^2).

After multiplying by α_m m² and dividing by n, the relative standard deviation is

  sqrt(J_1(m)/J_0(m)^2 - 1) = β_m/√m,  with β_m = sqrt(m(J_1(m)/J_0(m)^2 - 1)).

Laplace's method on J_0 and J_1 gives β_∞ = √(3 ln 2 − 1) ≈ 1.03896. So harmonic averaging buys 1.04/√m, down from LogLog's 1.30/√m, right at the ~1/√m floor. The fat-tailed buckets that cost LogLog its accuracy are exactly the ones the reciprocal sum discounts. That's the whole gain, and it came purely from swapping the mean.

Let me sanity-check the memory and range claims before I trust the formula. A register value itself is on the log₂ N scale, but storing that small integer costs only log₂ log₂ N + O(1) bits; in the 32-bit practical range, the value lies in 0..L+1-p, so 5-bit "short bytes" cover it. Take m = 2048 registers at 5 bits ≈ 1.3 kB packed; the predicted error is 1.04/√2048 ≈ 2.3%. So about a kilobyte-and-a-half with ordinary overhead counts to a billion at a couple percent. This sits near the two limits I can actually argue: any ε-approximate counter over a range up to N needs Ω(log log N) bits, because the cardinalities to be distinguished live on an exponential scale 1, (1+ε), (1+ε)², … and there are log_{1+ε} N of them, so log₂ log_{1+ε} N bits just to name the answer; and 1/√m is the order-statistics accuracy scale, which the harmonic estimator reaches up to the constant 1.04.

Now I have to make this survive contact with real cardinalities, not just the asymptotic ideal, because two regimes break the clean formula and I have to patch both.

First the small-n regime. Initialize registers to 0 (so the structure is usable even when n is only a small multiple of m). When n is small relative to m, lots of buckets get *no* element at all and their registers stay 0 — and a register stuck at 0 contributes 2^{-0} = 1 to the sum Σ 2^{-M[j]}, the largest possible term. With many empty buckets the sum is dominated by these 1's and the estimator saturates: as n → 0 the raw E tends to α_m·m ≈ 0.7m, not to 0. So below roughly n ≈ 5m/2 the harmonic estimator is distorted and I can't trust it. But notice — when buckets are mostly empty, the *count of empty buckets is itself a clean estimator*. This is the coupon-collector / balls-in-bins fact: throw n balls into m bins, the expected number of empty bins is m·e^{-n/m}. So if V registers are still 0, then V ≈ m·e^{-n/m}, which inverts to n ≈ m·log(m/V) — linear counting. So the rule: if the raw E ≤ 5m/2 and there are empty registers (V ≠ 0), return E* = m·log(m/V) instead; the empty-bucket count carries the signal precisely where the harmonic estimator fails. (If E ≤ 5m/2 but no register is empty, the harmonic estimate is already fine, keep E.)

Second the large-n regime. With a 32-bit hash there are only 2^{32} possible hashed values, so as n climbs toward 2^{32} distinct hashes start *colliding* — two distinct elements grab the same hash and the structure can't tell them apart, so E undercounts. Model it again as balls in bins: E really estimates the number of *distinct hashed values* seen, which after collisions is about 2^{32}(1 − e^{-n/2^{32}}). Invert that relation while E is still below the hash range: E* = −2^{32}·log(1 − E/2^{32}). So above E > 2^{32}/30, apply this large-range correction; if the raw estimate reaches 2^{32}, the 32-bit hash has saturated and no finite inversion remains. In the wide middle band, no correction. With all three regimes stitched together the estimate is Gaussian around the truth with σ ≈ 1.04/√m, landing within σ, 2σ, 3σ of the exact count about 65%, 95%, 99% of the time.

Let me write it as a real program. The per-element work is: hash to a 32-bit word, peel off the first p bits for the bucket index j, feed the remaining K = 32 - p bits to ρ, assign the all-zero suffix rank K+1, and keep the max in register j. The estimate sums 2^{-M[j]} across registers, scales by α_m·m², and applies the small/large-range corrections whose thresholds are dimensioned for the 32-bit hash range.

```python
import math
from hashlib import sha1

HASH_BITS = 32
HASH_RANGE = 1 << HASH_BITS

def hash32(value):
    # fixed pseudo-uniform hash D -> {0,1}^32
    data = value if isinstance(value, bytes) else str(value).encode("utf-8")
    return int.from_bytes(sha1(data).digest()[:4], byteorder="big")

def rho(w, max_width):
    # finite-suffix rank: 1 + leading-zero run; all-zero suffix maps to max_width + 1
    if not (0 <= w < (1 << max_width)):
        raise ValueError("w does not fit in max_width bits")
    return max_width - w.bit_length() + 1

def alpha(m):
    # bias correction: the constant that makes E asymptotically unbiased
    if m == 16:  return 0.673
    if m == 32:  return 0.697
    if m == 64:  return 0.709
    return 0.7213 / (1.0 + 1.079 / m)        # m >= 128

class HyperLogLog:
    def __init__(self, p=11):
        # m = 2**p registers; standard error is about 1.04/sqrt(m)
        if not (4 <= p <= 16):
            raise ValueError("32-bit hash range requires 4 <= p <= 16")
        self.p = p
        self.m = 1 << p
        self.alpha = alpha(self.m)
        self.M = [0] * self.m                 # registers init to 0 (usable at small n)

    def add(self, value):
        x = hash32(value)
        j = self._bucket_index(x)
        w = self._remaining_bits(x)
        obs = self._observable(w)
        self.M[j] = self._combine_into_bucket(self.M[j], obs)

    def merge(self, other):
        if self.p != other.p:
            raise ValueError("precisions must match")
        self.M = [
            self._combine_buckets(a, b)
            for a, b in zip(self.M, other.M)
        ]

    def _bucket_index(self, x):
        suffix_width = HASH_BITS - self.p
        return x >> suffix_width              # first p bits pick the bucket

    def _remaining_bits(self, x):
        suffix_width = HASH_BITS - self.p
        return x & ((1 << suffix_width) - 1)  # remaining bits carry the observable

    def _observable(self, w):
        return rho(w, HASH_BITS - self.p)     # one-based leftmost-1 position

    def _combine_into_bucket(self, current, obs):
        return max(current, obs)              # bucket stores the MAX rho it has seen

    def _combine_buckets(self, left, right):
        return max(left, right)               # merge by unioning maxima

    def estimate(self):
        m = self.m
        # harmonic mean of the 2**M[j], via the indicator Z = 1 / sum(2**-M[j])
        Z = 1.0 / sum(2.0 ** -mj for mj in self.M)
        E = self.alpha * m * m * Z            # raw harmonic estimate

        if E <= 2.5 * m:                      # small-range: harmonic estimator distorts
            V = self.M.count(0)               # number of empty registers
            if V != 0:
                return m * math.log(m / V)    # linear counting from empty-bucket fraction
            return E                          # no empty buckets: raw estimate is fine
        if E <= HASH_RANGE / 30.0:            # middle band: no correction
            return E
        if E >= HASH_RANGE:
            return float("inf")               # the 32-bit hash range is saturated
        return -HASH_RANGE * math.log1p(-E / HASH_RANGE)  # large-range: undo hash collisions

    def count(self):
        return self.estimate()
```

So the chain is: exact counting is linear, kill it; sampling can't see cardinality through replication, so hash to make every observable set-only and replication-blind; long leading-zero runs in uniform hashes thermometer log₂ n, but one such reading has σ over a bit, so split the hash into m independent buckets and average — stochastic averaging — to get 0.78/√m on full bitmaps; but a bitmap per bucket is L bits to recover one number, so keep only the max leading-zero run per bucket, a log-log-sized register, which is LogLog at 1.30/√m; and that 1.30 is the fat right tail of 2^{max} leaking through the geometric mean, so replace it with the harmonic mean — Z = (Σ 2^{-M[j]})^{-1}, E = α_m m² Z — which discounts the freak-large buckets and pulls the error down to 1.04/√m at the ~1/√m floor, in 5-bit registers; finally patch the two ends, linear counting from empty buckets when n is small and a collision-inversion when n nears the hash range, and the estimate is Gaussian within 1.04/√m of the truth.
