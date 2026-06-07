# HAC §14.3.3 Barrett reduction — verbatim excerpt (source: hac_ch14.pdf, CACR Waterloo official copy)

Source: Menezes, van Oorschot, Vanstone, *Handbook of Applied Cryptography*, Ch.14,
downloaded this run from https://cacr.uwaterloo.ca/hac/about/chap14.pdf (saved as refs/hac_ch14.pdf / refs/hac_ch14.txt).

## Precomputation
µ = ⌊b^{2k} / m⌋. Advantageous when many reductions share one modulus m (e.g. RSA: every
encryption for an entity reduces mod that entity's public modulus). b is chosen near the
processor word size; assume b > 3.

## Algorithm 14.42 (Barrett modular reduction)
INPUT: x = (x_{2k-1}···x_0)_b, m = (m_{k-1}···m_0)_b with m_{k-1} ≠ 0, and µ = ⌊b^{2k}/m⌋.
OUTPUT: r = x mod m.
1. q1 ← ⌊x / b^{k-1}⌋, q2 ← q1·µ, q3 ← ⌊q2 / b^{k+1}⌋.
2. r1 ← x mod b^{k+1}, r2 ← q3·m mod b^{k+1}, r ← r1 − r2.
3. If r < 0 then r ← r + b^{k+1}.
4. While r ≥ m do: r ← r − m.
5. Return r.

## Fact 14.43 (the error bound on the quotient estimate)
By the division algorithm there exist Q, R with x = Qm + R, 0 ≤ R < m. In step 1:
    Q − 2 ≤ q3 ≤ Q.
So q3 NEVER exceeds the true quotient Q, and is at most 2 too small.

## Note 14.44 (correctness)
(i) ⌊x/m⌋ = Q = ⌊ (x/b^{k-1})(b^{2k}/m)(1/b^{k+1}) ⌋ ; approximated by
    q3 = ⌊ ⌊x/b^{k-1}⌋ · µ / b^{k+1} ⌋. By 14.43, q3 ≤ Q, at most 2 below.
(ii) −b^{k+1} < r1 − r2 < b^{k+1}; r1 − r2 ≡ (Q − q3)m + R (mod b^{k+1});
     0 ≤ (Q − q3)m + R < 3m < b^{k+1} since m < b^k and 3 < b.
     If r1−r2 ≥ 0 then r1−r2 = (Q−q3)m+R; else r1−r2+b^{k+1} = (Q−q3)m+R.
     Either way step 4 runs AT MOST TWICE, since 0 ≤ r < 3m.

## Note 14.45 (efficiency)
(i) All the divisions are simple right-shifts of the base-b representation.
(ii) q2 only feeds q3; the k+1 least-significant digits of q2 are not needed → partial
     multiply suffices. Computing q3 takes at most (k+1)^2 − k^2 = (k²+5k+2)/2 single-precision mults.
(iii) r2 = q3·m mod b^{k+1} is also a partial multiply → at most (k+1 choose 2)+k single-precision mults.

## Example 14.46
b=4, k=3, x=(313221)_b=3561, m=(233)_b=47. µ=⌊4^6/47⌋=87=(1113)_b.
q1=⌊x/4^2⌋=(3132)_b, q2=(3132)_b·(1113)_b=(10231302)_b, q3=(1023)_b,
r1=(3221)_b, r2=(1023)_b·(233)_b mod 4^4=(3011)_b, r = r1−r2 = (210)_b → x mod m = 36. ✓ (3561 mod 47 = 36)
