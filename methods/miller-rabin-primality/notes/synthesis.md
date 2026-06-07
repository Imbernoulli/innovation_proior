# Synthesis: Miller-Rabin primality test

## Problem
Decide whether a large odd integer n is prime, quickly and reliably, without factoring it.
Trial division is O(sqrt(n)) = exponential in the bit-length. Need polynomial-in-log-n.

## Lineage / ancestors
- **Fermat's little theorem (FLT):** if n prime and gcd(a,n)=1, a^(n-1) ≡ 1 mod n. Contrapositive
  gives the Fermat test: if a^(n-1) != 1 mod n then n composite ("a is a Fermat witness").
  Compute a^(n-1) by fast modular exponentiation (binary/square-and-multiply): O(log n) mults.
- **Carmichael numbers** defeat Fermat: composite n with a^(n-1) ≡ 1 mod n for ALL a coprime to n.
  Smallest is 561 = 3·11·17. There are infinitely many (Alford-Granville-Pomerance 1994). For such n
  the Fermat nonwitnesses are the WHOLE unit group (Z/nZ)*, so no proportion bound holds; the only
  Fermat witnesses are the non-coprime a (a tiny fraction). Fermat test is unreliable.
- **Why Fermat nonwitnesses fail to be a proper subgroup:** Fermat nonwitnesses {a : a^(n-1)≡1} form a
  group; for non-Carmichael composite n it's PROPER (so <=50% nonwitnesses), but for Carmichael n it's
  the entire unit group. That is the precise hole.
- **Solovay-Strassen (1977):** Euler criterion a^((n-1)/2) ≡ Jacobi(a/n) mod n. Euler nonwitnesses ALWAYS
  form a proper subgroup of units for composite n -> >=50% witnesses, no Carmichael-style failure. But
  needs Jacobi symbols, and the bound is only 1/2.
- **Miller (1976):** deterministic test assuming GRH; a Miller witness exists up to O((log n)^2)
  (Bach later: <= 2(log n)^2). Deterministic poly-time CONDITIONAL on GRH.
- **Rabin (1980) / Monier (1980):** proved >=75% of bases are witnesses for composite n, unconditionally,
  turning Miller's test into a randomized algorithm with error <= 4^(-k). Knuth reformulated in the
  a^d, a^(2d),... form. Selfridge had the idea ~2 years before Miller (unpublished).

## The key insight chain
1. Fermat fails on Carmichael n because a^(n-1)≡1 leaves NO leverage. Need extra structure.
2. **Nontrivial square root of 1.** For prime n, the only solutions of x^2 ≡ 1 mod n are x ≡ ±1
   (because n | (x-1)(x+1) and prime n must divide one factor). For composite n with >=2 distinct prime
   factors, CRT gives extra square roots of 1 (e.g. mod 15: x ≡ ±1 mod 3 and ±1 mod 5 give 1,4,11,14).
   Finding x with x^2≡1 but x != ±1 PROVES n composite — even works on Carmichael numbers, since a
   Carmichael number has >=3 prime factors, hence many nontrivial sqrt(1).
3. **How to find one for free.** Write n-1 = 2^e·k, k odd (e>=1 since n odd). Then
   a^(n-1) = (a^k)^(2^e). Look at the sequence a^k, a^(2k), a^(4k), ..., a^(2^(e-1)k), a^(2^e k)=a^(n-1).
   Each term is the square of the previous. If n prime then FLT says the last term is 1, and (factoring
   x^(2^e k)-1 = (x^k-1)(x^k+1)(x^(2k)+1)...(x^(2^(e-1)k)+1)) one factor is ≡0, i.e.
   a^k ≡ 1 mod n  OR  a^(2^i k) ≡ -1 mod n for some i in {0,...,e-1}.  (*)
   If (*) FAILS, then somewhere in the sequence a value !=1 is squared to give 1 without being -1:
   a nontrivial square root of 1. Such a is a "Miller-Rabin witness"; n is composite.
4. **Witness definition.** a is a witness iff a^k != 1 mod n AND a^(2^i k) != -1 mod n for all
   i in {0,...,e-1}. Equivalently the MR-sequence neither starts with 1 nor contains -1.
5. **The 3/4 theorem (Monier-Rabin).** For odd composite n>1, more than 75% of a in {2,...,n-2} are
   witnesses; i.e. nonwitnesses < 25%. Hence k random rounds err with prob <= 4^(-k).

## Proof structure (grounded in Conrad's notes)
- Nonwitnesses lie in a PROPER subgroup of (Z/nZ)*  => over 50% witnesses (clean proof).
  - Prime-power case n=p^α, α>=2: nonwitnesses = {a : a^(p-1)≡1 mod p^α} (Thm 3.4), a group of order p-1,
    while |units| = ϕ(p^α)=p^(α-1)(p-1) > p-1. Proper.
  - Non-prime-power case: let i0 be the LARGEST i in {0,...,e-1} with some a0, a0^(2^i k)≡-1 mod n.
    (i0 exists: a0=-1 gives i=0.) Define G_n = {a : a^(2^i0 k) ≡ ±1 mod n}; it's a group and contains
    every nonwitness. It's PROPER: by CRT pick a ≡ a0 mod p^α, a ≡ 1 mod n' (n=p^α n', n'>1).
    Then a^(2^i0 k) ≡ -1 mod p^α (so !=1 mod n) and ≡ 1 mod n' (so !=-1 mod n) => a not in G_n.
- Sharpen 1/2 to 1/4: show ϕ(n)/|G_n| >= 4 when n not a prime power.
  - Prime power: nonwitness proportion = (p-1)/(p^α-1) = 1/(1+p+...+p^(α-1)) <= 1/(1+p) <= 1/4
    (equality only at n=9). 
  - n NOT Carmichael, >=2 prime factors: F_n={a:a^(n-1)≡1} sits units ⊃ F_n ⊃ G_n, both strict
    => ϕ(n)/|G_n| = (ϕ/|F_n|)(|F_n|/|G_n|) >= 2·2 = 4.
  - n Carmichael (>=3 distinct prime factors r>=3): map f: H_n -> ∏_{l=1}^r {±1 mod p_l^α_l},
    f(a)=(a^(2^i0 k) mod p_l^α_l). f surjective (CRT + def of i0) so |image|=2^r; f(G_n)={(1,..),(−1,..)}
    order 2; hence |H_n|/|G_n| = 2^(r-1) >= 4. So ϕ(n)/|G_n| >= |H_n|/|G_n| >= 4.
- 75% is asymptotically sharp (Monier): n=p(2p-1), p≡3 mod 4 both prime -> proportion -> 75%.

## Design decisions -> why
- Use n-1=2^e k decomposition (not random sqrt): gives the square-root chain for FREE inside the same
  modular exponentiation that the Fermat test already computes; no extra cost.
- Check a^d first, then square e-1 times watching for n-1: directly tests conditions (*).
- random a in [2,n-2]: 1 and n-1 are ALWAYS nonwitnesses, exclude them.
- k rounds independent: error <= 4^(-k); k=20 -> 4^-20 ~ 10^-12; k=40 cryptographic.
- Deterministic small witness sets: by exhaustive search, for n below explicit bounds a FIXED small set
  of bases catches all composites. {2,3,5,7,11,13,17,19,23,29,31,37} is correct for all n < 2^64.
  First composite passing bases {2,3} is 1373653; passing {2,3,5,7} is 3215031751.
- binary modular exponentiation (square-and-multiply): a^d mod n in O(log d) modular mults; total cost
  O(k log^3 n) with schoolbook mult (each mult O(log^2 n)).

## Code framework (scaffold) -> final
Scaffold (pre-method): modmul/modpow primitive exists; a generic `is_probably_prime(n)` stub with a
single congruence test slot. Final fills: decompose n-1=2^e k; the witness-round function
(a^k, then square loop checking ±1); the driver picking k random bases or a deterministic base list.

## Uncertainty flags
- "ϕ(n) < n-1 since n composite": true (some a in [1,n-1] not coprime). Fine.
- The exact small-witness bounds verified against Wikipedia list; smallest strong pseudoprime to {2,3}
  =1373653 cross-checked in Conrad. Good.
