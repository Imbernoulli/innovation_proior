# Beacon Hill: Phasing Periodic Transmissions on a Shared Channel

A hilltop relay hub repeats forever with period **M**.  **n** beacon jobs share
its single channel.  Job **i** fires a transmission burst of length **d_i**
once every **p_i** instants: choosing a phase offset **o_i** (0 ≤ o_i < p_i)
makes it transmit during the half-open intervals

  [o_i + k·p_i ,  o_i + k·p_i + d_i)      for k = 0, 1, …, M/p_i − 1,

where the burst may wrap around inside its own period (residues (o_i+τ) mod p_i).
Every period p_i divides M, so the whole system is periodic with horizon M and
it suffices to study one horizon.

Every instant t (0 ≤ t < M) is an **audit instant** with a posted interference
price w[t] (given in the input).  If load(t) jobs transmit at instant t, the
channel keeps one of them and the rest are lost to interference; the hub is
charged

  waste(t) = w[t] · max(0, load(t) − 1).

**Total waste F = Σ_t waste(t).  Minimize F.**

## Input (stdin)

```
M n
p_1 d_1
...
p_n d_n
w_0 w_1 ... w_{M-1}
```

All values are integers.  1 ≤ d_i < p_i, p_i | M, 0 ≤ w[t] ≤ 64.

## Output (stdout)

n integers o_1 … o_n — a phase offset per job, in input order, 0 ≤ o_i < p_i.
Any such assignment is feasible; only its waste is graded.

## Scoring

The checker simulates the horizon exactly (integer arithmetic) and computes F.
It also computes the baseline B = waste of the all-zero phase assignment
(o_i = 0 for all i), and reports

  Ratio = min(1.0, 0.1 · B / F)        (F = 0 scores Ratio 1.0)

Lower waste ⇒ higher ratio.  The all-zero assignment itself scores ≈ 0.1.

## Structure you can exploit

Two jobs i, j can interfere **only** at instants t with
t ≡ o_i (mod p_i) and t ≡ o_j (mod p_j); by the Chinese Remainder Theorem such
instants exist iff o_i ≡ o_j (mod gcd(p_i, p_j)), and then occur every
lcm(p_i, p_j).  So collision patterns are governed entirely by residue classes:
jobs whose periods share a large gcd are inherently collision-prone when their
phases align, and a job with d_i = 1 and g | p_i can keep *all* of its bursts
inside a single residue class mod g.  The posted prices w[t] are far from
uniform — read them and reason about *which* residue classes your unavoidable
overlaps should live in.

## Constraints

- M ≤ 14000, n ≤ 300.  Time limit 5 s, memory 512 MB.
- Scoring is exact and deterministic.

## Example

M = 24, jobs A(p=8,d=1), B(p=8,d=1), C(p=12,d=2), D(p=6,d=1), and prices
w[t] = 5 when t ≢ 3 (mod 4), w[t] = 1 when t ≡ 3 (mod 4).

All-zero phases: A and B both fire at {0,8,16}, C covers {0,1,12,13}, D fires
at {0,6,12,18}.  Colliding instants: t=0 (load 4 → 3·5), t=8,16 (A∩B → 5
each), t=12 (C∩D → 5).  F = 15+5+5+5 = 30 → Ratio 0.1.

Phases A→3, B→7, C→4, D→2 give bursts
A={3,11,19}, B={7,15,23}, C={4,5,16,17}, D={2,8,14,20}: every instant has load
≤ 1, F = 0 → Ratio 1.0.  Note A and B, with gcd 8, were staggered into
*distinct* residues mod 8 (3 vs 7), both inside the cheap class mod 4 — that is
residue-class tiling.
