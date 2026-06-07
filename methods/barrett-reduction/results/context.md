# Context: fast modular reduction for RSA on a general-purpose signal processor

## Research question

We want to run the RSA public-key cipher at usable speed on an inexpensive, off-the-shelf
processor. RSA encryption and decryption both reduce to one operation: modular exponentiation,
c = A^E mod M, where M and E are several hundred bits long (512 bits is the target). The standard
square-and-multiply method turns this into a sequence of full-length multiplications, each followed
by a reduction modulo M. For a 512-bit exponent that is on the order of a thousand multiply-and-reduce
steps. The multiply we can make fast. The bottleneck is the reduction: computing W mod M for a
double-length product W. Done naively, each reduction is a multiple-precision division by M, and
division is the single most expensive primitive on the kind of cheap hardware we can afford. The
question is whether the reduction can be reorganized so that it, too, is built only out of the cheap
operations the hardware does well — multiplies, shifts, subtractions — and never a true division. A
solution has to be exact (RSA is unforgiving of off-by-one errors), it has to handle operands far
wider than a machine word, and it has to be fast enough that the reductions stop dominating the
exponentiation.

## Background

The field state at the time: RSA (Rivest, Shamir, Adleman, CACM 1978) is known and its security is
taken seriously, but it is widely regarded as too slow for practical deployment on cheap processors.
The cost model is what matters. On the candidate platforms — an 8-bit micro, a 16-bit micro, discrete
logic, a bit-slice engine — a 512-bit exponentiation was estimated at minutes (≈4 min on an 8-bit
part, ≈50 s on a 16-bit part), which is unusable. A dedicated multiplier/accumulator (MAC) chip,
e.g. a 100 ns 16×16 unit, is fast at the multiply but needs custom hardware around it to feed
operands. The one attractive option is a new class of part: a digital signal processor that packs a
single-cycle 16×16 multiplier-accumulator together with a microprocessor core and on-chip memory on
one chip (the Texas Instruments TMS32010, ≈200 ns cycle), so a multiply costs essentially the same as
any other instruction and operand pointers can auto-increment during a multiply-accumulate, making
the data fetching for a long multiply effectively free.

The load-bearing facts about arithmetic on such a machine:

- **Schoolbook multiplication is O(n²) but cheap per step.** An n×n digit product is the
  paper-and-pencil algorithm: for each digit of one operand, multiply through the other and add the
  shifted partial product. Row-wise this is ≈6n² instructions (the multiplies plus all the fetches,
  stores, and carry handling); reorganized to work column by column, accumulating a whole column
  before resolving carries, it drops to ≈4n² (a third less work); and on a processor whose
  multiply-accumulate auto-increments its pointers, the operand fetches cost nothing, roughly halving
  it again. So a multiply is a tight inner loop of single-cycle multiply-accumulates.

- **Division is hard.** Multiple-precision division has no such cheap inner loop. The classical
  schoolbook division (Knuth, *The Art of Computer Programming* Vol. 2, §4.3.1, Algorithm D) is also
  O(n²), but each quotient digit must be *estimated* from the leading words of the running remainder
  and the divisor, the divisor must be multiplied through and subtracted, and the estimate is
  sometimes wrong and must be corrected with an add-back of the divisor. The trial-quotient estimate
  can be too large by at most 2, with the correction firing rarely (probability ≈ 3/b for radix b).
  Crucially, because the divisor changes from problem to problem, this estimation work is redone at
  every quotient position. There is no single cheap primitive a signal processor can lean on; it is
  branchy, serial, and slow.

- **A reciprocal can be precomputed when the divisor is fixed.** Inside one RSA exponentiation the
  modulus M never changes — every one of the ~thousand reductions divides by the *same* M. Dividing
  by a constant is the same as multiplying by its reciprocal, and the reciprocal of a fixed M can be
  computed once, ahead of time, and amortized over all the reductions (it can even be stored
  alongside M as part of the key). The reciprocal 1/M is a real number less than one, so to use it
  with integer hardware it has to be put into fixed-point form: scale it up by a power of the radix
  and round to an integer. Rounding *down* (taking a floor) makes the scaled reciprocal a slight
  underestimate of 1/M — a fact that will determine which direction the unavoidable correction goes.

- **The whole development is done under a correctness discipline.** Because an arithmetic bug in RSA
  is silent and catastrophic, the program is derived by stepwise refinement in Dijkstra's
  guarded-command notation with Gries-style pre/post-condition reasoning, each refinement provably
  equivalent to the last, rather than hand-coded directly in assembler.

## Baselines

- **Square-and-multiply exponentiation with reduction at each step (Knuth §4.6.3).** To compute
  A^E mod M, scan the bits of E, squaring a running accumulator at every bit and additionally
  multiplying in A at every 1-bit, reducing mod M after each multiply to keep operands bounded. This
  is the harness everything plugs into; its inner cost is one long multiply plus one reduction per
  bit. It leaves the reduction unspecified — and that is exactly the open slot.

- **Schoolbook (classical) long division for the reduction.** Compute W mod M directly as
  W − M·⌊W/M⌋ using Algorithm D. Correct and general, but it is the expensive path this whole effort
  is trying to avoid: per-position quotient-digit estimation, multiply-back, subtract, and occasional
  add-back correction, all O(n²) of branchy serial work with no cheap multiply-accumulate inner loop.
  Its gap: it pays full division cost on *every* reduction even though the divisor M is the same every
  time, so it never amortizes the one thing that is constant.

- **Montgomery's division-free modular multiplication (Montgomery, *Math. Comp.* 44(170):519–521,
  1985).** A contemporaneous route to the same goal of "modular reduction with no division by M."
  Represent each residue x not directly but as the N-residue xR mod N, for a radix R coprime to N
  with R > N and R a power of the base (so reduction mod R and division by R are just truncations and
  shifts). Precompute N′ with R·R⁻¹ − N·N′ = 1. Then the reduction step is REDC(T) for 0 ≤ T < RN:
  set m ← (T mod R)·N′ mod R, t ← (T + mN)/R, and return t−N if t ≥ N else t. One verifies mN ≡ −T
  (mod R) so t is an integer, tR ≡ T (mod N) so t ≡ T·R⁻¹ (mod N), and 0 ≤ t < 2N so a single
  conditional subtraction suffices. It replaces the division by N with one multiply mod R, one
  multiply by N, an add, and a shift. Its gaps relative to our setting: it computes T·R⁻¹ mod N, not
  T mod N, so it only pays off if you convert your operands into the N-residue domain once and stay
  there for the whole exponentiation (converting in and out via multiply-by-R mod N); it requires N
  odd (the coprimality with R = power of two), so a general modulus needs care; and it touches the
  *low* words of T through the N′ multiply. It is an excellent fit when you live in its residue
  domain, and a poor fit when you want a single ordinary reduction in the ordinary integer domain.

## Evaluation settings

The natural yardstick is a full RSA exponentiation with a 512-bit modulus and a 512-bit exponent,
timed on a first-generation DSP (TMS32010 at its 20 MHz maximum clock, ~200 ns cycle), with an
average-case exponent taken to be half 0-bits and half 1-bits (so roughly 1.5 long-multiply-plus-
reductions per bit). The relevant cost metrics are the instruction count and time per modular
reduction, the count per reduction of single-precision multiply-accumulates versus true divisions,
and the number of correction steps needed after the estimated quotient. The reduction subroutine has
the precondition that its input W satisfies 0 ≤ W < M², which is exactly what a product of two
already-reduced operands (each < M) supplies, so the reduction is always applied to a double-length
input. Second-generation parts (TMS320C25; Motorola DSP56200 with a 24×24 multiplier and 56-bit
accumulator) are natural portability targets because the same instruction-level cost model applies.

## Code framework

The pieces that already exist: a square-and-multiply exponentiation harness, a multiple-precision
representation of integers as arrays of base-b digits (b near the word size), and a fast long-multiply
routine built from the processor's multiply-accumulate. The per-modulus reducer is the empty slot.

```python
# Multiple-precision integers are sequences of base-b digits; b ~ machine word size.
# A fast long-multiply already exists (column method + multiply-accumulate inner loop):
def long_mult(u, v):
    """Return u * v exactly.  pre: 0 <= u, v < b**n."""
    pass

class ModularReducer:
    """Per-modulus reduction object for double-length inputs."""

    def __init__(self, mod):
        if mod <= 0:
            raise ValueError("modulus must be positive")
        self.modulus = mod
        # TODO: precompute constants tied only to mod.
        pass

    def reduce(self, x):
        """Return x mod self.modulus.  pre: 0 <= x < self.modulus**2."""
        # TODO: use multiplies, shifts, and subtractions instead of a
        #       multiple-precision division by self.modulus.
        pass

def fastexp(A, E, M):
    """c = A**E mod M by square-and-multiply, reducing after every multiply."""
    if not 0 <= A < M:
        raise ValueError("base must already be reduced")
    if E < 0:
        raise ValueError("exponent must be non-negative")
    reducer = ModularReducer(M)
    a, c, e = A, 1, E
    while e > 0:
        if e & 1:
            c = reducer.reduce(long_mult(c, a))
        e >>= 1
        if e:
            a = reducer.reduce(long_mult(a, a))
    return c
```
