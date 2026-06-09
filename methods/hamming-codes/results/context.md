# Context — Detecting Errors Is Not Enough; Can a Machine Locate and Fix Them?

## Research question

A large-scale digital computer runs a single long computation that passes through the same
relays, again and again, many thousands of times before an answer appears. Unlike a telephone
central office — where many parallel, more-or-less independent paths mean one bad relay only
produces an occasional wrong number while everything else keeps working — a digital machine
has one serial path: a single component failure usually means complete failure. If the failure
is detected, no more computing can be done until it is located and fixed; if it escapes
detection, it silently invalidates every subsequent operation.

The machines of the day already *detect* errors. The Model 5 relay computers built for the
Aberdeen Proving Grounds carried self-checking codes; observed failure rates ran about two or
three relay failures per day across the 8,900 relays of the two machines — roughly one failure
per two to three million relay operations. Self-checking meant these failures never introduced
*undetected* errors. But the machines were run unattended over nights and weekends, and a
detected error simply *halted* the computation (after a few automatic retries the machine
would abandon the problem and move on). A failure early on a Friday night could waste the
entire weekend's unattended run.

So the precise question is the step *beyond* detection: error **correction**. Given a block of
binary digits sent (or stored, or carried through a computation) over an unreliable channel
where any single bit may flip, can the receiving end not only tell *that* an error occurred but
*locate which bit* flipped and restore it — automatically, with no retransmission, and at a
redundancy cost far below the obvious "send three copies and vote" — while staying exact for
every input?

## Background

**The cost of a single failure in a serial machine.** The motivating observation is structural
and empirical at once: in a telephone office, parallel independent paths localize the damage of
a fault; in a digital computer, one long serial path through shared equipment means a single
fault is fatal to the whole run. With unattended operation the cost is not a wrong digit but a
wasted block of machine time — the computation halts and the standby effort to restart it is
expensive. This is the regime where paying redundancy to *correct* errors first becomes worth
it: (a) unattended operation over long periods with minimal standby equipment; (b) very large,
tightly interrelated systems where one failure incapacitates everything; (c) signaling in noise
that cannot economically be reduced.

**Binary is the natural alphabet.** The equipment — open/closed relays, flip-flops, dots and
dashes, perforated tape — represents information as sequences of 0s and 1s. So a code symbol is
a fixed-length string of bits, and the only thing that can go wrong in the simplest model is
that a bit flips. Familiarity with the binary number system (not common in 1947) turns out to
matter a great deal in what follows.

**Detection-only codes already in service.** Several codes were designed to *detect* isolated
errors. The 2-out-of-5 codes used in common-control switching and in the Bell relay computers
represent a decimal digit by exactly two "up" out of five relays — C(5,2)=10 patterns — and
flag any block in which not exactly two relays are up. The 3-out-of-7 code, another
constant-weight detection code, was used in radio telegraphy. Telegrams carried a word count at
the end. On the relay machines, when such a check failed the machine would, unattended, repeat
the step up to three times and then drop the problem. These catch a class of errors but only
ever say *something is wrong* — never *which bit*, and never *fix it*.

**The parity check, examined to its fundamentals.** A parity check appends one bit so that the
total number of 1s over a chosen set of positions is even. A single bit flip among those
positions changes the count from even to odd, so the check fails — detecting any single error
(indeed any odd number of errors). Two properties are worth recording: a parity check
is only a *detector* by itself, and a parity check need not cover all the
positions — it may be taken over selected positions only. Under the white-noise channel model
(each position equally and independently likely to flip with small probability p, so np small),
single errors dominate multiple errors, which is why a single-error-correcting scheme is the
right first target. How long to make the block n is an engineering judgment: small blocks spend
many check bits per information bit; large blocks lower that cost but raise the chance of an
uncorrectable multiple error.

**The expensive way to correct, already known.** Correction itself was not unimaginable: build
three copies of the machine, run them in lockstep with comparing circuits, and take the
majority vote. That *does* correct single failures — but at more than 200% overhead in
hardware. It sets the bar a real solution must beat: correction at a redundancy that grows
slowly, not by tripling everything.

**Redundancy as the figure of merit.** For a systematic code, each symbol has n binary digits,
of which m carry information and k = n − m are used for detection/correction. The redundancy is
R = n/m, the ratio of bits used to the minimum needed; it measures the channel-capacity cost of
the protection. The whole game is: achieve the desired correcting power at the least redundancy.

**The geometric picture latent in "a string of bits".** A natural representation of a length-n
bit string is a vertex of the unit n-dimensional cube. A single error moves the string along
one edge (one coordinate changes), two errors along two edges, and so on. So the number of
coordinates in which two strings differ is exactly the least number of edges between them — a
distance. This is the standard L1 distance on the cube; it satisfies non-negativity, identity,
symmetry, and the triangle inequality, i.e. it is a genuine metric. Around any point one can
speak of a sphere of a given radius (all vertices at that distance). This vocabulary is sitting
there, ready to be used, before any specific code is built.

**Prior art in correction.** The only previously printed work specifically on error *correction*
is M. J. E. Golay's short 1949 IRE note on digital coding. Everything else in service is
detection only.

## Baselines

**Plain parity (single-error detection).** Core idea: append one bit so the whole symbol has an
even number of 1s. Algorithm: check the parity of the received symbol; odd ⇒ at least one
error. Cost: one check bit; redundancy n/(n−1). Gap: it *detects* a single error but cannot say
which position flipped, so it cannot correct; and a double error passes undetected. It is the
right primitive but only half the job.

**m-out-of-n detection codes (2-out-of-5, 3-out-of-7).** Core idea: only the patterns with a
fixed number of 1s are legal symbols. Algorithm: count the 1s; wrong count ⇒ error, ask for a
retry. Cost: representational overhead (5 relays for a decimal digit). Gap: detection only; on
failure the machine can only retry or abandon — no localization, no in-place repair, and the
weekend-halting problem comes straight from here.

**Triple modular redundancy + majority vote.** Core idea: send/compute three independent copies;
where they disagree, the majority is taken as correct. Algorithm: replicate threefold, compare,
vote. Cost: more than 200% redundancy (R > 3). Gap: it *does* correct single errors, but the
cost is enormous and constant per bit; it does not exploit any structure of the message. Any
real solution must achieve correction with redundancy that grows like the logarithm of the
message length, not like a constant factor of 3.

**Residue/replication-style shortcuts in general.** Core idea: lean on some representation in
which the cost looks cheaper. Gap: these either don't localize errors or pay elsewhere; none
gives single-error *location and repair* at low redundancy.

## Evaluation settings

The operands are fixed-length binary blocks (code symbols) of length n carrying m message bits
and k = n − m check bits, over a white-noise bit-flip channel: each of the n positions flips
independently with small probability p (np small, so single errors dominate). The
yardsticks: (1) **redundancy** R = n/m (equivalently the number of check bits k for a given m) —
to be minimized; (2) **correcting/detecting power** — single-error correction is the primary
target, with single-error-correction-plus-double-error-detection as the next rung; (3)
**exactness** — the decoder must recover the original message for every input that suffers at
most the number of errors the code claims to handle, and must do so by computation at the
receiver, with no retransmission. The natural analytic instruments are elementary counting and
the geometry of the unit n-cube under
the L1 (bit-difference) metric — minimum distance between legal symbols, and packing of disjoint
spheres around them.

## Code framework

The primitives already available: bits as 0/1 lists; XOR (addition mod 2, i.e. `1 + 1 = 0`) as
the arithmetic of parity; the ability to take a parity check over *any chosen subset* of
positions; and binary representations of integers. The scaffold is an encoder that places m
message bits and some check bits into an n-bit codeword, and a decoder that inspects the
received word and (somehow) both flags and repairs a corrupted bit. How the check bits are
chosen and arranged, and how the decoder turns what it observes into a repair, is left open.

```python
def encode(data_bits, detect_double=False):
    m = len(data_bits)
    # TODO: how many check bits k? n = m + k.
    k = ...                      # TODO
    n = m + k
    code = [0] * (n + 1)         # 1-indexed positions 1..n
    # TODO: place data bits and check bits into positions 1..n
    # TODO: set each check bit from a parity over some chosen subset of positions
    if detect_double:
        pass                     # TODO: add one extra check position if useful
    return code[1:]


def decode(codeword, detect_double=False):
    code = [0] + list(codeword)
    # TODO: run the parity checks; combine the pass/fail pattern into a verdict
    # TODO: if a single bit is bad, identify which and flip it
    if detect_double:
        pass                     # TODO: separate a single bad bit from two bad bits
    return data_bits, status     # TODO
```
