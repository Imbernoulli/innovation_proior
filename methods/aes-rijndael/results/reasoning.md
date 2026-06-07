Let me start from what actually hurts, not from a cipher I want to build. The incumbent block cipher is dying for two reasons, and they're different in kind. The first is brute force: 56 effective key bits means the key space is 2^56, and that's already exhaustible by purpose-built hardware in tens of hours. That one is easy to fix in principle — take a 128-bit key and exhaustive search needs 2^128 work, which is outside any foreseeable brute-force budget. So I'll fix the key length by fiat and not worry about it again; the block I'll make 128 bits too, so the birthday bound on block collisions is comfortable.

The second pain is the one I actually have to design around, and it's about *trust*. The incumbent's S-boxes were chosen against secret criteria. Years later, when differential cryptanalysis went public, those boxes turned out to be suspiciously well-suited to resisting it — which is a polite way of saying the designers knew an attack the rest of us didn't. The strength was real but it was unaccountable: I cannot look at the public structure and *argue* that it's secure, and I can't rule out that someone hid a weakness in tables I can't see the rationale for. I don't want to inherit that. I want a cipher where security against the known attacks falls out of the public structure, where a skeptic can re-derive the bound themselves. So the design target isn't just "strong," it's "demonstrably strong from what's on the page."

To get there, I need to put the two public attacks under the same microscope and see whether they ask for the same kind of structure.

Differential cryptanalysis. I take pairs of plaintexts differing by a fixed XOR difference a', push them through the cipher, and watch the output difference b'. If, over almost the whole cipher, some particular a' produces some particular b' with probability much larger than the single-trail feasibility cutoff 2^(1-n) on an n-bit block, that statistical handle lets me peel off key bits with a chosen-plaintext attack; the random one-output-difference probability is 2^-n, but the attack threshold is about 2^(1-n) because even a full codebook contains only about 2^(n-1) pairs with a fixed input difference. The incumbent went down to about 2^47 chosen plaintexts this way. Now, how does a difference actually propagate through an iterated cipher? Round by round. A *trail* is a fixed sequence of differences, one after each round, q^(0) → q^(1) → … → q^(r), and the probability that a pair follows that whole trail is the product of the per-round probabilities, because each round acts on the difference coming out of the last. The linear parts of a round — a key XOR, a linear mixing — move a difference deterministically: XOR-ing a constant cancels in the difference, and a linear map sends a' to L(a') with probability 1. So the *only* place a difference is uncertain is at the S-boxes. If an S-box sees input difference zero, it outputs difference zero with certainty — it costs nothing, it's passive. Only S-boxes with a *nonzero* input difference, the active ones, multiply in a factor below 1, and that factor is at most the S-box's worst-case difference propagation probability. So the probability of a trail is essentially (worst S-box factor) raised to the number of *active* S-boxes along it.

Linear cryptanalysis has, when I write it down carefully, the exact same skeleton. Now I'm looking for a linear equation — some parity of plaintext bits equals some parity of ciphertext bits XOR some parity of key bits — that holds with probability off from 1/2. Measure the bias by the correlation C = 2·Prob − 1. A linear trail is a sequence of bit-selection patterns through the rounds, and its overall correlation is the *product* of the per-step correlations (this is the piling-up of biases, stated cleanly). The linear layers and the key XOR just transform the selection pattern, contributing correlation ±1 — magnitude one, they don't shrink anything. An S-box whose selection patterns are zero in and out also contributes ±1: passive. An *active* S-box, one with a nonzero mask, contributes a factor of magnitude below 1, at most its maximum correlation. So a linear trail's correlation magnitude is essentially (worst S-box correlation) raised to the number of active S-boxes.

Stare at those two conclusions side by side. Differential strength: product over active S-boxes of a per-box probability < 1. Linear strength: product over active S-boxes of a per-box correlation magnitude < 1. They are governed by the *same* structural quantity — how many S-boxes any trail is forced to make active — and by how good a single S-box can be made (how far below 1 its worst factor is). That's the whole game, and it tells me exactly the two things to engineer, separately. One: an S-box whose worst-case difference probability and worst-case correlation are both as small as I can prove them to be. Two: a round structure that *forces a large minimum number of active S-boxes in every trail*, and forces it *provably*. If I get both, I multiply a small per-box factor a guaranteed-large number of times and the best trail over the cipher drops below the attack threshold, and I can show my work.

Notice that splits cleanly into "make each active S-box hurt as much as possible" (a local, nonlinear-layer problem) and "make every trail activate as many S-boxes as possible" (a global, linear-layer problem). I'll design those two layers to do those two jobs and not blur them together — let the nonlinear layer be purely about per-box quality and the linear layer be purely about spreading activity. That separation of concerns is going to be the spine of the thing.

Let me build the S-box first, because the linear-layer bound will be stated in units of "active S-boxes" and I want to know how good one box can be.

I need a bijection on bytes (8→8, invertible, so the cipher is invertible) with small differential uniformity — small max number of solutions x to S(x⊕a) ⊕ S(x) = b — and small maximum correlation. I could search for a good table by trial, the way the incumbent's designers presumably did. But that's exactly the move that lost public trust: a table whose virtue I can only *measure*, not *argue*. I'd rather have an S-box with an algebraic definition whose differential and linear quality I can *prove* and anyone can re-check.

What algebraic map on bytes is maximally nonlinear in the differential sense? Treat the byte as an element of GF(2^8) and think about x ↦ x^(-1), the multiplicative inverse, with 0 sent to 0 because 0 has no inverse. The differential equation is S(x+a) + S(x) = b for a nonzero. If b = 0, there are no solutions, because a bijection cannot map x+a and x to the same value. If b is nonzero and x is neither 0 nor a, the inverse equation becomes

  (x+a)^(-1) + x^(-1) = b
  a / (x(x+a)) = b
  x^2 + a x + a/b = 0.

Scale x = a y and I get y^2 + y = 1/(a b). Over GF(2^8), an equation y^2 + y = c has either zero or two solutions, depending on the trace of c. The two exceptional inputs x=0 and x=a both give output difference 1/a. For b=1/a, the scaled equation is y^2 + y = 1, and because the degree is even, Tr(1)=0, so there are two more non-exceptional solutions. For every other b, there are at most two. So the inverse map has maximum four solutions, hence maximum difference propagation probability 4/256 = 2^-6. A two-uniform APN box would be the ideal differential object, but this gives me a simple, invertible, algebraically provable 8-bit box with a very low worst-case probability.

The linear side has the same algebraic flavor. For nonzero input and output masks u and v, the correlation is 2^-8 times a Walsh sum

  sum_x (-1)^(Tr(u x + v x^(-1))).

After rescaling, this is a binary Kloosterman sum. In GF(2^8) those sums have magnitude at most 32, and the bound is attained, so the maximum normalized correlation is 32/256 = 2^-3. So the inverse map isn't just *good*, it's *provably near-optimal on both axes at once* — and that's the entire reason to reach for it instead of a hand-tuned table. I can write down why it resists both attacks; I don't have to ask anyone to trust me.

So let me tentatively set S(x) = x^(-1) in GF(2^8) and check whether I'm done. I'm not, and the reasons are about *structure*, not about differential/linear numbers. First, x^(-1) is an involution-flavored, extremely simple algebraic object: the whole S-box is one tidy equation y = x^(-1), i.e. x·y = 1. A cipher built from a single clean algebraic relation invites algebraic attacks — interpolation, expressing the cipher as a low-degree system — precisely because the S-box has a short algebraic description. Second, it has fixed points: 0 ↦ 0 and 1 ↦ 1, and that kind of regularity (and any opposite-fixed-point regularity, S(x) = x ⊕ 0xff) is the sort of pattern an attacker probes. It also just *looks* too clean, and "looks clean" is uncomfortably close to "has exploitable structure."

Affine equivalence gives me exactly the lever I need. The differential uniformity and the maximum correlation of an S-box are unchanged if I compose it with invertible affine maps. On the differential side, an input affine map just relabels input differences and an output affine map just relabels output differences, so the number of x solving each difference equation is unchanged. On the linear side, affine maps relabel masks and may flip a sign with the constant term, but they do not change the absolute Walsh coefficients. So I can take the algebraically optimal core, x^(-1), and post-compose it with an invertible GF(2)-affine transformation on the eight bits, and *the provable differential and linear quality is untouched* while the algebraic description gets scrambled. The affine map adds nothing an attacker can exploit differentially or linearly, but it destroys the simple "x·y=1" description, can be chosen to kill the fixed point and the opposite fixed point, and makes the boolean expression of each output bit messy. Best of both: provable resistance from the inverse, no clean algebra from the affine.

What affine map? An invertible 8×8 GF(2) matrix plus a constant byte. I'll pick the matrix to be a circulant — each row is the previous one rotated — because then the whole thing is "rotate the byte and XOR," which is cheap, and a circulant with the right pattern is invertible. Concretely each output bit is the input bit XOR-ed with four of its cyclic neighbors:

  b'_i = b̃_i ⊕ b̃_{(i+4) mod 8} ⊕ b̃_{(i+5) mod 8} ⊕ b̃_{(i+6) mod 8} ⊕ b̃_{(i+7) mod 8} ⊕ c_i,

with bit indices taken modulo 8 from the least significant bit. Here b̃ is the inverse-mapped byte and c is a constant byte I choose to remove fixed points. Taking c = {01100011} = 0x63 does it: with this affine map and this constant, there is no byte with S(x) = x and none with S(x) = x ⊕ 0xff. And the 0 input, which has no inverse — I just define b̃ = 0 for x = 0, then run it through the same affine map; the affine constant means 0 ↦ 0x63, not a fixed point, which is exactly what I wanted. Let me sanity-check one value against the construction: take x = 0x53. Its inverse in the field is 0xca; push 0xca through the affine map above and I get 0xed. And the trivial ones: 0x00 ↦ 0x63 (the constant), 0x01 ↦ 0x7c. Good — the two-step recipe reproduces a consistent table, and crucially I can hand someone the inverse-plus-affine and they can regenerate every entry, no secret criteria anywhere.

One S-box, used for every byte. I considered using different S-boxes in different positions, but there is no argument that it improves resistance to the attacks I care about, and a single box saves code in software and chip area in hardware. So: bricklayer the same S-box across all 16 bytes. Call this layer the byte substitution. Its job is done — every active byte through it costs a factor of at most 2^-6 differentially, at most 2^-3 in correlation, provably.

Now the hard half: the linear layer, whose only job is to make every trail activate a *lot* of S-boxes. Per box I have at best 2^-6 differential; to beat the 2^-127 threshold on a 128-bit block I need on the order of 22+ active S-boxes in the best trail across the rounds I'll attack. So I want the round structure to guarantee, say, 25 active S-boxes over a handful of rounds, and then iterate enough rounds to bury the attack with margin. The question is how to *force* activity.

Let me set up the state geometry, because the spreading argument will live on it. I have 16 bytes. The S-box is byte-wise, so "active" is a per-byte notion. Arrange the 16 bytes as a 4×4 array of bytes — four columns of four bytes each. Why a square? Because I'm going to need two complementary kinds of spreading — within small groups and across groups — and a 4×4 grid lets a "mix within a column" step interlock with a "permute across columns" step. Hold that thought.

First idea for diffusion: take each column of 4 bytes and apply a fixed linear map over GF(2^8) that mixes them, output 4 bytes from 4 bytes. I want this map to have the property that you can't have *both* few active input bytes and few active output bytes. Define, for a linear map θ on bundles (bytes here), its *branch number*:

  B(θ) = min over nonzero inputs a of [ wt(a) + wt(θ(a)) ],

where wt counts the *nonzero bytes*. Why is this the right measure? Picture two consecutive S-box layers with θ between them. The active S-boxes in the first layer are exactly the nonzero input bytes of θ; the active S-boxes in the second are exactly the nonzero output bytes of θ. So wt(a) + wt(θ(a)) is precisely the number of active S-boxes that any trail crossing this θ must light up across those two layers. The *minimum* of that over all nonzero differences is the worst case the attacker can aim for — and B(θ) is exactly that minimum. So the branch number of my column map *is* a guaranteed lower bound on active S-boxes over the two layers it sits between. I want it as large as possible.

How large can B be for a map on 4 bytes? If a has a single nonzero byte (wt = 1), then to make B big I need θ(a) to have all 4 bytes nonzero (wt = 4), giving 1 + 4 = 5. Can I beat 5? No: there is a Singleton-type ceiling. Think of the map x ↦ (x, θ(x)) as a code: it takes a 4-byte input to an 8-byte codeword (input concatenated with output), and B(θ) is the minimum nonzero Hamming weight of that code in bytes, i.e. its minimum distance. An [8,4] code over the byte alphabet has minimum distance at most 8 − 4 + 1 = 5. So B ≤ 5, full stop, and B = 5 is *optimal* — that's the Maximum Distance Separable (MDS) case. I want an MDS map: any single active input byte forces all four output bytes active, any 2-in forces ≥3-out, and so on, never fewer than 5 active total.

So I need a 4×4 matrix over GF(2^8) that's MDS — every square submatrix nonsingular. There are many; among them I want the one with the *cheapest* arithmetic, because I'm going to run it on four columns every round, including on small processors. Cheap means small coefficients: {01} is free (no multiply), {02} is one xtime (shift + conditional XOR with 0x1B), {03} = {02} ⊕ {01} is an xtime and an XOR. A circulant matrix with first row [{02},{03},{01},{01}] — equivalently treating the column as a polynomial and multiplying by the fixed polynomial {03}x^3 + {01}x^2 + {01}x + {02} modulo x^4 + 1 — uses only {01},{02},{03}. So each output byte of a column is, for the top entry, {02}·a_0 ⊕ {03}·a_1 ⊕ a_2 ⊕ a_3, and cyclically down.

I cannot certify branch number 5 by testing only a one-active-byte input; cancellations with two or three active inputs are exactly how a non-MDS matrix would fail. The criterion I need is stronger and clean: for a 4×4 linear map, branch number 5 is equivalent to every square submatrix being nonsingular. If an input supported on t byte positions could make more than t-1 selected output positions vanish, the corresponding t×t minor would be singular; conversely, a singular minor gives a t-active input with too many zero outputs. For this matrix the 1×1, 2×2, 3×3, and 4×4 minors are all nonzero in GF(2^8), so the map is MDS and the branch number is exactly 5. A single nonzero input byte then forces four active output bytes, a two-byte input forces at least three output bytes, and so on. Branch number 5, minimal cost. The inverse matrix needs coefficients {09},{0b},{0d},{0e} — heavier, so decryption will be a bit slower than encryption. I'll accept that asymmetry; encryption speed and the forward MDS property are what I'm optimizing, and the inverse still exists and is MDS.

But this column-mix alone gives me a two-layer guarantee of 5 active S-boxes *only within a single column*. A clever attacker keeps all their activity inside one column forever: difference confined to column 0, mixed within column 0, S-boxed, mixed within column 0 again — the column map never talks to the other columns, so the attacker pays the 5-active-box toll once and then nothing forces it to grow. The bound doesn't compound. The column mix diffuses *within* a column but not *between* columns, and that's the wall.

What I need is something that, between applications of the column mix, scatters the four bytes of any column out into four *different* columns, so that the next column-mix is forced to engage bytes that came from all over. That's the second, complementary spreading step, and it's why I wanted the square geometry. Take the rows of the 4×4 array and cyclically shift them by different amounts: leave row 0 alone, rotate row 1 by one byte, row 2 by two, row 3 by three. Now look at any single column after this shift — its four bytes were pulled from four different columns of the pre-shift state, because each row contributed from a different horizontal offset. Call it the row shift.

The bound compounds because the column mix gives a two-round lower bound inside each active column: for every nonzero column entering the mix, the S-box layer before the mix and the S-box layer after the mix together contain at least B = 5 active bytes in that column pair. The row shift is the piece that lets that column-local statement become global. It is a diffusion-optimal byte transposition: the four byte positions of any one column are sent to four different columns, and the inverse has the same property. In the middle of a four-round trail, I can count active columns on the two sides of the inter-column part. Because the row shift is diffusion-optimal, this middle column-level map has column branch number 5: the number of active columns before it plus the number of active columns after it is at least 5. Each active column in that middle count costs at least 5 active byte S-boxes in the adjacent two-round column-mix count. Multiplying the column-bundle count by the per-column byte count gives the wide-trail result I need: any differential or linear trail over four rounds activates at least 25 = 5 × 5 S-boxes.

That 25 is enough to rule out the classical high-probability single trails. Worst differential factor per active box is 2^-6, so any four-round trail has probability at most (2^-6)^25 = 2^-150, far below the 2^-127 single-trail threshold for a 128-bit block. That is a trail bound, not a claim that no input-output difference can ever have a larger total probability after many tiny trails are summed, but it removes the high-probability characteristics that the classical attack needs to predict. Same for linear: (2^-3)^25 = 2^-75 in correlation, and the data needed scales like the inverse square, beyond a full 128-bit codebook for a single trail. I don't have to *find* the best trail and check it; the branch-number argument *bounds all trails at once* from the public matrix and shift offsets. That's the transparent, structural security the incumbent couldn't offer — anyone can recompute the 25 and the 2^-150.

So the diffusion layer is two steps: row shift (cyclic shifts 0,1,2,3 across the rows, scattering bytes between columns) followed by column mix (the MDS matrix within each column). Together with the byte substitution, that's the round's confusion-and-diffusion core.

Now the key, and a structural choice that makes the *whole trail argument legitimate*. I'll make the cipher key-alternating: the round transformation itself is fixed and key-independent — substitute, shift, mix — and the key enters only by XOR-ing a 128-bit round key into the state, once before the first round and once after each round. Why XOR, and why isolate the key like this? Because in the differential/linear analysis above I quietly assumed the key just transforms differences and masks deterministically (contributing factor 1) and otherwise doesn't interfere with the per-box counting. A simple XOR of a round key does exactly that: it cancels in any difference, and it only translates a linear mask. So by making the cipher key-alternating I've arranged that *the active-S-box bound holds independently of the key value* — the 25 doesn't depend on which key you use. If the key were tangled nonlinearly into the round, I couldn't make that clean statement. The structure I chose for accountability's sake is the same structure that makes the bound provable.

I need round keys: Nr+1 of them, each 128 bits, from a cipher key of 128/192/256 bits. The key schedule has to (a) produce enough material, (b) inject nonlinearity and asymmetry so that a too-regular or too-linear schedule does not hand over easy related-key or slide structure, and (c) stay cheap. Work in 32-bit words (4 bytes); the schedule generates a stream of words w[i], and round key r is words w[4r..4r+3]. The first Nk words are the cipher key itself. Then recurse: most words are just w[i] = w[i−Nk] ⊕ w[i−1] — pure XOR, cheap, propagates key material forward. But a purely XOR schedule is linear and periodic, which is exactly the kind of symmetry slide and related-key attacks look for, so once per group of Nk words I perturb it nonlinearly: w[i] = w[i−Nk] ⊕ SubWord(RotWord(w[i−1])) ⊕ Rcon[i/Nk]. RotWord cyclically rotates the four bytes of the word; SubWord runs each byte through the *same* S-box (free nonlinearity, reusing the box I already have); and Rcon is a round constant whose left byte is x^(j-1) in GF(2^8) — {01},{02},{04},…,{1b},{36},… by repeated xtime. The round constant is the anti-symmetry device: it differs every round, so no two rounds' key schedules look alike. For the 256-bit key I add an extra SubWord at a second offset in each group, because the longer key needs the nonlinearity injected more often to diffuse properly. The number of rounds rises with the key: 10 for 128-bit, 12 for 192, 14 for 256 — enough iterations to stack the four-round 25-active-box bound several times over with comfortable margin.

One detail in the round sequence. The last round drops the column mix — substitute, shift, add key, and stop, no final MDS step. Why omit it? If I put an invertible linear column mix after the last substitution layer, it would not create another nonlinear layer. Even if I wrote it before the last key XOR, linearity lets me commute it through that XOR by transforming the last round key, so the result is only a public output change plus a reparameterized final key. It buys no extra resistance to the attacks I'm counting, and it costs work. Dropping it also makes encryption and decryption line up into the same structural shape, which simplifies implementation. So the final round is deliberately the odd one out.

Let me assemble the encryption of one 128-bit block. Load the 16 input bytes into the 4×4 state. XOR in round key 0 (initial whitening — so the first substitution isn't operating on attacker-known plaintext directly). Then Nr−1 full rounds, each: substitute every byte; shift the rows; mix the columns; XOR the round key. Then the final round: substitute; shift; XOR the last round key — no column mix. Output the 16 bytes of the state. Decryption runs the inverse steps in reverse order with the same round keys: inverse-shift, inverse-substitute (the S-box is a bijection, so it has a lookup inverse), XOR key, inverse-column-mix (the {09,0b,0d,0e} matrix), and so on.

That's a 128-bit permutation. But I rarely want to encrypt exactly one block — I want to encrypt a message of arbitrary length, and I want it to be safe to do so. A block cipher is a *permutation* on 128-bit blocks; if I naively split a message into blocks and encrypt each independently, identical plaintext blocks yield identical ciphertext blocks, leaking structure. The clean fix is to use the block cipher as a keystream generator and turn it into a stream cipher: counter mode. Pick a nonce, and for block i form the counter input (nonce combined with i), encrypt *that* with the cipher to get a pseudorandom keystream block, and XOR it into plaintext block i. Decryption is identical — regenerate the same keystream from the same counters and XOR again — so only the *forward* cipher is ever needed, and the blocks are independent, so it's fully parallel and supports random access. The one rule that mustn't be broken: never reuse a (key, counter) pair, or two keystream blocks coincide and XOR-ing the two ciphertexts cancels the keystream. With a unique nonce per message the counter blocks are all distinct, and the structural resistance I built is the reason I am willing to use the block outputs as keystream.

Let me write it down, byte-arithmetic and all, with the state as a 4×4 array.

```python
# GF(2^8), modulus m(x) = x^8 + x^4 + x^3 + x + 1  (low byte 0x1B)
def xtime(a):
    # multiply by {02}: shift left, fold in 0x1B if the top bit overflowed
    return (((a << 1) ^ 0x1B) & 0xFF) if (a & 0x80) else (a << 1)

# The byte substitution S(x) = affine( x^{-1} in GF(2^8) ), 0 -> 0 before affine.
# Provably differential-uniformity-4 and max-correlation-2^-3 from the inverse;
# the affine map (circulant + constant 0x63) hides the algebra and kills fixed points.
s_box = (  # generated as affine(inverse(x)); equivalently this fixed table
    0x63,0x7C,0x77,0x7B,0xF2,0x6B,0x6F,0xC5,0x30,0x01,0x67,0x2B,0xFE,0xD7,0xAB,0x76,
    0xCA,0x82,0xC9,0x7D,0xFA,0x59,0x47,0xF0,0xAD,0xD4,0xA2,0xAF,0x9C,0xA4,0x72,0xC0,
    0xB7,0xFD,0x93,0x26,0x36,0x3F,0xF7,0xCC,0x34,0xA5,0xE5,0xF1,0x71,0xD8,0x31,0x15,
    0x04,0xC7,0x23,0xC3,0x18,0x96,0x05,0x9A,0x07,0x12,0x80,0xE2,0xEB,0x27,0xB2,0x75,
    0x09,0x83,0x2C,0x1A,0x1B,0x6E,0x5A,0xA0,0x52,0x3B,0xD6,0xB3,0x29,0xE3,0x2F,0x84,
    0x53,0xD1,0x00,0xED,0x20,0xFC,0xB1,0x5B,0x6A,0xCB,0xBE,0x39,0x4A,0x4C,0x58,0xCF,
    0xD0,0xEF,0xAA,0xFB,0x43,0x4D,0x33,0x85,0x45,0xF9,0x02,0x7F,0x50,0x3C,0x9F,0xA8,
    0x51,0xA3,0x40,0x8F,0x92,0x9D,0x38,0xF5,0xBC,0xB6,0xDA,0x21,0x10,0xFF,0xF3,0xD2,
    0xCD,0x0C,0x13,0xEC,0x5F,0x97,0x44,0x17,0xC4,0xA7,0x7E,0x3D,0x64,0x5D,0x19,0x73,
    0x60,0x81,0x4F,0xDC,0x22,0x2A,0x90,0x88,0x46,0xEE,0xB8,0x14,0xDE,0x5E,0x0B,0xDB,
    0xE0,0x32,0x3A,0x0A,0x49,0x06,0x24,0x5C,0xC2,0xD3,0xAC,0x62,0x91,0x95,0xE4,0x79,
    0xE7,0xC8,0x37,0x6D,0x8D,0xD5,0x4E,0xA9,0x6C,0x56,0xF4,0xEA,0x65,0x7A,0xAE,0x08,
    0xBA,0x78,0x25,0x2E,0x1C,0xA6,0xB4,0xC6,0xE8,0xDD,0x74,0x1F,0x4B,0xBD,0x8B,0x8A,
    0x70,0x3E,0xB5,0x66,0x48,0x03,0xF6,0x0E,0x61,0x35,0x57,0xB9,0x86,0xC1,0x1D,0x9E,
    0xE1,0xF8,0x98,0x11,0x69,0xD9,0x8E,0x94,0x9B,0x1E,0x87,0xE9,0xCE,0x55,0x28,0xDF,
    0x8C,0xA1,0x89,0x0D,0xBF,0xE6,0x42,0x68,0x41,0x99,0x2D,0x0F,0xB0,0x54,0xBB,0x16,
)
inv_s_box = tuple(s_box.index(i) for i in range(256))  # the S-box is a bijection

def sub_bytes(s):                 # confusion: provably good per-active-box factor
    for i in range(4):
        for j in range(4):
            s[i][j] = s_box[s[i][j]]

def shift_rows(s):                # scatter each column's bytes into 4 columns
    s[0][1], s[1][1], s[2][1], s[3][1] = s[1][1], s[2][1], s[3][1], s[0][1]
    s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
    s[0][3], s[1][3], s[2][3], s[3][3] = s[3][3], s[0][3], s[1][3], s[2][3]

def mix_single_column(a):         # the MDS matrix [02 03 01 01] (branch number 5)
    t = a[0] ^ a[1] ^ a[2] ^ a[3]
    u = a[0]
    a[0] ^= t ^ xtime(a[0] ^ a[1])
    a[1] ^= t ^ xtime(a[1] ^ a[2])
    a[2] ^= t ^ xtime(a[2] ^ a[3])
    a[3] ^= t ^ xtime(a[3] ^ u)

def mix_columns(s):
    for i in range(4):
        mix_single_column(s[i])

def add_round_key(s, k):          # key-alternating: XOR only, so the bound is key-free
    for i in range(4):
        for j in range(4):
            s[i][j] ^= k[i][j]

# round constants: left byte is x^{j-1} in GF(2^8) -> breaks per-round symmetry
r_con = (0x00,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36)

def bytes2matrix(t): return [list(t[i:i+4]) for i in range(0, len(t), 4)]
def matrix2bytes(m): return bytes(sum(m, []))
def xor_bytes(a, b): return bytes(i ^ j for i, j in zip(a, b))

class AES:
    rounds_by_key_size = {16: 10, 24: 12, 32: 14}   # 128/192/256 -> 10/12/14
    def __init__(self, key):
        assert len(key) in AES.rounds_by_key_size
        self.n_rounds = AES.rounds_by_key_size[len(key)]
        self._rk = self._expand_key(key)

    def _expand_key(self, key):
        cols = bytes2matrix(key)
        nk = len(key) // 4
        i = 1
        while len(cols) < (self.n_rounds + 1) * 4:
            word = list(cols[-1])
            if len(cols) % nk == 0:                  # once per group: nonlinear + Rcon
                word.append(word.pop(0))             # RotWord
                word = [s_box[b] for b in word]      # SubWord (reuse the S-box)
                word[0] ^= r_con[i]                  # asymmetry across rounds
                i += 1
            elif len(key) == 32 and len(cols) % nk == 4:
                word = [s_box[b] for b in word]      # extra SubWord for the 256-bit key
            word = xor_bytes(word, cols[-nk])        # w[i] = w[i-Nk] XOR temp
            cols.append(word)
        return [cols[4*i:4*(i+1)] for i in range(len(cols) // 4)]

    def encrypt_block(self, pt):
        assert len(pt) == 16
        s = bytes2matrix(pt)
        add_round_key(s, self._rk[0])                # initial whitening
        for r in range(1, self.n_rounds):            # full rounds
            sub_bytes(s); shift_rows(s); mix_columns(s); add_round_key(s, self._rk[r])
        sub_bytes(s); shift_rows(s); add_round_key(s, self._rk[-1])  # final: no MixColumns
        return matrix2bytes(s)

def inc_bytes(a):                  # 128-bit counter increment
    out = list(a)
    for i in reversed(range(len(out))):
        if out[i] == 0xFF: out[i] = 0
        else: out[i] += 1; break
    return bytes(out)

def split_blocks(m, n=16):
    return [m[i:i+16] for i in range(0, len(m), n)]

def encrypt_ctr(aes, plaintext, nonce):
    # stream over arbitrary length: keystream_i = AES_K(counter_i), c_i = p_i XOR keystream_i
    assert len(nonce) == 16
    out, ctr = [], nonce
    for blk in split_blocks(plaintext):
        out.append(xor_bytes(blk, aes.encrypt_block(ctr)))  # only the forward cipher
        ctr = inc_bytes(ctr)                                 # unique counter per block
    return b''.join(out)
# decryption in CTR is identical: regenerate the same keystream and XOR again.

assert AES(bytes.fromhex("000102030405060708090a0b0c0d0e0f")).encrypt_block(
    bytes.fromhex("00112233445566778899aabbccddeeff")
) == bytes.fromhex("69c4e0d86a7b0430d8cdb78070b4c55a")
```

To recap the causal chain. The incumbent dies of a short key (fixed: 128-bit key) and of unaccountable security (the design target: argue resistance from public structure). Differential and linear cryptanalysis both reduce to "product over active S-boxes of a per-box factor below 1," so I split the round into a confusion layer that minimizes each box's factor and a diffusion layer that maximizes the active count. The confusion layer is the GF(2^8) inverse — provably differential-uniformity-4 and max-correlation-2^-3 — wrapped in an affine map that preserves those provable properties while hiding the algebra and removing fixed points. The diffusion layer is an MDS column mix (branch number 5, the optimum, at minimal {01,02,03} cost) interlocked with a row shift that scatters each column across four columns, so the branch number compounds to a guaranteed 25 active S-boxes over four rounds — driving any four-round trail's probability to 2^-150, below the 2^-127 single-trail threshold, a bound anyone can recompute. A key-alternating XOR keeps that bound independent of the key; a key schedule with per-round constants and S-box nonlinearity removes the simple schedule symmetries I do not want; ten-to-fourteen rounds stack the margin. Finally, counter mode lifts the 128-bit permutation to a parallel, random-access stream cipher that uses only the forward direction, safe as long as each counter is used once.
