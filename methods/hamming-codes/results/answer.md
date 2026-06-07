# Hamming Codes

## Problem

Send (or store, or compute over) a block of bits through a channel where any single bit may
flip. Detection alone — a parity bit, a 2-out-of-5 code — only tells you *that* something went
wrong, forcing a halt or a retransmission. Hamming codes let the receiver **locate and repair**
a single bit error on its own, with redundancy that grows only like the logarithm of the message
length (far below the 200%+ of triple-modular-redundancy), and exactly, for every input.

## Key idea

Use **several overlapping parity checks** instead of one. A single global parity bit detects a
single error but carries no address — every position looks the same to it. But if each check
covers a different subset of positions, a single flipped bit fails a specific *pattern* of
checks, and that pattern is an address.

Make the address explicit: arrange things so the vector of failed checks, read as a binary
number — the **syndrome** — equals the **position** of the bad bit, with all-zero meaning "no
error." This forces the design:

- **Parity check i covers exactly the positions whose i-th binary bit is 1** (check 1 →
  positions 1,3,5,7,…; check 2 → 2,3,6,7,…; check 3 → 4,5,6,7,…). Then a single error at
  position e fails exactly the checks corresponding to the 1-bits of e, so the syndrome reads
  out e directly.
- **Check bits sit at the power-of-two positions** 1, 2, 4, 8, …. Position 2^i belongs to check
  i alone, so each check bit is set independently and encoding is trivial; the rest of the
  positions carry the message.
- **Size condition** `2^r ≥ m + r + 1` (with r check bits and n = m + r): the r-bit syndrome
  must name all n positions plus "no error." In standard `(n,k)` coding notation, this is the
  sphere-packing/Hamming bound `2^(n-k) ≥ n + 1`. Equality gives the tight,
  minimum-redundancy codes: (7,4), (15,11), (31,26), ….

For the (7,4) code, put parity bits `p1,p2,p4` in positions 1, 2, 4 and data bits
`d1,d2,d3,d4` in positions 3, 5, 6, 7. Even parity gives:

```text
p1 = d1 ⊕ d2 ⊕ d4
p2 = d1 ⊕ d3 ⊕ d4
p4 = d2 ⊕ d3 ⊕ d4
```

**SEC-DED.** Adding one overall parity bit over the whole codeword lifts the minimum distance
from 3 to 4: a single error is corrected, a double error is detected (the original syndrome is
nonzero, but the overall parity comes out even). The decision table:

| original syndrome | overall parity | meaning |
|---|---|---|
| 0 | even | no error |
| 0 | odd | error in the overall-parity bit |
| ≠ 0 | odd | single error at position = syndrome → correct |
| ≠ 0 | even | double error → detect, do not correct |

**Geometric view.** Treat each n-bit string as a vertex of the unit n-cube and define the
distance between two strings as the number of positions in which they differ (the Hamming
distance / L1 metric on the cube). A code is a subset of vertices; minimum distance d governs
power: d=2 detects single errors, d=3 corrects single errors, d=4 corrects single and detects
double, and d=2t+1 corrects t. Single-error correction means disjoint radius-1 spheres of 1+n
points each, giving the sphere-packing bound `2^m ≤ 2^n/(n+1)` — equivalently
`2^(n-m) ≥ n+1` — the same condition as the syndrome count, with equality meaning the spheres
tile the cube perfectly.

## Algorithm

Encode:
1. Choose the smallest r with `2^r ≥ m + r + 1`; set n = m + r.
2. Place message bits in the non-power-of-two positions of 1..n.
3. Set each check bit at position 2^i to the even parity of all positions whose i-th bit is 1.
4. (SEC-DED) append one overall even-parity bit.

Decode:
1. Compute syndrome bit i = parity of all positions whose i-th bit is 1.
2. Without the extra overall parity bit, syndrome = 0 means no single error; a nonzero syndrome
   names the bit to flip.
3. (SEC-DED) use the overall parity to separate no error, an error in the overall bit, a
   correctable single error in the original n positions, and a detected double error.
4. Strip the check positions to recover the message.

## Code

```python
def _is_power_of_two(j):
    return j > 0 and j & (j - 1) == 0


def encode(data_bits, detect_double=False):
    m = len(data_bits)

    # smallest r whose r-bit syndrome names all n positions plus "no error"
    r = 0
    while (1 << r) < m + r + 1:
        r += 1
    n = m + r

    code = [0] * (n + 1)            # 1-indexed; powers of two = check positions
    di = 0
    for j in range(1, n + 1):
        if not _is_power_of_two(j):
            code[j] = data_bits[di]
            di += 1

    for i in range(r):             # set each check bit independently
        cpos = 1 << i
        parity = 0
        for j in range(1, n + 1):
            if j != cpos and (j >> i) & 1:
                parity ^= code[j]
        code[cpos] = parity

    codeword = code[1:]

    if detect_double:              # overall parity -> min distance 4
        overall = 0
        for b in codeword:
            overall ^= b
        codeword = codeword + [overall]

    return codeword


def decode(codeword, detect_double=False):
    if detect_double:
        overall_received = codeword[-1]
        code_part = codeword[:-1]
    else:
        code_part = list(codeword)

    n = len(code_part)
    code = [0] + list(code_part)

    r = 0
    while (1 << r) < n + 1:
        r += 1

    syndrome = 0                   # = binary position of a single error
    for i in range(r):
        parity = 0
        for j in range(1, n + 1):
            if (j >> i) & 1:
                parity ^= code[j]
        if parity:
            syndrome |= (1 << i)

    status = 'ok'
    if detect_double:
        overall_calc = 0
        for j in range(1, n + 1):
            overall_calc ^= code[j]
        overall_fail = (overall_calc ^ overall_received) != 0
        if syndrome == 0:
            status = 'corrected_overall_parity' if overall_fail else 'ok'
        elif overall_fail and syndrome <= n:
            code[syndrome] ^= 1
            status = 'corrected'
        elif overall_fail:
            status = 'uncorrectable'
        else:
            status = 'double_error'
    else:
        if 0 < syndrome <= n:
            code[syndrome] ^= 1
            status = 'corrected'
        elif syndrome != 0:
            status = 'invalid_syndrome'

    data_bits = [code[j] for j in range(1, n + 1) if not _is_power_of_two(j)]
    return data_bits, status
```

For the canonical (7,4) case: check positions 1, 2, 4; message positions 3, 5, 6, 7; the
parity-check matrix has column j equal to the binary representation of j, so `H·r (mod 2)` is
the binary position of the error. The extended (8,4) branch applies the four-case SEC-DED table
above.
