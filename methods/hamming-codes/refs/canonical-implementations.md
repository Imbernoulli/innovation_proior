# Canonical Hamming-code implementations (grounding for the final code)

## Source 1 — Wikipedia Hamming(7,4) matrices
Generator (so that codeword = G^T p, mod 2) and parity-check matrix with columns
ordered so the syndrome equals the binary position of the error:

H =
[1 0 1 0 1 0 1]
[0 1 1 0 0 1 1]
[0 0 0 1 1 1 1]

Column j of H is the binary representation of j (j = 1..7). So H·r (mod 2),
read as a binary number, gives the position of a single bit error directly.
Encoding: codeword = G^T p (mod 2). Decoding: syndrome z = H·r (mod 2);
if z = 0 no error, else flip bit at position z.
URL: https://en.wikipedia.org/wiki/Hamming(7,4)

## Source 2 — joeladdison gist (Python (7,4) with overall-parity SECDED)
https://gist.github.com/joeladdison/5244877
Computes three syndrome bits over the position groups (odd positions, etc.),
combines them into a binary syndrome that names the error position, flips it,
and uses an overall parity bit to distinguish single vs double errors.
Key structure: parity bits at power-of-two positions; syndrome = binary error position.

```python
SYNDROME_CHECK = [-1, 6, 5, 0, 4, 1, 2, 3]
def hamming_encode_nibble(data):
    d = [extract_bit(data, i) for i in range(4)]
    h = [0,0,0]
    h[0] = (d[1] + d[2] + d[3]) % 2
    h[1] = (d[0] + d[2] + d[3]) % 2
    h[2] = (d[0] + d[1] + d[3]) % 2
    p = 0 ^ d[0] ^ d[1] ^ d[2] ^ d[3] ^ h[0] ^ h[1] ^ h[2]
    ...
```

## Source 3 — Hamming's own construction (Art of Doing Science, ch.12; BSTJ 1950 §3)
Parity check #1 -> positions 1,3,5,7,9,11,13,15,...  (bit-0 of position is 1)
Parity check #2 -> positions 2,3,6,7,10,11,14,15,... (bit-1 of position is 1)
Parity check #3 -> positions 4,5,6,7,12,13,14,15,... (bit-2 of position is 1)
Parity check #4 -> positions 8,9,...,15,24,...        (bit-3 of position is 1)
Check bits sit at positions 1,2,4,8,... (powers of two) so each parity check sets
exactly one check bit (its own), independent of the others.
The (7,4) code: check positions 1,2,4; message positions 3,5,6,7.
Syndrome read right-to-left = binary number = position of the single error;
all-zero = no error.

The general-position implementation used in the deliverables follows this directly:
for a codeword of length n, check bits live at every power-of-two position; parity
check i covers exactly the positions whose i-th bit is set; the syndrome (the vector
of failed checks, read as a binary number) is the position of the bad bit.
