The problem is to send, store, or compute with a fixed block of bits through a channel where any single bit may flip. Detection-only schemes such as a single parity bit or a 2-out-of-5 constant-weight code can tell that something went wrong, but they cannot say which bit changed, so the receiver has no choice but to halt or ask for a retransmission. Triple modular redundancy can correct a single error by majority vote, but it pays more than two hundred percent redundancy, which is far too expensive for large blocks. What is needed is exact single-error correction whose overhead grows slowly with the message length rather than multiplying the whole transmission by a constant factor.

The reason a single parity check fails to locate an error is that every position looks the same to it: the check only says yes or no, with no address. The way to get an address is to run several overlapping parity checks, each covering a different subset of positions. A flipped bit then fails exactly those checks whose subsets contain it, and the pattern of failures becomes a binary signature. The method that makes this signature precise is Hamming codes.

In a Hamming code, the positions of the codeword are numbered starting from one. Check bits are placed at the power-of-two positions 1, 2, 4, 8, and so on, and the message bits occupy the remaining positions. For each check bit i, compute the even parity of all positions whose i-th binary digit is one, excluding the check position itself. Because position 2^i belongs only to check i, each check bit can be set independently without disturbing any other check. When the word is received, recompute the same r parity checks and assemble the results as an r-bit binary number called the syndrome. If the syndrome is zero, no single error occurred. Otherwise the syndrome is literally the binary address of the corrupted position, so flipping that bit restores the codeword. After correction, the check positions are removed and the original message is recovered.

The number of check bits is chosen as the smallest r satisfying 2^r >= m + r + 1, where m is the number of message bits and n = m + r is the codeword length. This condition is necessary because the r-bit syndrome must distinguish all n possible error locations plus the no-error case. The smallest tight example has m = 4 and r = 3, giving the classic (7,4) code: check bits sit at positions 1, 2, and 4, while data bits sit at positions 3, 5, 6, and 7. If a single bit flips in position 6, the three parity checks return the binary pattern 110, which is 6, so the decoder flips that position back. This reliability holds for every message because the codewords form a linear subspace and the syndrome depends only on the error vector, not on the data.

Geometrically, each n-bit string is a vertex of the unit n-cube, and the Hamming distance between two strings is the number of coordinates in which they differ. Single-error correction requires that spheres of radius one around the codewords do not overlap, since an overlap would be a received word equally close to two codewords. Each such sphere contains n + 1 vertices, so 2^m <= 2^n / (n + 1), which rearranges to 2^r >= n + 1. Hamming codes meet this sphere-packing bound with equality, so the tight sizes (7,4), (15,11), (31,26), and so on are optimal: no code of the same length can carry more message bits while still correcting any single error.

Adding one extra overall parity bit over the entire codeword raises the minimum distance from three to four, yielding single-error correction with double-error detection. In that extended form, a nonzero syndrome together with an odd overall parity indicates a correctable single error, while a nonzero syndrome with an even overall parity indicates a double error that should be flagged rather than corrected. If the syndrome is zero but the overall parity is odd, only the extra parity bit itself is wrong, so the data are already correct.

```python
def _is_power_of_two(j):
    return j > 0 and j & (j - 1) == 0


def encode(data_bits, detect_double=False):
    m = len(data_bits)
    r = 0
    while (1 << r) < m + r + 1:
        r += 1
    n = m + r

    code = [0] * (n + 1)  # 1-indexed positions
    di = 0
    for j in range(1, n + 1):
        if not _is_power_of_two(j):
            code[j] = data_bits[di]
            di += 1

    for i in range(r):
        cpos = 1 << i
        parity = 0
        for j in range(1, n + 1):
            if j != cpos and (j >> i) & 1:
                parity ^= code[j]
        code[cpos] = parity

    codeword = code[1:]

    if detect_double:
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

    syndrome = 0
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
