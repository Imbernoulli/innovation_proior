# AES (Rijndael)

## Problem

Replace an aging block cipher that has a brute-forceable 56-bit key and whose
security against modern attacks rests on secret S-box criteria. The replacement
must take 128/192/256-bit keys, run fast in software and hardware, and — crucially —
let anyone *argue* its resistance to differential and linear cryptanalysis directly
from the public structure.

## Key idea

Both differential and linear cryptanalysis reduce a multi-round attack to a product,
over the **active S-boxes** of the best trail, of a per-box factor strictly below 1.
So security is governed by two separable quantities: how small each active S-box's
worst factor can be made (a local, nonlinear-layer problem) and how many S-boxes
every trail is *forced* to activate (a global, linear-layer problem). The design
addresses both with public, checkable bounds.

- **SubBytes (confusion).** One byte S-box S(x) = A(x⁻¹), the multiplicative inverse
  in GF(2⁸) (with 0↦0) followed by an invertible GF(2) affine map. The inverse gives
  differential uniformity 4 (max difference probability 2⁻⁶) and maximum linear
  correlation 2⁻³ — provably near-optimal. The affine map (a bit-circulant plus the
  constant 0x63) is differential/linear-invariant, so it preserves those bounds while
  destroying the simple algebraic description and removing fixed points S(x)=x and
  opposite fixed points S(x)=x⊕0xff.
- **ShiftRows + MixColumns (diffusion, the wide trail).** On a 4×4 byte state,
  MixColumns multiplies each column by a fixed **MDS** matrix over GF(2⁸), giving
  the optimal **branch number 5**: any single active input byte forces all four output
  bytes active. ShiftRows cyclically shifts rows by 0,1,2,3 bytes, scattering each
  column's bytes into four different columns. Interlocking the two makes the branch
  number compound: **every differential or linear trail over four rounds activates
  ≥ 25 = 5×5 S-boxes**, so any four-round trail has probability ≤ (2⁻⁶)²⁵ = 2⁻¹⁵⁰,
  far under the 2⁻¹²⁷ single-trail threshold for a 128-bit block. This is a
  trail bound, not a claim about a summed differential hull, and it is recomputable
  from the matrix and the shift offsets.
- **AddRoundKey + key schedule.** Key-alternating: round keys enter only by XOR, which
  keeps the active-S-box bound independent of the key value. KeyExpansion derives the
  round keys with mostly XOR, perturbed once per group by RotWord + SubWord + a round
  constant Rcon[j] (left byte x^{j-1} in GF(2⁸)) to inject nonlinearity and per-round
  asymmetry, removing the simple schedule symmetries slide and related-key attacks
  would look for.

## Algorithm

State: 16 bytes as a 4×4 byte array; the code below stores it as four column
lists. GF(2⁸) modulus m(x) = x⁸+x⁴+x³+x+1 (0x11B); xtime(b) = b·{02}.

Cipher (Nr = 10/12/14 rounds for 128/192/256-bit keys):

```
state ← in
AddRoundKey(state, rk[0])                       # initial whitening
for r = 1 .. Nr-1:
    SubBytes; ShiftRows; MixColumns; AddRoundKey(state, rk[r])
SubBytes; ShiftRows; AddRoundKey(state, rk[Nr]) # final round: no MixColumns
return state
```

MixColumns matrix (per column, over GF(2⁸)):

```
[ 02 03 01 01 ]
[ 01 02 03 01 ]
[ 01 01 02 03 ]
[ 03 01 01 02 ]
```

S-box: b̃ = x⁻¹ (0↦0), then, with bit indices counted from the least significant
bit, b'_i = b̃_i ⊕ b̃_{(i+4)%8} ⊕ b̃_{(i+5)%8} ⊕ b̃_{(i+6)%8} ⊕ b̃_{(i+7)%8}
⊕ c_i with c = 0x63.

**CTR mode** turns the 128-bit permutation into a stream cipher:
keystreamᵢ = AES_K(counterᵢ), cᵢ = pᵢ ⊕ keystreamᵢ, counter incremented per block.
Uses only the forward cipher; parallel and random-access; each (key, counter) used once.

## Code

```python
# GF(2^8), modulus m(x) = x^8 + x^4 + x^3 + x + 1 (low byte 0x1B)
xtime = lambda a: (((a << 1) ^ 0x1B) & 0xFF) if (a & 0x80) else (a << 1)

s_box = (
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
inv_s_box = tuple(s_box.index(i) for i in range(256))
r_con = (0x00,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36)

def sub_bytes(s):
    for i in range(4):
        for j in range(4):
            s[i][j] = s_box[s[i][j]]

def inv_sub_bytes(s):
    for i in range(4):
        for j in range(4):
            s[i][j] = inv_s_box[s[i][j]]

def shift_rows(s):
    s[0][1], s[1][1], s[2][1], s[3][1] = s[1][1], s[2][1], s[3][1], s[0][1]
    s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
    s[0][3], s[1][3], s[2][3], s[3][3] = s[3][3], s[0][3], s[1][3], s[2][3]

def inv_shift_rows(s):
    s[0][1], s[1][1], s[2][1], s[3][1] = s[3][1], s[0][1], s[1][1], s[2][1]
    s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
    s[0][3], s[1][3], s[2][3], s[3][3] = s[1][3], s[2][3], s[3][3], s[0][3]

def mix_single_column(a):                 # the MDS matrix [02 03 01 01], branch number 5
    t = a[0] ^ a[1] ^ a[2] ^ a[3]
    u = a[0]
    a[0] ^= t ^ xtime(a[0] ^ a[1])
    a[1] ^= t ^ xtime(a[1] ^ a[2])
    a[2] ^= t ^ xtime(a[2] ^ a[3])
    a[3] ^= t ^ xtime(a[3] ^ u)

def mix_columns(s):
    for i in range(4):
        mix_single_column(s[i])

def inv_mix_columns(s):                   # inverse matrix [0e 0b 0d 09]
    for i in range(4):
        u = xtime(xtime(s[i][0] ^ s[i][2]))
        v = xtime(xtime(s[i][1] ^ s[i][3]))
        s[i][0] ^= u; s[i][1] ^= v; s[i][2] ^= u; s[i][3] ^= v
    mix_columns(s)

def add_round_key(s, k):
    for i in range(4):
        for j in range(4):
            s[i][j] ^= k[i][j]

def bytes2matrix(t): return [list(t[i:i+4]) for i in range(0, len(t), 4)]
def matrix2bytes(m): return bytes(sum(m, []))
def xor_bytes(a, b): return bytes(i ^ j for i, j in zip(a, b))

def inc_bytes(a):
    out = list(a)
    for i in reversed(range(len(out))):
        if out[i] == 0xFF: out[i] = 0
        else: out[i] += 1; break
    return bytes(out)

def split_blocks(m, n=16):
    return [m[i:i+16] for i in range(0, len(m), n)]


class AES:
    rounds_by_key_size = {16: 10, 24: 12, 32: 14}
    def __init__(self, master_key):
        assert len(master_key) in AES.rounds_by_key_size
        self.n_rounds = AES.rounds_by_key_size[len(master_key)]
        self._key_matrices = self._expand_key(master_key)

    def _expand_key(self, master_key):
        cols = bytes2matrix(master_key)
        nk = len(master_key) // 4
        i = 1
        while len(cols) < (self.n_rounds + 1) * 4:
            word = list(cols[-1])
            if len(cols) % nk == 0:
                word.append(word.pop(0))                 # RotWord
                word = [s_box[b] for b in word]          # SubWord
                word[0] ^= r_con[i]                      # round constant
                i += 1
            elif len(master_key) == 32 and len(cols) % nk == 4:
                word = [s_box[b] for b in word]          # extra SubWord (AES-256)
            word = xor_bytes(word, cols[-nk])
            cols.append(word)
        return [cols[4*i:4*(i+1)] for i in range(len(cols) // 4)]

    def encrypt_block(self, plaintext):
        assert len(plaintext) == 16
        s = bytes2matrix(plaintext)
        add_round_key(s, self._key_matrices[0])
        for r in range(1, self.n_rounds):
            sub_bytes(s); shift_rows(s); mix_columns(s); add_round_key(s, self._key_matrices[r])
        sub_bytes(s); shift_rows(s); add_round_key(s, self._key_matrices[-1])
        return matrix2bytes(s)

    def decrypt_block(self, ciphertext):
        assert len(ciphertext) == 16
        s = bytes2matrix(ciphertext)
        add_round_key(s, self._key_matrices[-1]); inv_shift_rows(s); inv_sub_bytes(s)
        for r in range(self.n_rounds - 1, 0, -1):
            add_round_key(s, self._key_matrices[r]); inv_mix_columns(s)
            inv_shift_rows(s); inv_sub_bytes(s)
        add_round_key(s, self._key_matrices[0])
        return matrix2bytes(s)

    def encrypt_ctr(self, plaintext, iv):
        # keystream_i = encrypt_block(counter_i); c_i = p_i XOR keystream_i
        assert len(iv) == 16
        out, nonce = [], iv
        for blk in split_blocks(plaintext):
            out.append(xor_bytes(blk, self.encrypt_block(nonce)))
            nonce = inc_bytes(nonce)
        return b''.join(out)

    decrypt_ctr = encrypt_ctr   # CTR is symmetric: regenerate keystream and XOR again


assert AES(bytes.fromhex("000102030405060708090a0b0c0d0e0f")).encrypt_block(
    bytes.fromhex("00112233445566778899aabbccddeeff")
) == bytes.fromhex("69c4e0d86a7b0430d8cdb78070b4c55a")
assert AES(bytes.fromhex("000102030405060708090a0b0c0d0e0f1011121314151617")).encrypt_block(
    bytes.fromhex("00112233445566778899aabbccddeeff")
) == bytes.fromhex("dda97ca4864cdfe06eaf70a0ec0d7191")
assert AES(bytes.fromhex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")).encrypt_block(
    bytes.fromhex("00112233445566778899aabbccddeeff")
) == bytes.fromhex("8ea2b7ca516745bfeafc49904b496089")
```
