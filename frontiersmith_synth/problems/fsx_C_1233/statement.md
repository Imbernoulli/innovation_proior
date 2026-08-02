# Cut Here, Not There: A Chunking Program for an Editable Archive

## Problem
You are designing the chunker for a block-based archive format: it stores a
byte corpus `A[0..N-1]` (values in `[0,K-1]`) as a sequence of **blocks**, a
boundary being a position where a new block starts. Instead of hand-picking
boundaries for this one corpus, submit a **fixed-schema straight-line
PROGRAM**, re-executed once per byte position — the same program must also
work, unmodified, on an **edited copy** of the corpus (one substring
spliced in). Score is charged for compressed size, the random-access index,
per-query seeks, *and* how much of the edited copy must be re-encoded from
scratch because your boundaries moved.

## Input (stdin)
```
N K
A_1 A_2 ... A_N          (0 <= A_i < K)
INS_POS INS_LEN
V_1 ... V_INS_LEN         (0 <= V_i < K, the spliced-in bytes)
WC WI WS WE
Q
q_1 q_2 ... q_Q           (query offsets, 0 <= q_i < N)
```
`1<=N<=3000`, `2<=K<=32`, `0<=INS_POS<=N`, `0<=INS_LEN<=25`, `0<=Q<=60`,
integer weights `1<=WC,WI,WS,WE<=10`.

## Output (stdout) — the artifact
A straight-line register program over `r0..r19`, re-executed once per byte
position `t` (fresh registers each call; memory persists across calls,
reset to 0 at the start of each of the two runs below). Each call: `r0..r7`
= memory `m0..m7` from the previous call (0 initially), `r8` = byte `A[t]`
of whichever sequence is being scanned, `r9` = `t`, `r10` = that sequence's
length, `r11` = `K`. Operands are a register `rN` or a bare, optionally
signed integer literal (e.g. `-7`):
```
ADD/SUB/MUL/DIV dst a b   MIN/MAX dst a b   LT dst a b  (1 if a<b else 0)
SEL dst c a b  (a if c!=0 else b)   MOV dst a
RESULT c m0 m1 m2 m3 m4 m5 m6 m7    (must be the single, final line)
```
`c!=0` cuts a new block boundary *before* position `t` (ignored at `t=0`:
block 0 always starts at `0`). `m0..m7` become next call's `r0..r7`. All
values clamp to `[-1e9,1e9]`; `DIV` by 0 yields `0`. At most 40 instruction
lines before `RESULT`; unknown opcode/register, bad arity, or a
missing/misplaced `RESULT` makes the artifact infeasible (score `0`).

## Scoring
The judge runs your program once over the base corpus to get boundary set
`Bnd`, splitting it into blocks, and once more (fresh memory) over the
**edited corpus** `A' = A[0:INS_POS] + V + A[INS_POS:]` to get blocks'.
Let `bits = max(1, ceil(log2 K))`.
- **Compressed size** `CS`: scan blocks left to right; a block whose exact
  byte content already occurred as an earlier block costs `2` (a dedup
  reference); otherwise it costs `6 + ceil(len*bits/8)`.
- **Index cost** `IC = 4 * (#blocks)`.
- **Seek cost** `SC = sum over queries q of (q - start(block containing q) + 1)`
  — reading `q` means decoding from its block's start, not the file start.
- **Edit cost** `EC = sum of len(b')` over every block `b'` of the edited
  layout whose exact content never occurs among the base blocks — bytes that
  must be re-encoded from scratch because the boundary moved.

`F = WC*CS + WI*IC + WS*SC + WE*EC` (**lower is better**). One block never
cut (`c` always `0`) is a legal baseline `B`. Score `= min(1, 0.1*B/F)`.

## Why one giant block is not the answer
Never cutting minimizes headers but forces every query and edit to touch
(or re-encode) almost the whole file. A fixed period keeps seeks and edits
local, but rarely lines up with where the corpus's repeated stretches
start, so on the edited copy every boundary after `INS_POS` lands on a new
relative offset and nothing dedups. Boundaries chosen from a small trailing
window of already-read bytes instead land at the *same relative content*
wherever it recurs — before or after `INS_POS` — so compression (via
dedup) and edit cost both improve together.

## Feasibility
Rejected (`Ratio: 0.0`) if the program is malformed per the grammar above.

## Example
`N=6 K=4`, corpus `1 2 1 2 1 2`, no edit (`INS_POS=6,INS_LEN=0`), weights
`1 1 1 1`, no queries. A program with `RESULT 0 0 0 0 0 0 0 0 0` (never cut) gives
one block of the whole corpus: `CS=6+ceil(6*2/8)=8`, `IC=4`, `SC=0`, `EC=0`
(edited corpus equals base since `INS_LEN=0`, so the single block matches)
→ `F=12=B` → `Ratio=0.1`.
