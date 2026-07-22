# Typo-Proof Genome: Growing a Body Through a Noisy Tape

A "genome" is a bounded tape of single-byte instructions that a tiny
**developmental machine** reads left-to-right to grow a body plan: a strip of
`P = 16` tissue cells, each with a type in `[0, A-1]`. The machine keeps a
write head `wp` (starts at 0). Each byte's **top 3 bits** select an opcode,
the **bottom 5 bits** are its argument, `byte = ((op & 7) << 5) | (arg & 31)`:

| op | name | effect |
|---|---|---|
| 0 | NOP | nothing |
| 1 | MOVE(d) | `wp = clamp(wp + d, 0, P-1)`, `d = (arg mod 9) - 4` (range -4..4) |
| 2 | SET(t) | cast a **vote** `t = arg mod A` for cell `wp` |
| 3 | DIV | a daughter cell copies the *current live type* of cell `wp` into `wp+1` (if `arg` even) or `wp-1` (odd), casts a vote there, and the head **moves** to the daughter |
| 4 | CKPT(p) | resynchronise: `wp = arg mod P` (an absolute jump, not a vote) |
| 5,6,7 | — | unused encodings, treated as NOP |

SET and DIV never overwrite outright — they **cast a vote** for the cell they
touch. After the whole tape runs, each cell's final type is whichever type got
the **most votes** cast on it (ties broken by the most-recently-cast tied
type); a cell that never received a vote defaults to type 0.

## Mutation ("typos")

Your tape is executed under transcription errors: for a fixed, seeded family
of `trials` independent runs, each **byte** independently has probability
`mut_rate` of suffering exactly ONE random bit-flip before that run. A flip
in the top 3 bits changes the opcode to whichever op is one bit away in the
3-bit field (`MOVE`=`0b001` flips to `NOP`=`0b000` or `DIV`=`0b011`, never to
`SET`=`0b010`, two bits away); a flip in the bottom 5 bits corrupts the
argument. Fidelity for one run is the fraction of the 16 cells matching the
hidden target; your instance score is the **mean** over `trials` runs.
Mutation seeds are private, generated only after you submit — `name` below
is a positional label, not a seed.

## Candidate contract (isolated stdin → stdout program)

```
stdin:  {"name": str, "P": 16, "A": int, "target": [t_0..t_15],
         "L_max": int, "mut_rate": float, "trials": int}
stdout: {"tape": [b_0, b_1, ...]}   # 1 <= len <= L_max, each b_i an int in [0,255]
```

`target[i]` is the type your genome should express at cell `i`. Invalid
shape/length/range, a crash, a timeout, or non-JSON output scores `0.0`.

## Scoring (deterministic)

For each instance the evaluator computes, **itself**, `q_base` = the mean
fidelity (over the same seeded `trials`) of its own do-nothing reference (a
single NOP byte — every cell defaults to type 0). Then:

```
r = clamp( 0.1 + 0.9 * (q_cand - q_base) / max(1e-9, 1.0 - q_base), 0, 1 )
```

Matching the do-nothing reference scores ≈0.1; perfect fidelity on every one
of the `trials` runs (essentially unreachable once `mut_rate > 0`) scores
1.0 — real headroom above any genome you can build. **Ratio** is the mean of
`r` over 10 fixed instances that vary `A`, target structure, `mut_rate`
(0.10–0.22) and `L_max`; **Vector** lists the per-instance `r`.

## Why this is hard

One easy way to draw an exact target is a **relative chain**: one `CKPT(0)`,
then `(SET, MOVE(+1))` pairs walking across the strip — exact when unmutated,
but every `MOVE` is load-bearing: one corrupted `MOVE` permanently shifts the
head, so every write *for the rest of the whole tape* lands on the wrong
cell. `CKPT`, in contrast, always jumps to a position computed from its own
argument alone — a corrupted `CKPT`+`SET` vote only ever costs that one vote,
it never propagates. `DIV` sits in between: it copies the *immediately
preceding* live cell, so a same-type run costs one byte per extra cell
instead of two (usually shorter than the chain) — but one corrupted `DIV`
mid-run can throw off the rest of *that run* (bounded, not tape-wide).

You have a fixed budget `L_max`. More relative chain doesn't help — you need
a genome robust to noise, not just correct when unmutated. Which addressing
is fully immune to corruption, and which only bounds the damage? How many
times can you afford to repeat an instruction so one corrupted repeat still
loses the vote? A genome that spends length on a spaced, self-correcting
code — not the shortest description — is what survives the noisy tape.
