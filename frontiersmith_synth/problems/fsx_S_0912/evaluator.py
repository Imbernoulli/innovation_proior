#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0912 -- "Typo-Proof Genome: Growing a Body Through a Noisy Tape"
(family: developmental-tape-error-correction; format B, quality-metric).

THEME.  A "genome" is a bounded tape of single-byte instructions that a tiny
developmental machine reads left-to-right to GROW a body plan: a linear strip of
P=16 tissue cells, each with a type in [0, A-1].  The machine has a write head
(position `wp`) and five opcodes (encoded in the TOP 3 bits of each byte; the
BOTTOM 5 bits are the argument):

    op 0  NOP          no effect
    op 1  MOVE(d)       wp = clamp(wp + d, 0, P-1)      d = (arg mod 9) - 4   (-4..4)
    op 2  SET(t)         differentiate the CURRENT cell: cast a vote t at wp   t = arg mod A
    op 3  DIV            divide: a daughter cell copies the CURRENT cell's live
                          type into wp+1 (arg even) or wp-1 (arg odd), casts a
                          vote there, and the head MOVES to the daughter
    op 4  CKPT(p)         resynchronise: wp = arg mod P   (an absolute jump, not a vote)
    op 5,6,7             unused encodings -> treated as NOP (never crash)

Every SET/DIV instruction casts a VOTE for the type of the cell it touches
(it does not overwrite outright).  After the whole tape has executed, each
cell's FINAL type is the type with the most votes cast on it during the run
(ties broken by whichever tied type voted most RECENTLY); a cell that never
received a vote defaults to type 0.  This vote-tally rule is what lets a
genome that repeats an instruction under independent noise self-correct: if
only one of three repeated votes for a cell is corrupted, the other two still
win the tally.

MUTATION (the "typos").  The submitted tape is executed under TRANSCRIPTION
ERRORS: for a fixed seeded family of `trials` independent runs, each BYTE of
the tape independently has probability `mut_rate` of suffering exactly ONE
random bit-flip before that run.  A flip in the top 3 bits can SWAP THE
OPCODE to any op that is one bit away in the 3-bit field (e.g. MOVE=0b001 can
become NOP=0b000, DIV=0b011 or op5=0b101 -- but NOT SET=0b010, which is two
bits away); a flip in the bottom 5 bits corrupts the argument (wrong delta /
wrong type / wrong checkpoint).  Fidelity for a run is the fraction of the P
cells whose final type matches the hidden target; the instance score is the
MEAN fidelity over all `trials` runs.

THE TRAP.  A short, natural way to draw an exact target is a dense relative
chain: one CKPT to zero the head, then repeated (SET, MOVE(+1)) pairs walking
left to right. It is exact when unmutated -- but every MOVE is load-bearing
and deltas accumulate: a single corrupted MOVE permanently shifts the head,
so every SET downstream of it lands on the wrong cell for the REST OF THE
WHOLE TAPE -- one typo derails all subsequent development. A CKPT, by
contrast, always jumps to an absolute position computed from ITS OWN arg
alone, independent of any prior state, so a corrupted CKPT+SET vote can only
ever cost that one vote -- it never propagates. DIV sits in between: it
copies the immediately-preceding LIVE cell, so a chain of DIVs compressing a
same-type run is cheap, but a single corrupted DIV mid-chain can throw off
the rest of THAT ONE RUN (bounded damage, not tape-wide). Genomes that (a)
re-anchor with CKPT for redundant votes (fully cascade-immune) and (b) spend
the freed-up length budget on DIV-compression plus multiple independent
CKPT-anchored votes per cell trade code minimality for a self-correcting,
mostly-checkpointed encoding -- and stay close to the target even when a
large fraction of the tape gets hit.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON public instance:
            {"name": str, "P": 16, "A": int, "target": [t_0..t_{P-1}],
             "L_max": int, "mut_rate": float, "trials": int}
          `name` is an opaque label (an instance INDEX, not a seed) and
          carries no information about the private mutation-trial RNG stream.
  stdout: ONE JSON object: {"tape": [b_0, b_1, ...]}
          1 <= len(tape) <= L_max, every b_i an int in [0, 255].
  Invalid shape/length/range, a crash, a timeout, or non-JSON -> score 0.0.

SCORING (deterministic; no wall-time).  Per instance the evaluator computes,
ITSELF, `q_base` = the mean fidelity (over the SAME seeded `trials`) of its
own do-nothing NULL reference genome (a single NOP byte -- every cell
defaults to type 0; deliberately the weakest honest construction). Then
    r = clamp( 0.1 + 0.9 * (q_cand - q_base) / max(1e-9, 1.0 - q_base), 0, 1 )
Matching the null reference scores ~0.1; perfect fidelity on every one of the
`trials` runs (essentially unreachable once mut_rate > 0) scores 1.0, so there
is always real headroom above any real genome.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance, whose `name`
field is a plain instance index -- it does NOT encode the internal generation
seed. The target body plan is public (candidates must read and exploit it),
but the null reference and every mutation-trial seed/bit-flip are private to
this parent process, generated from an internal-only salt strictly AFTER the
candidate has already returned its tape, so a submission cannot replay or
pre-compute the exact noise realizations it will be scored against.

CLI:  python3 evaluator.py <solution.py>
Prints:  Ratio: <mean r>   and   Vector: [r_1, ..., r_10]
"""
import sys, json
import isorun

P = 16  # fixed phenotype length (fits opcode args in 4 of the 5 low bits)


# ----------------------------- deterministic LCG ----------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- target generation -----------------------------
def _build_target(seed, A, structure):
    """Deterministic target body plan of length P, values in [0, A-1]."""
    ni = _rng(seed)
    target = []
    if structure == "runs":
        pos = 0
        prev = -1
        while pos < P:
            run_len = ni(2, 5)
            run_len = min(run_len, P - pos)
            t = ni(0, A - 1)
            tries = 0
            while t == prev and tries < 5:
                t = ni(0, A - 1); tries += 1
            target.extend([t] * run_len)
            prev = t
            pos += run_len
    elif structure == "mixed":
        pos = 0
        prev = -1
        while pos < P:
            if ni(0, 99) < 45:
                run_len = min(ni(2, 4), P - pos)
            else:
                run_len = 1
            t = ni(0, A - 1)
            tries = 0
            while t == prev and tries < 5:
                t = ni(0, A - 1); tries += 1
            target.extend([t] * run_len)
            prev = t
            pos += run_len
    else:  # "random": near-independent cells
        for _ in range(P):
            target.append(ni(0, A - 1))
    return target[:P]


# ----------------------------- ISA -----------------------------------------
def _encode(op, arg):
    return ((op & 7) << 5) | (arg & 31)


def _run_tape(tape_bytes, A):
    """Execute a (possibly mutated) byte tape; return final phenotype list len P."""
    live = [0] * P
    votes = [[] for _ in range(P)]
    wp = 0
    for b in tape_bytes:
        b &= 0xFF
        op = (b >> 5) & 7
        arg = b & 31
        if op == 1:                              # MOVE
            d = (arg % 9) - 4
            wp = 0 if wp + d < 0 else (P - 1 if wp + d > P - 1 else wp + d)
        elif op == 2:                             # SET
            t = arg % A
            live[wp] = t
            votes[wp].append(t)
        elif op == 3:                             # DIV
            d = 1 if (arg % 2 == 0) else -1
            nw = 0 if wp + d < 0 else (P - 1 if wp + d > P - 1 else wp + d)
            t = live[wp]
            live[nw] = t
            votes[nw].append(t)
            wp = nw
        elif op == 4:                             # CKPT
            wp = arg % P
        # op 0, 5, 6, 7 -> NOP
    ph = [0] * P
    for p in range(P):
        vs = votes[p]
        if not vs:
            continue
        counts = {}
        for i, v in enumerate(vs):
            c, _ = counts.get(v, (0, -1))
            counts[v] = (c + 1, i)
        best_v, best_c, best_last = None, -1, -1
        for v, (c, last) in counts.items():
            if c > best_c or (c == best_c and last > best_last):
                best_v, best_c, best_last = v, c, last
        ph[p] = best_v
    return ph


def _mutate(tape_bytes, mut_rate, seed):
    ni = _rng(seed)
    thresh = int(round(mut_rate * 1_000_000))
    out = []
    for b in tape_bytes:
        if ni(0, 999_999) < thresh:
            bit = ni(0, 7)
            b = b ^ (1 << bit)
        out.append(b & 0xFF)
    return out


def _mean_fidelity(tape_bytes, target, A, trials, base_seed):
    total = 0.0
    for t in range(trials):
        mt = _mutate(tape_bytes, MUT_RATE_CTX, base_seed * 1_000_003 + t * 97 + 13)
        ph = _run_tape(mt, A)
        total += sum(1 for p in range(P) if ph[p] == target[p]) / P
    return total / trials


MUT_RATE_CTX = 0.0  # set per-instance before calling _mean_fidelity


# ----------------------------- reference genomes -----------------------------
def _null_tape():
    """The do-nothing reference: a single NOP byte. Casts no votes, so every
    cell defaults to type 0. This is the evaluator's own normalization anchor
    (maps to r=0.1) -- deliberately the weakest honest construction."""
    return [_encode(0, 0)]


def _weak_chain_tape(target, A):
    """Dense relative-chain construction: CKPT(0), then (SET, MOVE(+1)) per
    cell (last MOVE omitted). No redundancy, no re-anchoring -> exactly
    reconstructs `target` when unmutated, but a single corrupted MOVE shifts
    every subsequent write. This is the natural "obvious" first genome an
    average coder writes -- short, exact, and cascade-fragile."""
    tape = [_encode(4, 0)]                       # CKPT(0)
    for i, t in enumerate(target):
        tape.append(_encode(2, t % A))            # SET(t)
        if i != len(target) - 1:
            tape.append(_encode(1, 5))            # MOVE(+1) since (5 mod 9)-4 = 1
    return tape


# ----------------------------- instance family -------------------------------
def _build_instances():
    # (seed, A, structure, mut_rate, L_max, trials)
    specs = [
        (101, 5, "random", 0.10, 42, 48),
        (102, 6, "runs",   0.12, 60, 48),
        (103, 4, "mixed",  0.14, 55, 48),
        (204, 6, "random", 0.13, 70, 48),
        (205, 5, "runs",   0.16, 90, 48),
        (306, 5, "mixed",  0.13, 46, 48),
        (307, 6, "runs",   0.18, 120, 48),
        (408, 4, "random", 0.17, 80, 48),
        # harder / held-out: tighter budgets and/or higher noise
        (509, 6, "mixed",  0.20, 44, 56),
        (610, 5, "random", 0.22, 60, 56),
    ]
    out = []
    for idx, (seed, A, structure, mut_rate, L_max, trials) in enumerate(specs):
        target = _build_target(seed, A, structure)
        # `name` is a plain positional label -- it must NOT let a candidate
        # recover `seed` (which seeds the PRIVATE mutation-trial RNG below).
        # The trial-RNG seed is also put through its own irreversible mixing
        # constant so it is not simply `seed` re-derivable from public target
        # bytes either -- it is oracle state that lives ONLY in this process.
        trial_seed = (seed * 2654435761 + idx * 40503 + 0x9E3779B9) & ((1 << 32) - 1)
        out.append({"name": f"instance_{idx:02d}", "A": A, "structure": structure,
                    "mut_rate": mut_rate, "L_max": L_max, "trials": trials,
                    "target": target, "trial_seed": trial_seed})
    return out


# ----------------------------- answer validation -----------------------------
def _valid_tape(answer, L_max):
    if not isinstance(answer, dict):
        return None
    tape = answer.get("tape")
    if not isinstance(tape, list):
        return None
    if len(tape) < 1 or len(tape) > L_max:
        return None
    out = []
    for x in tape:
        if isinstance(x, bool) or not isinstance(x, int):
            return None
        if x < 0 or x > 255:
            return None
        out.append(x)
    return out


# ----------------------------- scoring driver --------------------------------
def main():
    global MUT_RATE_CTX
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        A = inst["A"]
        target = inst["target"]
        L_max = inst["L_max"]
        trials = inst["trials"]
        mut_rate = inst["mut_rate"]
        trial_seed = inst["trial_seed"]        # PRIVATE: never sent to the candidate

        MUT_RATE_CTX = mut_rate
        # NOTE: q_base and q_cand MUST be evaluated against the SAME sequence of
        # mutation trials (same seed) so that an identical tape gets an IDENTICAL
        # score -- otherwise Monte-Carlo sampling noise between two independent
        # trial streams would make even a tape equal to the reference drift away
        # from the 0.1 anchor. q_base is the do-nothing NULL tape (deliberately
        # the weakest honest reference); it never depends on mutation content.
        null = _null_tape()
        q_base = _mean_fidelity(null, target, A, trials, trial_seed)

        public = {"name": inst["name"], "P": P, "A": A, "target": list(target),
                  "L_max": L_max, "mut_rate": mut_rate, "trials": trials}
        # timeout=5s per instance: a hung/adversarial candidate can cost at most
        # 10 * 5 = 50s worst-case across all instances, safely inside the
        # config.yaml "time: 60s" budget (our own solutions run in well under 1s).
        ans, st = isorun.run_candidate(cand, public, timeout=5)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            tape = _valid_tape(ans, L_max)
        except Exception:
            tape = None
        if tape is None:
            vec.append(0.0)
            continue
        q_cand = _mean_fidelity(tape, target, A, trials, trial_seed)

        denom = 1.0 - q_base
        if denom < 1e-9:
            denom = 1e-9
        r = 0.1 + 0.9 * (q_cand - q_base) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        r = 0.0 if r < 0.0 else (1.0 if r > 1.0 else r)
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
