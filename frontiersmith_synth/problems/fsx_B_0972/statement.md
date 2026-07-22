# Live Shorthand: Coining Radio-Brevity Codes While the Dictation Keeps Coming

A field dispatcher's log streams in live, **round by round**. Each round delivers a
batch of dictated **lines** (each line is a short sequence of spoken **terms** — plain
tokens). You are the stenographer: every line must be transcribed into the official
log the moment its round arrives.

To keep up, you may **coin shorthand**: a shorthand symbol permanently stands for one
fixed, ordered sequence of 2 to 24 terms (a phrase). Once coined, a shorthand's phrase can
**never be changed or withdrawn** — it is irrevocable — and across the **entire session**
you may coin **at most `cap`** shorthand symbols, ever (a hard lifetime cap, e.g. a
finite page in the codebook). You do not know the future: when you decide what to coin
during round `r`, you have only seen rounds `0..r` — never later rounds.

## Candidate program contract

Your solution is a **standalone program**, invoked **once per round** (multiple times
per test case — a fresh, isolated subprocess call each time, with no memory carried
over). Each call: read ONE JSON object from **stdin**, write ONE JSON object to
**stdout**.

```python
import sys, json
inst = json.load(sys.stdin)
# ... decide new shorthand + transcribe this round's lines ...
print(json.dumps({"new_macros": [...], "rewrites": [...]}))
```

### Public input for round `r` (stdin)

```json
{
  "round": 3, "total_rounds": 8, "cap": 6, "overhead": 4,
  "library": [ {"id": 0, "pattern": ["t7","t2","t9"]}, ... ],
  "history": [ [ ["term",...], ["term",...] ], ... ],
  "lines":   [ ["term","term","term",...], ... ]
}
```
- `library`: shorthand symbols already coined (in earlier rounds), each irrevocable.
- `history`: the raw dictated lines of every **earlier** round (`0..r-1`), for reference.
- `lines`: this round's **new** lines, which you must transcribe now.

### Answer (stdout)

```json
{
  "new_macros": [ {"pattern": ["term","term",...]}, ... ],
  "rewrites":   [ ["term", "$0", "term", ...], ... ]
}
```
- `new_macros`: 0 or more freshly coined shorthands this round (pattern length between
  2 and 24 terms, inclusive). `library size after this round <= cap`, checked across
  the whole session.
- `rewrites`: exactly one rewritten line per entry of `lines`, in order. Each token is
  either a raw term (copied verbatim) or a reference `"$k"` to shorthand `k` from the
  library **as it stands after this round's `new_macros` are added** (so a freshly
  coined shorthand may be used immediately, on the same round that coined it).
- A rewrite is valid only if **expanding** it — replacing every `"$k"` with shorthand
  `k`'s phrase and concatenating — reconstructs the original line **exactly**. Any
  invalid round (bad shape, cap exceeded, a rewrite that fails to reconstruct its
  line, a bad reference, a crash, a timeout, or non-JSON output) makes the **entire
  session score 0**.

## Scoring (deterministic)

Per round: `token cost` = total tokens written across all of this round's rewrites
(a raw term or a `"$k"` reference both cost 1), plus a one-time **entry fee** for every
shorthand coined this round: `fee = overhead + len(pattern)`. The session cost is the
sum over all rounds; **lower is better**.

For each of 10 fixed, seeded sessions, the grader also computes `baseline` = the cost
of never coining anything (pure identity transcription). Per-session score:
`r = min(1, 0.1 * baseline / cost)`. The reported **Ratio** is the mean over all 10
sessions; **Vector** lists the per-session scores. Never coining anything is always
valid and scores exactly `0.1`.

## What makes this hard

Some phrases spike hard within a single round and never reappear — coining one is a
safe, cheap win *for that round*, but eats a permanent slot. Other phrases barely
register early yet **recur, growing, across many later rounds** — a phrase like that is
worth far more over the whole session, but its early signal is weak. With only `cap`
slots for the whole session, spending them all on whatever looks locally frequent can
leave nothing for the phrase that turns out to matter most.

## Suggested strategies

1. **Identity** (baseline): never coin anything.
2. **Reactive**: each round, coin whichever phrase saves the most tokens *this batch*,
   while slots remain.
3. **Reserve then spend**: hold back in the earliest round(s); once a phrase has
   recurred across multiple distinct rounds (not just one big batch), spend a slot on
   it before spending the rest on local spikes.
4. **Trend-aware**: track a running estimate of each phrase's session-wide share and
   weight persistence over raw volume when deciding what to commit.
