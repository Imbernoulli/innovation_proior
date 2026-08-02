# Handshake Shim: Emulating Unmappable Wire Messages Under Observable-State Fidelity

## Problem
A legacy client only ever speaks an OLD wire protocol. You are replacing the server with a
NEW engine that speaks a different protocol. Write a **shim (translation table)** that
converts every OLD message type into a sequence of NEW messages, so that a legacy client
cannot tell the difference from the outside.

**State model.** The OLD protocol's session state is a vector of `K_OLD` integers, all
*observable* to the client (arithmetic mod `P`, a large prime). Each OLD message type `t`
applies a fixed affine transform `obs' = A_t . obs + b_t (mod P)`.

The NEW engine's state is a longer vector of `K_NEW = K_OLD + K_EXTRA` integers: the first
`K_OLD` coordinates are *observable* (aligned 1:1 with the OLD protocol's fields), the extra
`K_EXTRA` are private scratch the client never sees. Each NEW message type `s` applies its
own affine transform `v' = A'_s . v + b'_s (mod P)` on the full vector. **By contract, the
scratch coordinates reset to 0 every time the shim begins translating one incoming OLD
message**; the observable coordinates always carry over from the session's current value.

*(Illustrative FORM only, not this instance's hidden structure: think of a message type as
"add 1 to counter A, then copy counter B into counter C" -- the actual transforms are
arbitrary affine maps given numerically in the input, not this shape.)*

Most OLD types have an exact NEW-message counterpart. A few do **not**: no single NEW
message reproduces their effect on the observable coordinates. Those must be emulated by a
short *sequence* of NEW messages. Since the client only observes the final projection after
each OLD message, the scratch coordinates may legally pass through values unrelated to the
OLD engine internally -- only the observable output at each checkpoint must match.

## Input (stdin)
```
K_OLD K_NEW M_OLD M_NEW P L_MAX CPX_NUM CPX_DEN
NEWTYPES
  (M_NEW blocks, each: K_NEW rows of K_NEW ints = A'_s, then one row of K_NEW ints = b'_s)
OLDTYPES
  (M_OLD blocks, each: K_OLD rows of K_OLD ints = A_t, then one row of K_OLD ints = b_t)
SESSIONS
  num_sessions
  (for each session: one row of K_OLD ints = initial observable state, then an int L,
   then a row of L OLD-type ids = the session's message sequence)
```
Old/new type 0 is always the identity (a handshake/no-op), and every session's message
sequence begins with old type 0.

## Output (stdout)
```
M_OLD
(M_OLD rows, one per OLD type t=0..M_OLD-1, each: an int Lt in [0,L_MAX] followed by Lt
 NEW-type ids -- the fixed sequence to emit whenever OLD type t occurs)
```

## Scoring
The checker replays every session **statefully**: it tracks the shim's own carried-forward
observable vector (from the emulation output at each step, scratch reset to 0 before each
OLD message), separately from the true OLD-engine trajectory, and counts a checkpoint as
correct only if the shim's output exactly equals the true OLD state there. A wrong
emulation early in a session corrupts every later checkpoint in that session, since the
shim keeps building on its own (possibly wrong) state.

Objective `F = matches - CPX*(extra NEW messages beyond one per call, summed over every
invocation)`, `CPX = CPX_NUM/CPX_DEN` (read from the input). Baseline `B` = the same `F` for
the trivial "emit nothing for every type" table (always exactly `num_sessions`, since every
session's forced opening handshake is correctly reproduced by doing nothing). Final score:
`Ratio = min(1000, 100*max(0,F)/B) / 1000`.

## Feasibility
Output must declare `M_OLD` matching the input, list exactly `M_OLD` rows, each `Lt` in
`[0,L_MAX]` with that many NEW-type ids in `[0,M_NEW-1]`, all integers, no trailing tokens.
Any violation scores `Ratio: 0.0`.

## Example (worked score, illustrative sizes only)
With `num_sessions=10`, session length 6: if every type maps to an exact single or double
NEW sequence, `F` is close to `60 - small_penalty` against `B = 10`, near the score cap. A
shim that refuses every unmappable type instead gets the leading handshake and any messages
before the first unmappable type right, then drifts wrong for the rest of that session.

## Constraints
`K_OLD=3, K_EXTRA=2, M_OLD=7, M_NEW=7, L_MAX=2`. Sessions per instance up to 16, length up to
12. `P = 1000000007`.
