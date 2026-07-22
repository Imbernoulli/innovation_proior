# Automaton Combination Lock

A vault's combination is a length-`N` string over an alphabet `{0,...,A-1}`. The
factory only ever issues combinations that are **accepted by a hidden validation
circuit** modeled as a deterministic finite automaton `M`: states `0..S-1` are
"live", state `S` is an absorbing **REJECT** sink (never accepting, and the true
combination never enters it). From each live state only some symbols are
legal; an illegal symbol sends the circuit to REJECT forever. `M` (its full
transition table, alphabet size, accept states, start state) is **given to
you**; only the specific secret combination is hidden.

You must find the secret by **repeated guessing**. Your program is invoked
**once per guess round**: it reads one JSON object from stdin (the automaton
plus the history of every prior round) and must print one JSON guess to
stdout. Your process is restarted fresh every round — it has **no memory
between rounds** — so it must reconstruct everything it needs from the
`history` field each time.

## Feedback: it is about the CIRCUIT'S STATE, not raw symbol match

For your guess `G` (length `N`, symbols in `0..A-1`), the evaluator runs `M`
on `G` from the start state, producing a state trajectory `s_G[1..N]` (state
after each prefix). It does the same for the true combination `T`, producing
`s_T[1..N]`. The feedback is:

```
feedback[i] = 1  if s_G[i] == s_T[i]   (same automaton state after i symbols)
feedback[i] = 0  otherwise                                    (i = 1..N)
```

Because `M` can send two *different* symbols from the same state to the
*same* next state, `feedback[i] == 1` does **not** always mean `G[i] == T[i]`
— only that the two paths are indistinguishable to the circuit so far. You
also get a `correct` flag: `true` iff `G` equals `T` exactly (the win
condition). The round loop ends the instance the moment `correct` is `true`,
or after `max_guesses` rounds (a loss for that instance).

## Public instance JSON (given every round)

```json
{"N": 6, "A": 5, "S": 4, "reject_state": 4,
 "delta": [[...5 ints...], ...(S+1 rows total, row `reject_state` self-loops)],
 "accept": [1, 3], "start": 0, "max_guesses": 40,
 "history": [{"guess": [...], "feedback": [0,1,1,0,1,1], "correct": false}, ...]}
```

## Answer JSON (each round)

```json
{"guess": [g0, g1, ..., g_{N-1}]}     // each g_i in 0..A-1
```
A guess with wrong length, non-integer entries, or an out-of-range symbol
makes the **entire instance infeasible** (score 0) immediately.

## Scoring

Let `obj` = the round number on which you first got `correct: true` (or
"never" if you ran out of `max_guesses`, or sent a malformed guess — both
score `0` for that instance). The evaluator also computes, purely from the
automaton and the (hidden) target, `hi` = how many guesses a **feedback-blind
sweep** through all length-`N` automaton-accepted strings (ascending
lexicographic order) would need to reach this specific target — the cost of
guessing with no adaptation at all. Your score on that instance is

```
frac = clamp((hi - obj) / (hi - 1), 0, 1)
r    = 0.1 + 0.8 * frac
```

so matching the blind sweep exactly scores `0.1`; guessing right immediately
(`obj = 1`) scores `0.9`; failing scores `0`. The reported `Ratio` is the mean
of `r` over 10 deterministic, seeded instances — some with automata that never
merge states (feedback is fully reliable), others (including larger,
held-out ones) where merges are deliberately planted so that trusting a
single position's feedback in isolation can lock in a wrong symbol and derail
the rest of the game. There is no shortcut that works everywhere: a strategy
that only reacts to one coordinate of the feedback at a time, without
tracking which *strings* remain consistent with the automaton across the
whole history, will do fine on the clean instances and badly on the merging
ones.

Your program runs in an **isolated, OS-sandboxed subprocess each round** and
only ever sees the public payload above — never the target, its trajectory,
or `hi`.
