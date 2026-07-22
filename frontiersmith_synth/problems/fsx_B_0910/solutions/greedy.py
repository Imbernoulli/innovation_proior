# TIER: greedy
# The "obvious" strategy: solve one position at a time, left to right. To test
# position `front`, hold already-locked positions at their locked values, fill
# everything after `front` with symbol 0, and sweep the trial symbol at `front`
# through 0..A-1 until the returned per-position STATE-match feedback says
# feedback[front] == 1 -- then assume that symbol is "confirmed" and lock it.
#
# This treats state-match feedback as if it meant "this raw symbol is correct",
# which only holds when the automaton never merges two different symbols onto
# the same next state. When it DOES merge (a planted trap), this can lock the
# WRONG symbol -- and because later positions' states depend on the whole
# prefix, everything after a bad lock silently stops matching, and the
# position-by-position sweep can burn its whole guess budget without ever
# recovering, let alone noticing the mistake.
import sys, json


def main():
    inst = json.load(sys.stdin)
    L, A = inst["N"], inst["A"]
    history = inst.get("history", [])

    locked = {}
    front = 0
    trial_idx = 0
    for h in history:
        g, fb = h["guess"], h["feedback"]
        a_tried = g[front]
        if fb[front] == 1:
            locked[front] = a_tried
            front += 1
            trial_idx = 0
        else:
            trial_idx += 1
            if trial_idx >= A:
                locked[front] = a_tried  # give up, best effort
                front += 1
                trial_idx = 0

    if front >= L:
        guess = [locked[p] for p in range(L)]
    else:
        guess = [locked.get(p, 0) for p in range(L)]
        guess[front] = trial_idx if trial_idx < A else 0
        for p in range(front + 1, L):
            guess[p] = 0

    print(json.dumps({"guess": guess}))


main()
