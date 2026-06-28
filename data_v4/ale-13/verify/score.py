#!/usr/bin/env python3
"""Deterministic local scorer for "String Reassembly" (shortest superstring).

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single number: the score (an integer). A higher score is better.

Scoring rule (see context.md "Evaluation settings"):
  * The instance is `n s` on the first line, then `n` fragments (one per line).
  * A SOLUTION is a single string T on its own line (the reassembled superstring).
  * FEASIBILITY: T is feasible iff EVERY input fragment appears in T as a
    contiguous substring AND T uses only the allowed alphabet symbols. If any
    fragment is not a substring of T (or the file is missing / empty / has a
    stray character outside the fragments' alphabet), the solution is INFEASIBLE
    and the score is 0. (The feasibility -> 0 floor.)
  * Among feasible answers, SHORTER T is better (shortest common superstring).
    Let `Lsol = len(T)` and let `Lbase` be the length of the deterministic
    trivial-concatenation reference superstring (all fragments concatenated in
    the order given, which is always a feasible superstring). The score is

        score = round(1_000_000 * Lbase / Lsol)      (feasible, Lsol > 0)
        score = 0                                     (infeasible)

    The trivial concatenation scores exactly 1_000_000; any shorter feasible
    superstring scores strictly more. A higher score is better.

The scorer is self-contained and deterministic: it recomputes Lbase itself and
checks every fragment with Python's exact substring test, so it never trusts the
solver's claimed length.
"""
import sys


def read_instance(path):
    with open(path) as f:
        lines = f.read().split("\n")
    # first non-empty line holds "n s"
    idx = 0
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    header = lines[idx].split()
    n = int(header[0])
    s = int(header[1]) if len(header) > 1 else 0
    frags = []
    idx += 1
    while idx < len(lines) and len(frags) < n:
        # fragments are taken verbatim (no surrounding whitespace expected); a
        # fragment line is used exactly as-is after stripping a trailing CR.
        line = lines[idx].rstrip("\r")
        frags.append(line)
        idx += 1
    return n, s, frags


def read_solution(path):
    """Return the submitted superstring T (first non-empty line), or None."""
    try:
        with open(path) as f:
            data = f.read()
    except OSError:
        return None
    # The solution is a single line. Take the first non-empty line, stripped of
    # surrounding whitespace / CR. (Heuristic solvers print exactly one token.)
    for raw in data.split("\n"):
        t = raw.strip("\r").strip()
        if t != "":
            return t
    return None


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    n, s, frags = read_instance(sys.argv[1])

    # Alphabet actually present in the fragments (used to reject stray symbols).
    alphabet = set()
    for f in frags:
        alphabet.update(f)

    T = read_solution(sys.argv[2])
    if T is None or T == "":
        print(0)  # INFEASIBLE -> floored to 0
        return

    # Reject any symbol not in the fragment alphabet (keeps the solver honest).
    for ch in T:
        if ch not in alphabet:
            print(0)
            return

    # Feasibility: every fragment must be a contiguous substring of T.
    for f in frags:
        if f == "":
            continue
        if f not in T:
            print(0)  # a fragment is missing -> infeasible -> 0
            return

    Lsol = len(T)
    if Lsol <= 0:
        print(0)
        return

    # Deterministic reference: trivial concatenation of all fragments in order.
    Lbase = sum(len(f) for f in frags)
    if Lbase <= 0:
        # Degenerate (all-empty) instance: any non-empty answer is full credit.
        print(1_000_000)
        return

    score = int(round(1_000_000.0 * Lbase / Lsol))
    print(score)


if __name__ == "__main__":
    main()
