# TIER: greedy
"""
The obvious first approach: touch every endpoint that needs no dependency
reasoning (this much any working fuzzer does -- try what's immediately
available), then try every OTHER endpoint exactly once too (in raw label
order, no graph reasoning about WHICH order would work), and finally spend
whatever budget is left spraying uniformly-random opcodes at whatever it
managed to create. It never builds a dependency graph and never computes
which opcode sequence a type's own transition table actually requires --
it just pokes blindly. This reliably finds the depth-1 ("any opcode
works") states, and only stumbles onto depth-3 chains when three
independent random opcode picks happen to land in the right order, on the
right resource -- a low-probability event that essentially never fully
clears a chain within budget.

Random state is seeded deterministically from a CRC32 of the raw input
bytes (never from wall-clock/OS entropy), so it is reproducible.
"""
import sys
import random
import zlib


def main():
    raw = sys.stdin.buffer.read()
    data = raw.split()
    it = iter(data)
    T = int(next(it)); C = int(next(it)); budget = int(next(it))
    deps = [None] * T
    for t in range(T):
        k = int(next(it))
        deps[t] = [int(next(it)) for _ in range(k)]
        S = int(next(it))
        for _ in range(S):
            for _ in range(C):
                next(it)

    seed = zlib.crc32(raw) & 0xffffffff
    rng = random.Random(seed)

    calls = []

    # phase 1: every type with NO dependency at all -- create + one op
    # (any opcode is fine to check, this phase never looks at the table).
    for t in range(T):
        if deps[t]:
            continue
        if len(calls) + 2 > budget:
            break
        line = len(calls)
        calls.append(f"C {t}")
        calls.append(f"O {line} 0")

    attempted = set(t for t in range(T) if not deps[t])
    phase2_lines = []

    # phase 2: try every remaining type once too, raw label order, no
    # dependency-graph reasoning about ordering.
    for t in range(T):
        if len(calls) >= budget:
            break
        if t in attempted:
            continue
        calls.append(f"C {t}")
        phase2_lines.append(len(calls) - 1)
        attempted.add(t)

    # phase 3: spray uniformly-random opcodes at whatever phase-2 resources
    # exist, for the rest of the budget.
    while len(calls) < budget and phase2_lines:
        r = rng.choice(phase2_lines)
        c = rng.randrange(C)
        calls.append(f"O {r} {c}")

    out = [str(len(calls))] + calls
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
