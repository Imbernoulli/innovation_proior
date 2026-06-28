#!/usr/bin/env python3
"""Independent differential tester for verify/sol.cpp.

The oracle here does not use AHU labels and does not import verify/brute.py.
For tiny forests it enumerates every bijection from nodes of forest 1 to nodes
of forest 2 and directly checks the definition: roots map to roots and parent
relations are preserved.
"""

from __future__ import annotations

import argparse
import itertools
import random
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    par1: tuple[int, ...]
    par2: tuple[int, ...]
    name: str


def oracle(par1: tuple[int, ...], par2: tuple[int, ...]) -> str:
    n = len(par1)
    if n != len(par2):
        return "NO"

    roots1 = {i + 1 for i, p in enumerate(par1) if p == 0}
    roots2 = {i + 1 for i, p in enumerate(par2) if p == 0}
    if len(roots1) != len(roots2):
        return "NO"

    nodes = tuple(range(1, n + 1))
    parent1 = {i + 1: par1[i] for i in range(n)}
    parent2 = {i + 1: par2[i] for i in range(n)}

    for perm in itertools.permutations(nodes):
        mapping = dict(zip(nodes, perm))
        if {mapping[r] for r in roots1} != roots2:
            continue
        ok = True
        for u in nodes:
            p = parent1[u]
            mapped_parent = 0 if p == 0 else mapping[p]
            if parent2[mapping[u]] != mapped_parent:
                ok = False
                break
        if ok:
            return "YES"
    return "NO"


def run_solution(exe: str, par1: tuple[int, ...], par2: tuple[int, ...]) -> str:
    inp = " ".join(
        [str(len(par1)), *(str(x) for x in par1), str(len(par2)), *(str(x) for x in par2)]
    )
    proc = subprocess.run(
        [exe],
        input=inp + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"solution exited {proc.returncode}: {proc.stderr}")
    return proc.stdout.strip()


def random_forest(n: int, rng: random.Random) -> tuple[int, ...]:
    par = []
    for i in range(1, n + 1):
        if i == 1:
            par.append(0)
        else:
            par.append(rng.randrange(0, i))
    return tuple(par)


def relabel(par: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
    n = len(par)
    old_to_new = list(range(n + 1))
    labels = list(range(1, n + 1))
    rng.shuffle(labels)
    for old, new in enumerate(labels, start=1):
        old_to_new[old] = new

    new_par = [0] * n
    for old in range(1, n + 1):
        new = old_to_new[old]
        p = par[old - 1]
        new_par[new - 1] = 0 if p == 0 else old_to_new[p]
    return tuple(new_par)


def mutate_parent(par: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
    n = len(par)
    if n <= 1:
        return par
    out = list(par)
    i = rng.randrange(2, n + 1)
    choices = [p for p in range(0, i) if p != out[i - 1]]
    out[i - 1] = rng.choice(choices)
    return tuple(out)


def adversarial_cases() -> list[Case]:
    return [
        Case((), (), "empty-vs-empty"),
        Case((), (0,), "empty-vs-single"),
        Case((0,), (0,), "single-vs-single"),
        Case((0, 0), (0, 1), "two-roots-vs-chain"),
        Case((0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5, 6, 7), "long-small-chain"),
        Case((0, 1, 1, 1, 1, 1, 1, 1), (0, 1, 1, 1, 1, 1, 1, 1), "wide-star"),
        Case((0, 1, 1, 2), (0, 1, 1, 3), "sibling-reordered-shape"),
        Case((0, 1, 1, 2, 2, 3), (0, 1, 1, 2, 3, 3), "balanced-near-miss"),
        Case((0, 0, 1, 3, 2, 5), (0, 0, 2, 1, 4, 3), "multi-root-relabel"),
        Case((0, 0, 1, 3, 2, 5), (0, 0, 1, 3, 2, 2), "multi-root-near-miss"),
    ]


def make_cases(seed: int, random_count: int) -> list[Case]:
    rng = random.Random(seed)
    cases = adversarial_cases()
    for t in range(random_count):
        n = rng.randrange(0, 9)
        if rng.random() < 0.35:
            base = random_forest(n, rng)
            other = relabel(base, rng)
            name = f"random-iso-{t}"
        else:
            n2 = rng.randrange(0, 9)
            base = random_forest(n, rng)
            other = random_forest(n2, rng)
            if rng.random() < 0.25 and n == n2:
                other = mutate_parent(relabel(base, rng), rng)
            name = f"random-free-{t}"
        cases.append(Case(base, other, name))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("exe")
    parser.add_argument("--seed", type=int, default=7300628)
    parser.add_argument("--random-count", type=int, default=500)
    args = parser.parse_args()

    yes = no = 0
    for case in make_cases(args.seed, args.random_count):
        want = oracle(case.par1, case.par2)
        got = run_solution(args.exe, case.par1, case.par2)
        if got != want:
            print(f"MISMATCH {case.name}")
            print(f"par1={case.par1}")
            print(f"par2={case.par2}")
            print(f"want={want} got={got}")
            raise SystemExit(1)
        if want == "YES":
            yes += 1
        else:
            no += 1
    print(f"PASS cases={len(make_cases(args.seed, args.random_count))} yes={yes} no={no}")


if __name__ == "__main__":
    main()
