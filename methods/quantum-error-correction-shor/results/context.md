## Quantum Memory Is Not Classical Memory

A quantum memory cell can store an unknown superposition, not just a classical bit value. The useful information is carried by amplitudes and relative phases, so a repair procedure cannot freely inspect the stored value. A measurement that distinguishes the alternatives carrying the information changes the state being protected.

## Decoherence Is The Immediate Failure Mode

Large quantum computations require coherence to survive while many alternatives evolve and later interfere. Coupling to an environment threatens that coherence by entangling memory cells with degrees of freedom outside the computer. Before an error-correcting construction, the practical question is whether stored quantum information can be protected for longer than the raw physical memory would allow.

## Classical Redundancy Suggests A Blocked Strategy

The classical instinct is to make several copies, compare them, and overwrite the minority value. That cannot be used directly for an unknown quantum state. Linearity rules out a universal machine that clones arbitrary quantum states, and measuring several would-be copies would expose the state rather than merely diagnose damage.

## Error Correction Must Ask A Different Question

The repair operation must not ask, "what state was stored?" It must ask, "what disturbance happened?" This requires a larger physical system whose extra degrees of freedom can carry a recoverable record of the error while the logical amplitudes remain inaccessible to the diagnostic measurement.

## Success Criteria Before The Construction

A satisfactory quantum memory code must encode one logical qubit into a subspace of several physical qubits, tolerate a specified local error model, extract only an error syndrome, and apply a recovery that works for every unknown superposition. It must also state its limits: how many local faults are allowed, what independence assumptions are made, and how much overhead the protection introduces.
