I start from the older picture: a channel is a physical line with limited signaling speed, and a message is a sequence of selections among distinguishable alternatives. The logarithm is already forced on me because independent selections should add. If I make `n` selections from `s` alternatives, `n log s` is the right scale. That gives me a measure, but it does not yet tell me what to do when the channel is noisy.

My first tempting move is to count the symbols that arrive correctly. That breaks immediately. If a binary channel is so noisy that the output is independent of the input, half the received symbols still match by chance. Calling that half a transmission rate would give credit to a useless channel. So the thing to subtract is not the expected number of wrong symbols. It is the uncertainty that remains about what was sent after the receiver sees what arrived.

That suggests a rate of actual transmission: uncertainty in the input minus uncertainty in the input given the output. In modern notation this is `I(X;Y) = H(X) - H(X|Y)`. For a fixed way of using the channel, this is the information rate the output carries about the input. But the channel does not force me to use each input symbol with a fixed frequency. Some input distributions fit the channel better than others, so the channel's boundary should be the maximum of this quantity over input distributions:

`C = max_{p(x)} I(X;Y)`.

At this point I have a plausible definition, not a theorem. The hard question is whether `C` is merely a descriptive number or an operational limit. If noise always requires more and more redundancy as I demand smaller error, then the reliable rate might collapse toward zero. Repetition codes encourage that pessimism. They reduce error, but their rate is visibly poor.

The way out is to stop designing a codeword pattern by hand. I look at long blocks. For long blocks, random behavior becomes regular in aggregate. The likely input blocks form an exponentially sized set. The likely output blocks form another. For each likely output, only an exponentially smaller fan of input blocks could reasonably have caused it. The exponent of that fan is the residual ambiguity. The gap between the full input exponent and that ambiguity is the information carried by the channel.

Now I need to choose many messages and assign them to long input blocks. If I try to construct the assignment explicitly, I am stuck. I do not know a clean deterministic pattern that avoids all the likely confusion fans. So I average over assignments instead. I randomly associate messages with long channel-input sequences generated from a good input distribution and ask for the average error probability of this whole ensemble.

For a sent message, the received block will normally be statistically consistent with the sent codeword. The dangerous event is that some other message's codeword also looks consistent with the same received block. In the modern typical-set version, a wrong codeword is independent of the received output, so the chance that this independent pair is jointly typical is about `2^{-n I(X;Y)}`. There are about `2^{nR}` wrong codewords. A union bound gives a collision term of order

`2^{nR} 2^{-n I(X;Y)} = 2^{-n(I(X;Y)-R)}`.

If `R < I(X;Y)`, this tends to zero. Since the average over randomly selected codebooks has small error, at least one fixed codebook has small error. The randomness is only a proof device; after the averaging argument, the communicating system can use a deterministic codebook.

I still have to check that I am not proving only an average-message guarantee while hiding a few terrible messages. The standard repair is expurgation: discard the worst fraction of codewords. The rate loss is negligible for long blocks, and the remaining code has small maximal error. This turns the ensemble argument into the reliability statement I actually want.

Maximizing over the input distribution gives every rate below `C`. That proves the surprising half: a noisy channel can have a definite positive reliable rate, and making the error probability arbitrarily small does not force the rate to zero.

The other half prevents the argument from being just a clever packing trick. Suppose I claim to communicate above `C` with vanishing error. Then the receiver's decoded message determines the original message with vanishing uncertainty. In modern terms, `H(M|Y^n) = o(n)`, so `nR = H(M) <= I(M;Y^n) + o(n)`. But the message reaches the output only through the channel input, and each channel use can contribute at most `C` bits of mutual information. Thus `I(M;Y^n) <= I(X^n;Y^n) <= nC`, giving `R <= C` in the limit.

In Shannon's original equivocation language, the same wall appears as unavoidable leftover uncertainty above capacity. If the attempted rate exceeds capacity, the excess cannot be made reliable by encoding; it reappears as equivocation or error. So the definition of capacity earns its name only when both directions are together: below it, long random-like codes exist; above it, no code can create more information than the channel law allows.

The proof is therefore deliberately nonconstructive. It explains why good codes must exist, not how to build efficient ones. That limitation is real, and it becomes the agenda for later coding theory. But the conceptual boundary is already fixed: reliable communication is governed by a maximized mutual-information rate, with typical long-block behavior making achievability possible and information inequalities making higher rates impossible.
