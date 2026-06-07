# Hoare — self-account extracts (verbatim where possible)

Consolidated from the three downloaded self-account sources. These are the backbone of reasoning.md.

## Source A — CHM Oral History (Bowen interview, 2006), `hoare-oral-history-chm.pdf`/`.txt`

> "I think Quicksort is the only really interesting algorithm that I ever developed, and I had
> already developed that when I was studying at Moscow State University. I got a letter from the
> National Physical Laboratory, which at that time was just starting a project for the automatic
> translation of Russian into English. ... I looked up the Russian literature on the subject. I met
> several of the people in Moscow who were working on the machine translation."

> "In those days, the dictionary, in which you had to look up in order to translate from Russian to
> English, was stored on a long magnetic tape, and it took several minutes for the tape to be read
> from the beginning to the end. Well, this dictionary was stored in alphabetical order, starting
> with A and ending with Z in English, and therefore it paid to sort the words of the sentence into
> the same alphabetical order before consulting the dictionary so that you could look up all the
> words in the sentence on a single pass of the magnetic tape. You didn't have to rewind the tape in
> order to look up the next word, because it was already in the right alphabetic order."

> Bowen: "But there were existing sorting algorithms. Were they already using those?"
> Hoare: "Oh, I didn't know anything about what existed in those days."

> "I thought with my knowledge of Mercury Autocode, I'll be able to think up how I would conduct
> this preliminary sort. After a few moments, I thought of the obvious algorithm, which is now
> called bubble sort, and rejected that, because that was obviously rather slow, and thought of
> Quicksort as the second thing I thought of. It didn't occur to me that this was anything very
> difficult. It was all an interesting exercise in programming."

## Source B — Turing Award lecture "The Emperor's Old Clothes" (1980/CACM 1981), `emperor.pdf`/`.txt`

> "My first task was to implement for the new Elliot 803 computer, a library subroutine for a new
> fast method of internal sorting just invented by Shell. ... My boss and tutor, Pat Shackleton, was
> very pleased with my completed program. I then said timidly that I thought I had invented a sorting
> method that would usually run faster than SHELLSORT, without taking much extra store. He bet me
> sixpence that I had not. Although my method was very difficult to explain, he finally agreed that I
> had won my bet."

> "Around Easter 1961, a course on ALGOL 60 was offered in Brighton ... It was there that I first
> learned about recursive procedures and saw how to program the sorting method which I had earlier
> found such difficulty in explaining. It was there that I wrote the procedure, immodestly named
> QUICKSORT, on which my career as a computer scientist is founded. Due credit must be paid to the
> genius of the designers of ALGOL 60 who included recursion in their language and enabled me to
> describe my invention so elegantly to the world."

## Source C — casual-coder interview (2015), web only

> "The first idea I had was insertion sort, but I recognized that this [is] a bit slow (quadratic)."
> "My second idea was quicksort. It occurred to me just as quickly as the first idea. I wrote the
> program for the partition, but I couldn't write the program to account for the list of unsorted
> segments."
> "Later, I discovered recursion from my reading of the Report on the ALGOL language. This enabled me
> to publish the program in [the] algorithms section of the ACM."

(Note the small discrepancy across sources on the rejected first sort: CHM oral history says "bubble
sort", the casual-coder interview says "insertion sort", the Turing lecture frames it against
Shellsort, which he had just hand-coded. I treat the CHM oral history as authoritative for the
"rejected-first, Quicksort-second" beat and use the Shellsort context from the Turing lecture.)

## The recursion WALL, in Hoare's own words — the crux

Two independent self-accounts converge on the same wall: he could write the *partition* in Mercury
Autocode immediately, but could NOT write the bookkeeping for the *list of still-unsorted segments*
(the recursion), and gave up on programming it until ALGOL 60 gave him recursion. The 1962 primary
paper confirms this from the other side: it spends most of its prose describing an explicit "nest"
(a stack / pushdown store) to hold postponed segments, and a footnote admits the ALGOL recursive
version is "deceptively simple, since the use of recursion means that the administration of the nest
does not have to be explicitly described."
