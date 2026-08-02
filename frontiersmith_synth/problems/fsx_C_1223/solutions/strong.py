# TIER: strong
"""The insight: choose the synchronization set from what the parse STACK says
is still open, not from one fixed token. A statement position is naturally
terminated by `;` (or closed by `}` if it is a block); commit to those. A
parenthesized group is naturally terminated by `)` or separated by `,` -- it
has no `;` of its own, so including `;` there only invites the paren to
swallow whatever follows. A for-header genuinely uses `;` to separate its
three clauses, but also needs `)` to escape immediately when the very first
clause is broken. This table finds each construct's OWN terminator first,
so it neither swallows a sibling opening bracket nor a second nearby `?`."""
import sys


def main():
    sys.stdin.read()
    print("0 5")  # STMT   : ;  }   (statement end, or the enclosing block closes)
    print("1 3")  # PAREN  : ,  )   (its own separator / its own close)
    print("0 3")  # FORHDR : ;  )   (its own clause separator / its own close)


if __name__ == "__main__":
    main()
