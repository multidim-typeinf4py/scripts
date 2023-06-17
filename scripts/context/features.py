from dataclasses import dataclass

@dataclass(frozen=True)
class RelevantFeatures:
    # is the annotatable in a loop (while / for)
    loop: bool

    # has the annotatable previously been assigned to
    reassigned: bool

    # is the annotatable within a nested context,
    # i.e. function in function, class in class
    nested: bool

    # is the annotatable involved in some form of flow control,
    # e.g. if / elif / else; try / except / else / finally
    branching: bool

    # does the annotation that is to be predicted,
    # require builtin, local or import analysis
    scope_analysis: bool

    # what kind of annotatable are we looking at (AnnAssign, AugAssign, etc.)
    annotatable_kind: bool