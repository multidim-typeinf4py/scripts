from dataclasses import dataclass

@dataclass
class RelevantFeatures:
    loop: bool
    reassigned: bool
    nested: bool
    builtin: bool
    branching: bool
#    scope: bool    
