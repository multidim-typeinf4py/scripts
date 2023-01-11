from dataclasses import dataclass

@dataclass
class RelevantFeatures:
    loop: bool
    reassigned: bool
    nested: bool
    user_defined: bool
#    scope: bool    
