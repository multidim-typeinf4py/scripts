from __future__ import annotations

from dataclasses import dataclass

@dataclass
class RelevantFeatures:
    loop: bool
    reassigned: bool
    nested: bool
    source: bool
    flow_control: bool


    @staticmethod
    def default() -> RelevantFeatures:
        return RelevantFeatures(
            loop=True,
            reassigned=True,
            nested=True,
            source=True,
            flow_control=True,
        )