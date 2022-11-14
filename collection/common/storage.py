from dataclasses import dataclass

import pandas as pd


class TypeCollection:
    @dataclass(frozen=True)
    class Schema:
        CANONICAL_NAME = "canon_name"

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
