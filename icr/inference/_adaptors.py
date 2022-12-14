import pathlib

import pandera.typing as pt

from common.schemas import TypeCollectionSchema
from symbols.cli import _collect


def hints2df(folder: pathlib.Path) -> pt.DataFrame[TypeCollectionSchema]:
    result, collection = _collect(folder)
    if result.failures or result.warnings:
        print(
            "WARNING: Failures and / or warnings occurred, the symbol collection may be incomplete!"
        )

    return collection.df

# Consider special treatment?
stubs2df = hints2df