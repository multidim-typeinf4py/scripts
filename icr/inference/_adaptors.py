import pathlib

import pandera.typing as pt

from common.schemas import TypeCollectionSchema
from symbols.cli import _collect


def stubs2df(stub_folder: pathlib.Path) -> pt.DataFrame[TypeCollectionSchema]:
    result, collection = _collect(stub_folder)
    if result.failures or result.warnings:
        print(
            "WARNING: Failures and / or warnings occurred, the symbol collection may be incomplete!"
        )

    return collection.df
