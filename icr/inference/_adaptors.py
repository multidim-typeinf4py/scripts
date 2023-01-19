import pathlib

import pandera.typing as pt

from common.schemas import TypeCollectionSchema
from symbols.collector import build_type_collection


def hints2df(folder: pathlib.Path) -> pt.DataFrame[TypeCollectionSchema]:
    collection = build_type_collection(folder)
    return collection.df

def stubs2df(folder: pathlib.Path) -> pt.DataFrame[TypeCollectionSchema]:
    collection = build_type_collection(folder, allow_stubs=True)
    return collection.df