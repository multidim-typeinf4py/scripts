import pathlib

import pandera.typing as pt

from common.schemas import TypeCollectionSchema
from symbols.collector import build_type_collection


def hints2df(folder: pathlib.Path) -> pt.DataFrame[TypeCollectionSchema]:
    collection = build_type_collection(folder)
    return collection.df


def stubs2df(folder: pathlib.Path) -> pt.DataFrame[TypeCollectionSchema]:
    collection = build_type_collection(folder, allow_stubs=True)
    df = collection.df

    # Remove all source files to avoid skewing results
    source = df[TypeCollectionSchema.file].str.endswith(".py")
    df = df[~source]

    # Rename stub files to match their corresponding filename
    stubs = df[TypeCollectionSchema.file].str.endswith(".pyi")
    df.loc[stubs, TypeCollectionSchema.file] = (
        df.loc[stubs, TypeCollectionSchema.file].str.removesuffix(".pyi") + ".py"
    )

    return df
