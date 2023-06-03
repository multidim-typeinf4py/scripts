import pathlib
from typing import Optional

import pandera.typing as pt

from scripts.common.schemas import TypeCollectionSchema
from scripts.symbols.collector import build_type_collection


def hints2df(folder: pathlib.Path, subset: Optional[set[pathlib.Path]]) -> pt.DataFrame[TypeCollectionSchema]:
    collection = build_type_collection(folder, allow_stubs=False, subset=subset)
    return collection.df


def stubs2df(folder: pathlib.Path, subset: Optional[set[pathlib.Path]]) -> pt.DataFrame[TypeCollectionSchema]:
    # Convert subset to stub naming
    if subset is not None:
        subset = {p.with_suffix(".pyi") for p in subset}

    collection = build_type_collection(folder, allow_stubs=True, subset=subset)
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
