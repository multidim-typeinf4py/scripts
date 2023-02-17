from common.schemas import TypeCollectionSchema, InferredSchema
import pandas as pd
import pandera.typing as pt

from common.storage import TypeCollection

import libcst.codemod as codemod
import libcst.codemod._cli as cstcli

from infer.insertion import TypeAnnotationApplierTransformer

import pytest


@pytest.mark.skip
def test_into_annotations():
    # annotations = TypeCollection.to_annotations(pyre_df)

    pyre_df = TypeCollection.load("tests/resources/proj1@(HiTyper)/.icr.csv")
    result = codemod.parallel_exec_transform_with_prettyprint(
        transform=TypeAnnotationApplierTransformer(context=codemod.CodemodContext(), tycol=pyre_df),
        files=cstcli.gather_files(["tests/resources/proj1@(HiTyper)"]),
        jobs=1,
        repo_root="tests/resources/proj1@(HiTyper)",
    )
