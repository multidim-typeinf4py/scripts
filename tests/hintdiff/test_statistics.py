import pathlib
import pandera.typing as pt

from pandas._libs import missing
import pandas as pd

import hintdiff
from common.storage import MergedAnnotations
from common.schemas import MergedAnnotationSchema, TypeCollectionCategory

import pytest


@pytest.fixture
def annotations() -> MergedAnnotations:
    xpy = str(pathlib.Path("x.py"))

    # proj1 has 100% coverage, proj2 has 1/4 coverage

    return MergedAnnotations(
        pt.DataFrame[MergedAnnotationSchema](
            [
                (xpy, TypeCollectionCategory.CALLABLE_RETURN, "f", "int", missing.NA),
                (xpy, TypeCollectionCategory.VARIABLE, "a", "int", "str"),
                (xpy, TypeCollectionCategory.CALLABLE_PARAMETER, "f.b", "bytes", missing.NA),
                (xpy, TypeCollectionCategory.CALLABLE_PARAMETER, "f.c", "bytestring", missing.NA),
            ],
            columns=["file", "category", "qname", "proj1_anno", "proj2_anno"],
        )
    )


@pytest.mark.parametrize(
    argnames=["projects", "coverages"],
    argvalues=[
        (["proj1"], [1.0]),
        (["proj2"], [0.25]),
        (["proj1", "proj2"], [1.0, 0.25]),
    ],
    ids=["proj1", "proj2", "both"],
)
def test_coverage(
    annotations: MergedAnnotations, projects: list[str], coverages: list[float]
) -> None:
    statistic = hintdiff.Coverage()

    expected = pt.DataFrame[hintdiff.CoverageSchema](
        {"repository": projects, "coverage": coverages}
    )

    repos = list(map(pathlib.Path, projects))
    actual = statistic.forward(repos=repos, annotations=annotations)

    diff = pd.concat([actual, expected], ignore_index=True).drop_duplicates(keep=False)
    print(diff)
    assert diff.empty
