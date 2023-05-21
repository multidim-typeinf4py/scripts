import pathlib
import typing

import pytest

from common.schemas import TypeCollectionSchema
from infer.inference import PyreInfer
from ._utils import Project, example_project, example_project_subset, ProjectSubset


@pytest.fixture()
def pyreinfer() -> PyreInfer:
    return PyreInfer(cache=None)

def test_run_pyreinfer(pyreinfer: PyreInfer, example_project: Project):
    inferred = pyreinfer.infer(mutable=example_project.mutable, readonly=example_project.readonly)
    # print(inferred)
    assert not inferred.empty


def test_run_pyreinfer_subset(
    pyreinfer: PyreInfer, example_project_subset: ProjectSubset
):
    inferred = pyreinfer.infer(
        mutable=example_project_subset.mutable,
        readonly=example_project_subset.readonly,
        subset=example_project_subset.subset,
    )

    assert not inferred.empty
    assert (
        inferred[TypeCollectionSchema.file]
        .isin(set(map(str, example_project_subset.subset)))
        .all()
    )
