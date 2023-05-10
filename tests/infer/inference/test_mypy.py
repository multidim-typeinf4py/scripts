import pathlib
import typing

import pytest

from common.schemas import TypeCollectionSchema
from infer.inference import MyPy
from ._utils import Project, example_project, example_project_subset, ProjectSubset


@pytest.fixture()
def mypy() -> MyPy:
    return MyPy(cache=None)


def test_run_mypyinfer(mypy: MyPy, example_project: Project):
    inferred = mypy.infer(mutable=example_project.mutable, readonly=example_project.readonly)
    assert not inferred.empty


def test_run_mypy_subset(mypy: MyPy, example_project_subset: ProjectSubset):
    inferred = mypy.infer(
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
