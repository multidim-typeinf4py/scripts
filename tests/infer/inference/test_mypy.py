import pathlib
import typing

import pytest

from common.schemas import TypeCollectionSchema
from infer.inference import MyPy
from ._utils import Project, example_project, example_project_subset, ProjectSubset


@pytest.fixture()
def mypy() -> MyPy:
    return MyPy(cache=None)


def test_run_pyreinfer(mypy: MyPy, example_project: Project):
    mypy.infer(mutable=example_project.mutable, readonly=example_project.readonly)
    print(mypy.inferred)
    assert not mypy.inferred.empty


def test_run_pyreinfer_subset(mypy: MyPy, example_project_subset: ProjectSubset):
    mypy.infer(
        mutable=example_project_subset.mutable,
        readonly=example_project_subset.readonly,
        subset=example_project_subset.subset,
    )

    assert not mypy.inferred.empty
    assert (
        mypy.inferred[TypeCollectionSchema.file]
        .isin(set(map(str, example_project_subset.subset)))
        .all()
    )
