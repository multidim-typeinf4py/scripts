from common.schemas import TypeCollectionSchema

import pathlib
import typing

import pytest

from infer.inference import PyreQuery
from ._utils import Project, ProjectSubset, example_project, example_project_subset


@pytest.fixture()
def pyrequery() -> PyreQuery:
    return PyreQuery(cache=None)


def test_run_pyrequery(pyrequery: PyreQuery, example_project: Project):
    inferred = pyrequery.infer(mutable=example_project.mutable, readonly=example_project.readonly)
    assert not inferred.empty


def test_run_pyrequery_subset(pyrequery: PyreQuery, example_project_subset: ProjectSubset):
    inferred = pyrequery.infer(
        mutable=example_project_subset.mutable,
        readonly=example_project_subset.readonly,
        subset=example_project_subset.subset,
    )

    assert not inferred.empty
    assert inferred[TypeCollectionSchema.file].isin(set(map(str, example_project_subset.subset))).all()
