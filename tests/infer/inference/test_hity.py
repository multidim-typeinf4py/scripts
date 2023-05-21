import pathlib
import typing

import pytest

from common.schemas import TypeCollectionSchema
from infer.inference import HiTyper
from ._utils import Project, example_project, example_project_subset, ProjectSubset


@pytest.fixture()
def hityper() -> HiTyper:
    return HiTyper(cache=None, topn=5)


def test_run_hityper(hityper: HiTyper, example_project: Project):
    inferred = hityper.infer(mutable=example_project.mutable, readonly=example_project.readonly)
    print(inferred)
    assert not inferred.empty


def test_run_hityper_subset(
        hityper: HiTyper, example_project_subset: ProjectSubset
):
    inferred = hityper.infer(
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
