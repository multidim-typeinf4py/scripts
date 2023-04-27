import pathlib
import typing

import pytest

from infer.inference import PyreQuery
from ._utils import Project, example_project


@pytest.fixture()
def pyrequery_ctor() -> typing.Callable[[Project], PyreQuery]:
    return lambda project: PyreQuery(mutable=project.mutable, readonly=project.readonly, cache=None)


def test_run_pyrequery(
    pyrequery_ctor: typing.Callable[[Project], PyreQuery], example_project: Project
):
    pyre_query = pyrequery_ctor(example_project)
    pyre_query.infer()

    print(pyre_query.inferred)
    assert not pyre_query.inferred.empty
