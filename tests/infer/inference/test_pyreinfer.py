import pathlib
import typing

import pytest

from infer.inference import PyreInfer
from ._utils import Project, example_project


@pytest.fixture()
def pyreinfer_ctor() -> typing.Callable[[Project], PyreInfer]:
    return lambda project: PyreInfer(
        mutable=project.mutable, readonly=project.readonly, cache=None
    )


def test_run_pyreinfer(
    pyreinfer_ctor: typing.Callable[[Project], PyreInfer], example_project: Project
):
    pyre_infer = pyreinfer_ctor(example_project)
    pyre_infer.infer()

    print(pyre_infer.inferred)
    assert not pyre_infer.inferred.empty
