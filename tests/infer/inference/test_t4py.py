import pathlib
import shutil
import typing

import pytest

from infer.inference import Type4PyN1

from ._utils import example_project, Project


@pytest.fixture()
def type4py_t() -> typing.Callable[[Project], Type4PyN1]:
    return lambda project: Type4PyN1(
        pathlib.Path.cwd() / "models" / "type4py",
        mutable=project.mutable,
        readonly=project.readonly,
        cache=None,
    )


def test_run_type4py(type4py_t, example_project: Project):
    type4py: Type4PyN1 = type4py_t(example_project)
    type4py.infer()

    print(type4py.inferred)
    assert not type4py.inferred.empty
