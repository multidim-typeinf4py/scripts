import pathlib
import typing

import pytest

from infer.inference import Type4Py
from infer.inference._base import DatasetFolderStructure


@pytest.fixture(scope="module")
def type4py_ctor() -> typing.Callable[[pathlib.Path], Type4Py]:
    return lambda dataset: Type4Py(
        pathlib.Path.cwd() / "models" / "type4py", dataset=dataset, topn=10
    )


@pytest.fixture(scope="module")
def mt4py() -> pathlib.Path:
    return pathlib.Path.cwd() / "tests" / "infer" / "inference" / "dataset"


def test_run_type4py_on_mt4py(
    type4py_ctor: typing.Callable[[pathlib.Path], Type4Py], mt4py: pathlib.Path
):
    type4py = type4py_ctor(mt4py)
    #type4py.infer(DatasetFolderStructure.MANYTYPES4PY)
    print(type4py.inferred)
