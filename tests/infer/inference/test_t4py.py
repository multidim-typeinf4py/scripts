import pathlib

import pytest

from infer.inference import Type4PyN10
from ._utils import Project, example_project


@pytest.fixture()
def type4py() -> Type4PyN10:
    return Type4PyN10(
        cache=None,
        model_path=pathlib.Path.cwd() / "models" / "type4py",
    )


def test_run_type4py(type4py: Type4PyN10, example_project: Project):
    type4py.infer(mutable=example_project.mutable, readonly=example_project.readonly)

    print(type4py.inferred)
    assert not type4py.inferred.empty
