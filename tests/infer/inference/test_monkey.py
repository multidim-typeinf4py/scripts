import pytest

from infer.inference.monkey import MonkeyType
from ._utils import Project, example_project, better_example_project


@pytest.fixture()
def monkey() -> MonkeyType:
    return MonkeyType(cache=None)


def test_run_monkeytype(monkey: MonkeyType, better_example_project: Project) -> None:
    monkey.infer(mutable=better_example_project.mutable, readonly=better_example_project.readonly)
    print(monkey.inferred)
    assert not monkey.inferred.empty
