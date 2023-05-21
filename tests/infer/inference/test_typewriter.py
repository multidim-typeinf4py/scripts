import pathlib

import pytest

from infer.inference import TypeWriter
from ._utils import Project, example_project


@pytest.fixture()
def typewriter() -> TypeWriter:
    return TypeWriter(cache=None, topn=10)


def test_run_tw(typewriter: TypeWriter, example_project: Project):
    inferred = typewriter.infer(mutable=example_project.mutable, readonly=example_project.readonly)

    print(inferred)
    assert not inferred.empty
