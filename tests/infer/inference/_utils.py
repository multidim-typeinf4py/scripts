import dataclasses
import pathlib
import shutil

import pytest


@dataclasses.dataclass
class Project:
    mutable: pathlib.Path
    readonly: pathlib.Path


@pytest.fixture()
def example_project(tmp_path) -> Project:
    dataset = pathlib.Path.cwd() / "tests" / "resources" / "proj1"
    shutil.copytree(dataset, tmp_path, dirs_exist_ok=True)
    return Project(mutable=tmp_path, readonly=dataset)
