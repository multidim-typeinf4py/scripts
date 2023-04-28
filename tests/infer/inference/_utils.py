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


@dataclasses.dataclass
class ProjectSubset:
    mutable: pathlib.Path
    readonly: pathlib.Path
    subset: set[pathlib.Path]


@pytest.fixture()
def example_project_subset(tmp_path) -> ProjectSubset:
    dataset = pathlib.Path.cwd() / "tests" / "resources" / "proj1"
    shutil.copytree(dataset, tmp_path, dirs_exist_ok=True)
    return ProjectSubset(
        mutable=tmp_path, readonly=dataset, subset={pathlib.Path("x.py")}
    )
