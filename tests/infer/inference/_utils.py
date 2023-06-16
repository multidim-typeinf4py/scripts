import dataclasses
import pathlib
import shutil

import libcst
from libcst import codemod

from scripts.utils import format_parallel_exec_result
import pytest


proj_root = pathlib.Path(__file__).parent.parent.parent.parent


@dataclasses.dataclass
class Project:
    mutable: pathlib.Path
    readonly: pathlib.Path


@pytest.fixture()
def example_project(tmp_path) -> Project:
    dataset = proj_root / "tests" / "resources" / "proj1"
    shutil.copytree(dataset, tmp_path, dirs_exist_ok=True)
    return Project(mutable=tmp_path, readonly=dataset)


@dataclasses.dataclass
class ProjectSubset:
    mutable: pathlib.Path
    readonly: pathlib.Path
    subset: set[pathlib.Path]


@pytest.fixture()
def example_project_subset(tmp_path) -> ProjectSubset:
    dataset = proj_root / "tests" / "resources" / "proj1"
    shutil.copytree(dataset, tmp_path, dirs_exist_ok=True)
    return ProjectSubset(mutable=tmp_path, readonly=dataset, subset={pathlib.Path("x.py")})


@pytest.fixture()
def unannotated(tmp_path) -> Project:
    dataset = proj_root / "tests" / "resources" / "unannotated"
    shutil.copytree(dataset, tmp_path, dirs_exist_ok=True)
    return Project(mutable=tmp_path, readonly=dataset)


def preprocess_project(dataset: pathlib.Path, preprocessor: codemod.Codemod):
    res = codemod.parallel_exec_transform_with_prettyprint(
        transform=preprocessor,
        files=codemod.gather_files([dataset]),
        repo_root=str(dataset),
        jobs=1,
    )
    f = format_parallel_exec_result(
        action=f"Preprocessing with {type(preprocessor).__qualname__}", result=res
    )
    print(f)
