from __future__ import annotations

import abc
import enum
import pathlib
import typing

import pandas as pd
from libcst import codemod


class DatasetFolderStructure(abc.ABC):
    def __init__(self, dataset_root: pathlib.Path):
        self.dataset_root = dataset_root

    def __new__(cls, *args, **kwargs) -> DatasetFolderStructure:
        path: pathlib.Path = args[0] if args else kwargs.pop("dataset_root")
        if path.name.lower() == "many-types-4-py-dataset":
            cls = ManyTypes4Py
        elif path.name.lower() == "better-types-4-py-dataset":
            cls = BetterTypes4Py
        else:
            cls = Project
        return object.__new__(cls)

    def __repr__(self) -> str:
        return f"{type(self).__qualname__} @ {self.dataset_root}"

    @abc.abstractmethod
    def project_iter(self) -> typing.Generator[pathlib.Path, None, None]:
        ...

    @abc.abstractmethod
    def author_repo(self, repo: pathlib.Path) -> dict:
        ...

    @abc.abstractmethod
    def test_set(self, dataset_root: pathlib.Path) -> dict[pathlib.Path, set[pathlib.Path]]:
        ...


class ManyTypes4Py(DatasetFolderStructure):
    def project_iter(self) -> typing.Generator[pathlib.Path, None, None]:
        repo_suffix = self.dataset_root / "repos"
        authors = (
            author
            for author in repo_suffix.iterdir()
            if author.is_dir() and not author.name.startswith(".")
        )
        for author in authors:
            repos = (repo for repo in author.iterdir() if repo.is_dir())
            yield from repos

    def author_repo(self, repo: pathlib.Path) -> dict:
        return {"author": repo.parent.name, "repo": repo.name}

    def test_set(self, dataset_root: pathlib.Path) -> dict[pathlib.Path, set[pathlib.Path]]:
        splits = pd.read_csv(
            dataset_root / "data" / "dataset_split.csv",
            header=None,
            names=["split", "filepath"],
        )
        test_split = splits[splits["split"] == "test"]["filepath"]

        test_set: dict[pathlib.Path, set[pathlib.Path]] = {}
        for key, g in (
            test_split.str.strip(to_strip='"')
            .str.removeprefix("repos/")
            .str.split(pat=r"\/", n=2, expand=True)
            .set_axis(["author", "project", "file"], axis=1)
            .groupby(by=["author", "project"])
        ):
            author, project = key
            test_set[dataset_root / "repos" / author / project] = set(
                map(pathlib.Path, g["file"].tolist())
            )
        return test_set


class BetterTypes4Py(DatasetFolderStructure):
    def project_iter(self) -> typing.Generator[pathlib.Path, None, None]:
        repo_suffix = self.dataset_root / "repos" / "test"
        repos = (repo for repo in repo_suffix.iterdir() if repo.is_dir())
        yield from repos

    def author_repo(self, repo: pathlib.Path) -> dict:
        return dict(
            zip(
                ("author", "repo"),
                repo.name.split("__"),
                strict=True,
            )
        )

    def test_set(self, dataset_root: pathlib.Path) -> dict[pathlib.Path, set[pathlib.Path]]:
        repo_suffix = dataset_root / "repos" / "test"

        mapping = dict[pathlib.Path, set[pathlib.Path]]()

        for repo in repo_suffix.iterdir():
            if repo.is_dir() and (fs := codemod.gather_files([str(repo_suffix / repo)])):
                mapping[repo_suffix / repo] = set(
                    map(
                        lambda p: pathlib.Path(p).relative_to(repo_suffix / repo),
                        fs,
                    )
                )

        return mapping


class Project(DatasetFolderStructure):
    def project_iter(self) -> typing.Generator[pathlib.Path, None, None]:
        yield self.dataset_root

    def author_repo(self, repo: pathlib.Path) -> dict:
        return {"author": repo.name, "repo": repo.name}

    def test_set(self, dataset_root: pathlib.Path) -> dict[pathlib.Path, set[pathlib.Path]]:
        subfiles = set(
            map(
                lambda p: pathlib.Path(p).relative_to(dataset_root),
                codemod.gather_files([str(dataset_root)]),
            )
        )
        return {dataset_root: subfiles}
