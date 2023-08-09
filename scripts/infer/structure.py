from __future__ import annotations

import abc
import dataclasses
import pathlib
import typing

import pandas as pd
from libcst import codemod


@dataclasses.dataclass(frozen=True)
class AuthorRepo:
    author: str
    repo: str

    def __str__(self) -> str:
        return f"{self.author}__{self.repo}"


class DatasetFolderStructure(abc.ABC):
    def __init__(self, dataset_root: pathlib.Path):
        self.dataset_root = dataset_root

    def __new__(cls, *args, **kwargs) -> DatasetFolderStructure:
        path: pathlib.Path = args[0] if args else kwargs.pop("dataset_root")
        if path.name.lower() == "many-types-4-py-dataset":
            cls = ManyTypes4Py
        elif path.name.lower() == "cdt4py":
            cls = CrossDomainTypes4Py
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
    def author_repo(self, repo: pathlib.Path) -> AuthorRepo:
        ...

    @abc.abstractmethod
    def test_set(self) -> dict[pathlib.Path, set[pathlib.Path]]:
        ...


class CrossDomainTypes4Py(DatasetFolderStructure):
    def project_iter(self) -> typing.Generator[pathlib.Path, None, None]:
        for suffix in ("flask", "numpy"):
            authors = (
                author
                for author in (self.dataset_root / suffix).iterdir()
                if author.is_dir() and not author.name.startswith(".")
            )

            for author in authors:
                repos = (repo for repo in author.iterdir() if repo.is_dir())
                yield from repos

    def author_repo(self, repo: pathlib.Path) -> AuthorRepo:
        return AuthorRepo(repo.parent.name, repo.name)

    def test_set(self) -> dict[pathlib.Path, set[pathlib.Path]]:
        ts = dict[pathlib.Path, set[pathlib.Path]]()

        # TODO: Parse deduplication file, select all files regardless of split
        # TODO: as we plan to use the entirety of CDT4Py for testing purposes
        for project in self.project_iter():
            subpyfiles = codemod.gather_files([str(project)])
            ts[project] = set(map(lambda p: pathlib.Path(p).relative_to(project), subpyfiles))

        return ts


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
        return AuthorRepo(repo.parent.name, repo.name)

    def test_set(self) -> dict[pathlib.Path, set[pathlib.Path]]:
        splits = pd.read_csv(
            self.dataset_root / "data" / "dataset_split.csv",
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
            test_set[self.dataset_root / "repos" / author / project] = set(
                map(pathlib.Path, g["file"].tolist())
            )
        return test_set


class BetterTypes4Py(DatasetFolderStructure):
    def project_iter(self) -> typing.Generator[pathlib.Path, None, None]:
        repo_suffix = self.dataset_root / "repos" / "test"
        repos = (repo for repo in repo_suffix.iterdir() if repo.is_dir())
        yield from repos

    def author_repo(self, repo: pathlib.Path) -> AuthorRepo:
        return AuthorRepo(*repo.name.split("__"))

    def test_set(self) -> dict[pathlib.Path, set[pathlib.Path]]:
        mapping = dict[pathlib.Path, set[pathlib.Path]]()

        repo_suffix = self.dataset_root / "repos" / "test"

        for repo in repo_suffix.iterdir():
            if repo.is_dir() and (fs := codemod.gather_files([str(repo)])):
                mapping[repo] = set(
                    map(
                        lambda p: pathlib.Path(p).relative_to(repo),
                        fs,
                    )
                )

        assert mapping, f"No repos found!"
        return mapping


class Project(DatasetFolderStructure):
    def project_iter(self) -> typing.Generator[pathlib.Path, None, None]:
        yield self.dataset_root

    def author_repo(self, repo: pathlib.Path) -> AuthorRepo:
        return AuthorRepo(author=repo.name, repo=repo.name)

    def test_set(self) -> dict[pathlib.Path, set[pathlib.Path]]:
        subfiles = set(
            map(
                lambda p: pathlib.Path(p).relative_to(self.dataset_root),
                codemod.gather_files([str(self.dataset_root)]),
            )
        )
        return {self.dataset_root: subfiles}
