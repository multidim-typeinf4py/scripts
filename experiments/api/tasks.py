import glob
import pathlib

import polars as pl

from scripts.infer.structure import DatasetFolderStructure, AuthorRepo
from scripts.common.output import InferredIO, DatasetIO, ExtendedDatasetIO, ExtendedInferredIO
from scripts.common.schemas import TypeCollectionCategory


def inferreds(
    dataset: DatasetFolderStructure, tool: str, task: TypeCollectionCategory
) -> dict[AuthorRepo, pathlib.Path]:
    test_set = dataset.test_set()

    proj2datasets = [
        (
            project,
            InferredIO(
                artifact_root=pathlib.Path(),
                dataset=dataset,
                repository=project,
                tool_name=tool,
                task=task,
            ),
        )
        for project in test_set
    ]
    existing = [
        (dataset.author_repo(project), inferred_dataset.full_location())
        for project, inferred_dataset in proj2datasets
        if inferred_dataset.full_location().exists()
    ]
    return dict(existing)


def extended_inferreds(
    dataset: DatasetFolderStructure, tool: str, task: TypeCollectionCategory
) -> dict[AuthorRepo, pathlib.Path]:
    test_set = dataset.test_set()

    proj2datasets = [
        (
            project,
            ExtendedInferredIO(
                artifact_root=pathlib.Path(),
                dataset=dataset,
                repository=project,
                tool_name=tool,
                task=task,
            ),
        )
        for project in test_set
    ]
    existing = [
        (dataset.author_repo(project), inferred_dataset.full_location())
        for project, inferred_dataset in proj2datasets
        if inferred_dataset.full_location().exists()
    ]
    return dict(existing)


def ground_truths(dataset: DatasetFolderStructure) -> dict[AuthorRepo, pathlib.Path]:
    test_set = dataset.test_set()

    proj2datasets = [
        (
            project,
            DatasetIO(
                artifact_root=pathlib.Path(),
                dataset=dataset,
                repository=project,
            ),
        )
        for project in test_set
    ]
    existing = [
        (dataset.author_repo(project), gt_dataset.full_location())
        for project, gt_dataset in proj2datasets
        if gt_dataset.full_location().exists()
    ]
    return dict(existing)


def extended_ground_truths(dataset: DatasetFolderStructure) -> dict[AuthorRepo, pathlib.Path]:
    test_set = dataset.test_set()

    proj2datasets = [
        (
            project,
            ExtendedDatasetIO(
                artifact_root=pathlib.Path(),
                dataset=dataset,
                repository=project,
            ),
        )
        for project in test_set
    ]
    existing = [
        (dataset.author_repo(project), gt_dataset.full_location())
        for project, gt_dataset in proj2datasets
        if gt_dataset.full_location().exists()
    ]
    return dict(existing)


def dump_polars(
    df: pl.DataFrame, tool: str, task: TypeCollectionCategory, author_proj: str
) -> None:
    p = pathlib.Path(tool) / str(task) / f"{author_proj}.csv"
    assert not p.is_file(), f"Preventing overwrite of already existing dataset: {p}"
    p.parent.mkdir(parents=True, exist_ok=True)

    df.write_csv(file=p)
