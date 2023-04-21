from __future__ import annotations

import dataclasses
import pathlib
from dataset import Dataset, DatasetConsumer

from common.schemas import TypeCollectionSchema
from symbols.collector import build_type_collection

import pandas as pd


@dataclasses.dataclass
class SplitData:
    train: pd.Series
    valid: pd.Series
    test: pd.Series

    @staticmethod
    def from_csv(path: pathlib.Path) -> SplitData:
        split_df = pd.read_csv(path, names=["split", "path"], header=None)
        return SplitData(
            train=split_df[split_df["split"] == "train"].drop(columns="split"),
            valid=split_df[split_df["split"] == "valid"].drop(columns="split"),
            test=split_df[split_df["split"] == "test"].drop(columns="split"),
        )


class ManyTypes4PyConsumerFull(DatasetConsumer):
    def __init__(self, split: pathlib.Path) -> None:
        super().__init__()
        self.split = split

    def produce(self, dataset: pathlib.Path) -> Dataset:
        collection = build_type_collection(dataset).df
        train, valid, test = SplitData.from_csv(self.split)

        return Dataset(
            train=collection[collection[TypeCollectionSchema.file].isin(train)],
            valid=collection[collection[TypeCollectionSchema.file].isin(valid)],
            test=collection[collection[TypeCollectionSchema.file].isin(test)],
        )


class ManyTypes4PyConsumerNeat(DatasetConsumer):
    def __init__(
        self, duplicates: pathlib.Path, type_checked_files: pathlib.Path, split: pathlib.Path
    ) -> None:
        super().__init__()
        self.duplicates = duplicates
        self.type_checked_files = type_checked_files
        self.split = split

    def produce(self, dataset: pathlib.Path) -> Dataset:
        collection = build_type_collection(dataset).df

        # Drop duplicate files
        forbidden: list[str] = []
        for path in self.duplicates.open():
            forbidden.append(path)
        collection = collection.drop(forbidden, columns=TypeCollectionSchema.file)

        # Retain type checked files
        permitted: list[str] = []
        for quoted_path in self.type_checked_files.open():
            assert quoted_path.startswith("'") and quoted_path.endswith("'")
            permitted.append(quoted_path[1:-1])

        collection = collection.drop(permitted, columns=TypeCollectionSchema.file)

        train, valid, test = _splits(self.split)
        return Dataset(
            train=collection[collection[TypeCollectionSchema.file].isin(train)],
            valid=collection[collection[TypeCollectionSchema.file].isin(valid)],
            test=collection[collection[TypeCollectionSchema.file].isin(test)],
        )
