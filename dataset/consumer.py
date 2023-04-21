import abc
import dataclasses
import pathlib

from common.schemas import TypeCollectionSchema
import pandera.typing as pt


@dataclasses.dataclass
class Dataset:
    train: pt.DataFrame[TypeCollectionSchema]
    valid: pt.DataFrame[TypeCollectionSchema]
    test: pt.DataFrame[TypeCollectionSchema]


class DatasetConsumer(abc.ABC):
    @abc.abstractmethod
    def produce(self, dataset: pathlib.Path) -> Dataset:
        ...
