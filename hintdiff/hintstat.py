import abc
import enum
import pathlib

from common.schemas import MergedAnnotationSchemaColumns, TypeCollectionSchema
from common.storage import MergedAnnotations

import pandas as pd
from pandas._libs import missing
import pandera as pa
import pandera.typing as pt


class Statistic(enum.Enum):
    COVERAGE = "coverage"

    def __str__(self) -> str:
        return str(self.name)


class StatisticImpl:
    @property
    @abc.abstractmethod
    def ident(self) -> Statistic:
        ...

    @abc.abstractmethod
    def forward(self, *, repos: list[pathlib.Path], annotations: MergedAnnotations) -> pd.DataFrame:
        ...


class CoverageSchema(pa.SchemaModel):
    repository: pt.Series[str] = pa.Field()
    coverage: pt.Series[float] = pa.Field(ge=0.0, le=1.0)


class Coverage(StatisticImpl):
    ident = Statistic.COVERAGE

    def forward(
        self, *, repos: list[pathlib.Path], annotations: MergedAnnotations
    ) -> pt.DataFrame[CoverageSchema]:

        repository = list()
        coverage = list()
        for repo in repos:
            repository.append(repo.name)

            normed = annotations.df[f"{repo.name}_anno"].notna().value_counts(normalize=True)[True]
            coverage.append(float(normed))

        return pd.DataFrame({"repository": repository, "coverage": coverage}).pipe(
            pt.DataFrame[CoverageSchema]
        )
