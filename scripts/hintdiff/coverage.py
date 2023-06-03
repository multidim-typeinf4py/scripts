import pathlib

from scripts.common import MergedAnnotations

import pandas as pd
import pandera as pa
import pandera.typing as pt

from .hintstat import Statistic, StatisticImpl


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

            annos = self._read_anno_for_repo(repo=repo, annotations=annotations)
            normed = annos.notna().value_counts(normalize=True)[True]
            coverage.append(float(normed))

        return pd.DataFrame({"repository": repository, "coverage": coverage}).pipe(
            pt.DataFrame[CoverageSchema]
        )
