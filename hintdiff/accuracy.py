import pathlib

import pandas as pd
import pandera as pa
import pandera.typing as pt

from common.storage import MergedAnnotations
from . import hintstat


class AccuracySchema(pa.SchemaModel):
    repository: pt.Series[str] = pa.Field()
    # match: pt.Series[bool] = pa.Field()
    accuracy: pt.Series[float] = pa.Field(ge=0.0, le=1.0)


class Accuracy(hintstat.StatisticImpl):
    ident = "accuracy"

    def __init__(self, reference: pathlib.Path) -> None:
        self._reference = reference
        super().__init__()

    def forward(self, *, repos: list[pathlib.Path], annotations: MergedAnnotations) -> pd.DataFrame:
        reference_ser = self._read_anno_for_repo(self._reference, annotations)

        repository = list()
        accuracy = list()

        for repo in repos:
            repository.append(repo.name)

            annos = self._read_anno_for_repo(repo=repo, annotations=annotations)
            acc = (reference_ser.fillna("-") == annos.fillna("-")).sum() / len(annos)

            accuracy.append(float(acc))

        return pd.DataFrame({"repository": repository, "accuracy": accuracy}).pipe(
            pt.DataFrame[AccuracySchema]
        )
