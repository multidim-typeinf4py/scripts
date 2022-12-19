from ._base import ConflictResolution, Metadata
from common.schemas import InferredSchema

import numpy as np
import pandera.typing as pt
import pandas as pd


class Argumentation(ConflictResolution):
    def forward(
        self,
        static: pt.DataFrame[InferredSchema],
        dynamic: pt.DataFrame[InferredSchema],
        probabilistic: pt.DataFrame[InferredSchema],
        metadata: Metadata,
    ) -> pt.DataFrame[InferredSchema]:
        # print(static, dynamic, probabilistic, metadata, sep="\n")

        # Basic voting procedure; If more than one approach predicts the same type,
        # then infer this type, and mark accordingly
        combined = pd.concat([static, dynamic, probabilistic], ignore_index=True)

        # Ignores NA
        anno_freqs = combined["anno"].value_counts(ascending=False)
        # print(anno_freqs)

        # More than one approach inferred just the one type
        # Others diverged individually
        if len(no_single_guesses := anno_freqs[anno_freqs > 1]) == 1:
            correct_inferrences = pd.merge(
                left=no_single_guesses,
                right=combined,
                how="inner",
                left_index=True,
                right_on="anno",
            )
            inferrer_naming = correct_inferrences["method"].unique()
            methodology = "+".join(inferrer_naming)

            return pt.DataFrame[InferredSchema](
                {
                    "method": [methodology],
                    "file": [metadata.file],
                    "category": [metadata.category],
                    "qname": [metadata.qname],
                    "anno": [correct_inferrences["anno"].iloc[0]],
                }
            )

        return pt.DataFrame[InferredSchema](
            {
                "method": ["dummy"],
                "file": [metadata.file],
                "category": [metadata.category],
                "qname": [metadata.qname],
                "anno": [ConflictResolution.UNRESOLVED],
            }
        )
