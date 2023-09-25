import abc
import typing

from scripts.common.schemas import (
    RepositoryInferredSchema,
    RepositoryTypeCollectionSchema,
    TypeCollectionCategory,
    ContextCategory,
)
from pandera import typing as pt
import pandas as pd

from experiments import inferred


class ModelPredictionPipeline(abc.ABC):
    """
    groundtruth should be from experiments.inferred.load_groundtruths
    they have form of RepositoryTypeCollectionSchema + context_category + nested
    """

    def __init__(self, groundtruth: pd.DataFrame) -> None:
        self.groundtruth = groundtruth
        self.groundtruth = self.precise_filter(self.groundtruth)

        trivial_mask = self.groundtruth[
            RepositoryTypeCollectionSchema.adjusted_anno
        ].isin(["None", "Any"])
        context_mask = self.groundtruth["context_category"].isin(self.context_kind())

        self.groundtruth = self.groundtruth[~trivial_mask & context_mask]

        if not self.supports_nested():
            self.groundtruth = self.groundtruth[
                ~self.groundtruth["nested"].astype(bool)
            ]

        super().__init__()

    @abc.abstractmethod
    def context_kind(self) -> list[ContextCategory]:
        """List of context types that the model is capable of predicting for"""
        ...

    def precise_filter(
        self, predictions: pt.DataFrame[RepositoryInferredSchema]
    ) -> pt.DataFrame[RepositoryInferredSchema]:
        return predictions

    @abc.abstractmethod
    def supports_nested(self) -> bool:
        ...

    def adjusted(
        self, predictions: pt.DataFrame[RepositoryInferredSchema]
    ) -> pd.DataFrame:
        inferred.error_if_duplicate_keys(predictions)
        print(f"Initial prediction size: {predictions.shape}")

        print("Deriving limited form")
        limited = inferred.typet5_limited_form(predictions)

        print("Deriving adjusted form from limited form")
        adjusted = inferred.typet5_adjusted_form(limited)
        aligned = inferred.join_truth_to_preds(
            truth=self.groundtruth,
            predictions=adjusted,
            comparable_anno=RepositoryTypeCollectionSchema.adjusted_anno,
        )
        print(f"Size after joining predictions to groundtruth: {aligned.shape}")

        evaluatable = inferred.evaluatable(aligned)
        assert evaluatable["gt_anno"].notna().all()
        assert evaluatable["anno"].notna().all()

        print(f"Reduced to evaluatable: {evaluatable.shape}")
        return evaluatable

    def base(
        self,
        predictions: pt.DataFrame[RepositoryInferredSchema],
    ) -> pd.DataFrame:
        inferred.error_if_duplicate_keys(predictions)

        print("Deriving limited form")
        limited = inferred.typet5_limited_form(predictions)

        print("Deriving adjusted form from limited form")
        adjusted = inferred.typet5_adjusted_form(limited)

        print("Deriving base form from adjusted form")

        base = inferred.typet5_base_form(predictions)
        aligned = inferred.join_truth_to_preds(
            truth=self.groundtruth,
            predictions=base,
            comparable_anno=RepositoryTypeCollectionSchema.base_anno,
        )
        print(f"{aligned.shape}")

        evaluatable = inferred.evaluatable(aligned)
        assert evaluatable["gt_anno"].notna().all()
        assert evaluatable["anno"].notna().all()

        print(f"Reduced to evaluatable: {evaluatable.shape}")
        return evaluatable

class TypilusPipeline(ModelPredictionPipeline):
    def context_kind(self) -> list[ContextCategory]:
        return [
            ContextCategory.CALLABLE_RETURN,
            ContextCategory.CALLABLE_PARAMETER,
            ContextCategory.SINGLE_TARGET_ASSIGN,
            ContextCategory.ANN_ASSIGN,
            ContextCategory.INSTANCE_ATTRIBUTE,
        ]

    def supports_nested(self) -> bool:
        return False


class Type4PyPipeline(ModelPredictionPipeline):
    def context_kind(self) -> list[ContextCategory]:
        return [
            ContextCategory.CALLABLE_RETURN,
            ContextCategory.CALLABLE_PARAMETER,
            ContextCategory.SINGLE_TARGET_ASSIGN,
            ContextCategory.ANN_ASSIGN,
            ContextCategory.INSTANCE_ATTRIBUTE,
        ]

    def supports_nested(self) -> bool:
        return True


class TypeT5Pipeline(ModelPredictionPipeline):
    def context_kind(self) -> list[ContextCategory]:
        return [
            ContextCategory.CALLABLE_RETURN,
            ContextCategory.CALLABLE_PARAMETER,
            ContextCategory.SINGLE_TARGET_ASSIGN,
            ContextCategory.ANN_ASSIGN,
            ContextCategory.INSTANCE_ATTRIBUTE,
        ]

    def supports_nested(self) -> bool:
        return False

    def precise_filter(self, groundtruth: pd.DataFrame) -> pd.DataFrame:
        # Additional filter: only include top-level variables, i.e. variables that have at most one dot in them and are not in functions
        variable_mask = (
            groundtruth[RepositoryInferredSchema.category]
            == TypeCollectionCategory.VARIABLE
        )
        at_most_single_dotted = (
            groundtruth[RepositoryInferredSchema.qname].str.count(r"\.") <= 1
        )

        function_names = groundtruth[
            groundtruth[RepositoryInferredSchema.category]
            == TypeCollectionCategory.CALLABLE_RETURN
        ].qname
        symbols_in_functions = groundtruth.qname.str.rsplit(".", n=1, expand=True)[
            0
        ].isin(function_names)

        predicted_by_typet5 = groundtruth[
            ~variable_mask
            | (variable_mask & at_most_single_dotted & ~symbols_in_functions)
        ]
        return predicted_by_typet5
    

class HiTyperPipeline(ModelPredictionPipeline):
    def context_kind(self) -> list[ContextCategory]:
        return [
            ContextCategory.CALLABLE_RETURN,
            ContextCategory.CALLABLE_PARAMETER,
            ContextCategory.SINGLE_TARGET_ASSIGN,
            ContextCategory.ANN_ASSIGN,
            ContextCategory.INSTANCE_ATTRIBUTE,
        ]
    
    def supports_nested(self) -> bool:
        return False


def factory(
    tool: typing.Literal["type4pyN1", "typilusN1", "TypeT5TopN1", "HiTyper"],
    groundtruth: pd.DataFrame,
    inferred: pt.DataFrame[RepositoryInferredSchema],
    form: typing.Literal["adjusted", "base"],
) -> pd.DataFrame:
    match (tool, form):
        case "type4pyN1", "adjusted":
            return Type4PyPipeline(groundtruth).adjusted(inferred)
        case "typilusN1", "adjusted":
            return TypilusPipeline(groundtruth).adjusted(inferred)
        case "TypeT5TopN1", "adjusted":
            return TypeT5Pipeline(groundtruth).adjusted(inferred)
        case "HiTyper", "adjusted":
            return HiTyperPipeline(groundtruth).adjusted(inferred)
        case "type4pyN1", "base":
            return Type4PyPipeline(groundtruth).base(inferred)
        case "typilusN1", "base":
            return TypilusPipeline(groundtruth).base(inferred)
        case "TypeT5TopN1", "base":
            return TypeT5Pipeline(groundtruth).base(inferred)
        case "HiTyper", "base":
            return HiTyperPipeline(groundtruth).base(inferred)
