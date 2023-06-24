from __future__ import annotations

import abc
import collections
import enum
import json
import logging
import pathlib
from abc import ABC
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import hityper
import hityper.__main__ as htm
import libcst
import pandas as pd
import pandera.typing as pt
import pydantic
from libcst import codemod, metadata, helpers
from libcst.codemod.visitors._apply_type_annotations import (
    Annotations,
    FunctionKey,
    FunctionAnnotation,
)
from typet5.experiments.hityper import HiTyperResponseParser
from typet5.static_analysis import SignatureMap

from ..annotators.hityper import HiTyperProjectApplier
from scripts import utils
from scripts.common.annotations import ApplyTypeAnnotationsVisitor
from scripts.common.schemas import InferredSchema, TypeCollectionCategory
from scripts.symbols.collector import build_type_collection
from ._base import ProjectWideInference


@dataclass
class _FindUserTypeArguments:
    repo: str
    core: int
    validate: bool
    output_directory: str

    groundtruth = None
    source = None


@dataclass
class _GenTDGArguments:
    repo: str
    optimize: bool
    output_directory: str
    output_format: str

    source = None
    location = None
    alias_analysis = True
    call_analysis = True


@dataclass
class _InferenceArguments:
    repo: str
    output_directory: str
    topn: int
    recommendations: str

    type4py = False
    source = None


class _HiTyperPredictionCategory(str, enum.Enum):
    ARG = "arg"
    LOCAL = "local"
    RET = "return"


class _HiTyperPrediction(pydantic.BaseModel):
    category: _HiTyperPredictionCategory
    name: str
    type: list[Optional[str]]

    class Config:
        use_enum_values = True


_HiTyperScope2Prediction = dict[str, list[_HiTyperPrediction]]


class _HiTyperPredictions(pydantic.BaseModel):
    __root__: dict[pathlib.Path, _HiTyperScope2Prediction]


class ModelAdaptor(abc.ABC):
    Prediction = tuple[str, float]
    VarPrediction = dict[str, list[Prediction]]

    class FuncPrediction(pydantic.BaseModel):
        q_name: str
        params_p: Optional[dict[str, list[ModelAdaptor.Prediction]]]
        ret_type_p: Optional[list[ModelAdaptor.Prediction]]
        variables_p: ModelAdaptor.VarPrediction

    class ClassPrediction(pydantic.BaseModel):
        q_name: str
        funcs: list[ModelAdaptor.FuncPrediction]
        variables_p: ModelAdaptor.VarPrediction

    class FilePredictions(pydantic.BaseModel):
        classes: list[ModelAdaptor.ClassPrediction]
        funcs: list[ModelAdaptor.FuncPrediction]
        variables_p: ModelAdaptor.VarPrediction

    class ProjectPredictions(pydantic.BaseModel):
        __root__: dict[str, ModelAdaptor.FilePredictions]

    @abc.abstractmethod
    def topn(self) -> int:
        ...

    @abc.abstractmethod
    def predict(
        self, project: pathlib.Path, subset: set[pathlib.Path]
    ) -> ModelAdaptor.ProjectPredictions:
        ...

    @abc.abstractmethod
    def preprocessor(self, task: TypeCollectionCategory) -> codemod.Codemod:
        ...


ModelAdaptor.FuncPrediction.update_forward_refs()
ModelAdaptor.ClassPrediction.update_forward_refs()
ModelAdaptor.FilePredictions.update_forward_refs()
ModelAdaptor.ProjectPredictions.update_forward_refs()


class HiTyper(ProjectWideInference, ABC):
    def __init__(self, adaptor: ModelAdaptor) -> None:
        super().__init__()
        self.adaptor = adaptor
        logging.getLogger(hityper.__name__).setLevel(logging.ERROR)

    def preprocessor(self, task: TypeCollectionCategory) -> codemod.Codemod:
        return self.adaptor.preprocessor(task)

    def _infer_project(
        self, mutable: pathlib.Path, subset: set[pathlib.Path]
    ) -> pt.DataFrame[InferredSchema]:
        output_dir = mutable / ".hityper"
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True, exist_ok=True)

        # Create model predictions and pass to HiTyper
        self.logger.info(
            f"Inferring over {len(subset)} files in {mutable} with {self.adaptor.__class__.__qualname__}"
        )

        # Provide ML predictions for ALL files
        # so that static analysis has best chance
        model_preds = self.adaptor.predict(
            mutable,
            subset=set(
                map(
                    lambda r: pathlib.Path(r).relative_to(mutable),
                    codemod.gather_files([str(mutable)]),
                )
            ),
        )

        outpath = output_dir / f"__{self.adaptor.__class__.__qualname__}__.json"
        assert not outpath.is_file()

        self.logger.info(f"Writing model predictions to {outpath} for HiTyper to use")
        with outpath.open("w") as f:
            json.dump(model_preds.dict(exclude_none=True)["__root__"], f, indent=2)

        self.logger.info(f"Registering {self.adaptor.__class__.__qualname__}'s artifacts...")
        self.register_artifact(model_preds)

        # input("Waiting for input...")

        htm.infertypes(
            _InferenceArguments(
                repo=str(mutable),
                output_directory=str(output_dir),
                topn=self.adaptor.topn(),
                recommendations=str(outpath),
            )
        )

        inferred_types_path = (
            str(output_dir)
            + "/"
            + str(mutable).replace("/", "_")
            + "_INFERREDTYPES.json"
        )

        with pathlib.Path(inferred_types_path).open() as ps:
            repo_predictions = json.load(ps)

        predictions = self._parse_predictions(repo_predictions, mutable)
        self.logger.info(f"Registering HiTyper's artifacts...")
        self.register_artifact(predictions)

        return HiTyperProjectApplier.collect_topn(
            project=mutable,
            subset=subset,
            predictions=predictions,
            topn=self.adaptor.topn(),
            tool=self,
        )

    def _parse_predictions(
        self, predictions: dict, project: pathlib.Path
    ) -> dict[pathlib.Path, list[SignatureMap]]:
        path2batchpreds = collections.defaultdict[pathlib.Path, list[SignatureMap]](list)
        for file, predictions in predictions.items():
            module = helpers.calculate_module_and_package(project, filename=file).name
            parser = HiTyperResponseParser(module=module)
            sigmap = parser.parse(predictions)

            path2batchpreds[pathlib.Path(file).relative_to(project)].append(sigmap)
        return path2batchpreds


def _derive_qname(scope: str) -> list[str]:
    if scope == "global@global":
        return []

    *funcname, classname = scope.replace(",", ".").split("@")
    if classname == "global":
        return funcname

    return [classname, *funcname]
