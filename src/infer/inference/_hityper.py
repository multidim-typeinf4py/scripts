from __future__ import annotations

import abc
import collections
import enum
import json
import logging
import pathlib
from abc import ABC
from dataclasses import dataclass
from typing import Optional

import hityper
import hityper.__main__ as htm
import libcst
import pandas as pd
import pandera.typing as pt
import pydantic
from libcst import codemod, metadata
from libcst.codemod.visitors._apply_type_annotations import (
    Annotations,
    FunctionKey,
    FunctionAnnotation,
)

import utils
from src.common.annotations import ApplyTypeAnnotationsVisitor
from src.common.schemas import InferredSchema
from src.symbols.collector import build_type_collection
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


class ParallelTypeApplier(codemod.ContextAwareTransformer):
    def __init__(
        self,
        context: codemod.CodemodContext,
        path2batches: dict[pathlib.Path, list[Annotations]],
        topn: int,
    ) -> None:
        super().__init__(context)

        self.paths2batches = path2batches
        self.topn = topn

    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        assert self.context.filename is not None
        assert self.context.metadata_manager is not None

        path = pathlib.Path(self.context.filename).relative_to(
            self.context.metadata_manager.root_path
        )

        annotations = self.paths2batches[path][self.topn]
        return metadata.MetadataWrapper(tree, unsafe_skip_copy=True).visit(
            ApplyTypeAnnotationsVisitor(self.context, annotations=annotations)
        )


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

    def __init__(self, model_path: pathlib.Path) -> None:
        self.model_path = model_path
        super().__init__()

    @abc.abstractmethod
    def topn(self) -> int:
        ...

    @abc.abstractmethod
    def predict(self, project: pathlib.Path, subset: set[pathlib.Path]) -> ModelAdaptor.ProjectPredictions:
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
        model_preds = self.adaptor.predict(mutable, subset=set(map(
            lambda r: pathlib.Path(r).relative_to(mutable),
            codemod.gather_files([str(mutable)])
        )))

        outpath = output_dir / f"__{self.adaptor.__class__.__qualname__}__.json"
        assert not outpath.is_file()

        self.logger.info(f"Writing model predictions to {outpath} for HiTyper to use")
        with outpath.open("w") as f:
            json.dump(model_preds.dict(exclude_none=True)["__root__"], f)

        htm.infertypes(
            _InferenceArguments(
                repo=str(mutable),
                output_directory=str(output_dir),
                topn=self.adaptor.topn(),
                recommendations=str(outpath),
            )
        )

        inferred_types_path = (
            str(output_dir) + "/" + str(mutable).replace("/", "_") + "_INFERREDTYPES.json"
        )
        repo_predictions = _HiTyperPredictions.parse_file(inferred_types_path)
        predictions = self._parse_predictions(repo_predictions, mutable)

        collections = []
        for topn in range(1, self.adaptor.topn() + 1):
            with utils.scratchpad(mutable) as sc:
                tw_hint_res = codemod.parallel_exec_transform_with_prettyprint(
                    transform=ParallelTypeApplier(
                        context=codemod.CodemodContext(),
                        path2batches=predictions,
                        topn=topn - 1,
                    ),
                    jobs=utils.worker_count(),
                    repo_root=str(sc),
                    files=[sc / s for s in subset],
                )
                self.logger.info(
                    utils.format_parallel_exec_result(
                        f"Annotated with HiTyper @ topn={topn}", result=tw_hint_res
                    )
                )
                collections.append(
                    build_type_collection(root=sc, allow_stubs=False, subset=subset).df.assign(
                        topn=topn
                    )
                )
        return (
            pd.concat(collections, ignore_index=True)
            .assign(method=self.method())
            .pipe(pt.DataFrame[InferredSchema])
        )

    def _parse_predictions(
        self, predictions: _HiTyperPredictions, project: pathlib.Path
    ) -> dict[pathlib.Path, list[Annotations]]:
        path2batchpreds: collections.defaultdict[
            pathlib.Path, list[Annotations]
        ] = collections.defaultdict(list)
        for file, scopes in predictions.__root__.items():
            for batch in self._batchify_scopes(scopes):
                annotations = Annotations.empty()

                for scope, predictions in batch.items():
                    scope_components = _derive_qname(scope)
                    for prediction in filter(
                        lambda p: p.category == _HiTyperPredictionCategory.LOCAL, predictions
                    ):
                        if (ty := prediction.type[0]) is None:
                            continue
                        annotation = libcst.Annotation(libcst.parse_expression(ty))

                        scope_key = ".".join((*scope_components, prediction.name))
                        annotations.attributes[scope_key] = annotation

                        scope_key = ".".join((*scope_components, "self", prediction.name))
                        annotations.attributes[scope_key] = annotation

                    if scope != "global@global":
                        # hityper does not infer self, so add it manually for non-global functions
                        if not scope.endswith("@global"):
                            parameters: list[libcst.Param] = [
                                libcst.Param(name=libcst.Name("self"))
                            ]
                        else:
                            parameters = []
                        returns: Optional[libcst.Annotation] = None

                        for prediction in filter(
                            lambda p: p.category != _HiTyperPredictionCategory.LOCAL, predictions
                        ):
                            ty = prediction.type[0]
                            if prediction.category == _HiTyperPredictionCategory.ARG:
                                parameters.append(
                                    libcst.Param(
                                        name=libcst.Name(prediction.name),
                                        annotation=(
                                            libcst.Annotation(libcst.parse_expression(ty))
                                            if ty is not None
                                            else None
                                        ),
                                    )
                                )

                            else:
                                returns = (
                                    libcst.Annotation(libcst.parse_expression(ty))
                                    if ty is not None
                                    else None
                                )

                        scope_key = ".".join(scope_components)

                        ps = libcst.Parameters(parameters)
                        fkey = FunctionKey.make(scope_key, ps)
                        annotations.functions[fkey] = FunctionAnnotation(ps, returns)

                path2batchpreds[pathlib.Path(file).relative_to(project)].append(annotations)
        return path2batchpreds

    def _batchify_scopes(self, scopes: _HiTyperScope2Prediction) -> list[_HiTyperScope2Prediction]:
        batches: list[_HiTyperScope2Prediction] = []

        for n in range(self.adaptor.topn()):
            batch = _HiTyperScope2Prediction()

            for scope, predictions in scopes.items():
                batch_predictions: list = []
                for prediction in predictions:
                    batch_predictions.append(
                        _HiTyperPrediction(
                            category=prediction.category,
                            name=prediction.name,
                            type=[prediction.type[n] if n < len(prediction.type) else None],
                        )
                    )

                batch[scope] = batch_predictions
            batches.append(batch)
        return batches


def _derive_qname(scope: str) -> list[str]:
    if scope == "global@global":
        return []

    *funcname, classname = scope.replace(",", ".").split("@")
    if classname == "global":
        return funcname

    return [classname, *funcname]