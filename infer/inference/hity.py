import collections
import enum
import pathlib
from dataclasses import dataclass

import hityper.__main__ as hityper
import libcst
from libcst import metadata, matchers as m
import pandas as pd
import pandera.typing as pt
import pydantic
from libcst import codemod, helpers as h
from libcst.codemod.visitors._apply_type_annotations import (
    Annotations,
    FunctionKey,
    FunctionAnnotation,
)

from common import transformers as t
from common.annotations import ApplyTypeAnnotationsVisitor, TypeAnnotationRemover
from common.schemas import (
    InferredSchema,
)
from common.transformers import Actions
from symbols.collector import TypeCollectorVisitor
from ._base import PerFileInference
from ..insertion import TypeAnnotationApplierTransformer


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
    type4py: bool

    source = None
    recommendations = None


class _HiTyperPredictionCategory(str, enum.Enum):
    ARG = "arg"
    LOCAL = "local"
    RET = "return"


class _HiTyperPrediction(pydantic.BaseModel):
    category: _HiTyperPredictionCategory
    name: str
    type: list[str | None]

    class Config:
        use_enum_values = True


_HiTyperScope2Prediction = dict[str, list[_HiTyperPrediction]]


class _HiTyperPredictions(pydantic.BaseModel):
    __root__: dict[pathlib.Path, _HiTyperScope2Prediction]


class HiTyper(PerFileInference):
    method = "HiTyper"

    def __init__(self, project: pathlib.Path) -> None:
        super().__init__(project)
        self.output_dir = self.project / ".hityper"
        self.topn = 3

        # if self.output_dir.is_dir():
        #     shutil.rmtree(path=str(self.output_dir))

    def _infer_file(self, relative: pathlib.Path) -> pt.DataFrame[InferredSchema]:
        if not hasattr(self, "predictions"):
            if not self.output_dir.is_dir():
                self.output_dir.mkdir(parents=True, exist_ok=True)

            inferred_types_path = (
                str(self.output_dir)
                + "/"
                + str(self.project).replace("/", "_")
                + "_INFERREDTYPES.json"
            )
            if not pathlib.Path(inferred_types_path).is_file():
                self._predict()
            self.predictions = _HiTyperPredictions.parse_file(inferred_types_path)

        return self._predictions2df(self.predictions, relative)

    def _predict(self):
        # hityper.findusertype(
        #     _FindUserTypeArguments(
        #         repo=str(self.project), core=8, validate=True, output_directory=str(self.output_dir)
        #    )
        # )

        # hityper.gentdg(
        #     _GenTDGArguments(
        #         repo=str(self.project),
        #         optimize=True,
        #         output_directory=str(self.output_dir),
        #         output_format="json",
        #     )
        # )

        hityper.infertypes(
            _InferenceArguments(
                repo=str(self.project),
                output_directory=str(self.output_dir),
                topn=self.topn,
                type4py=True,
            )
        )

    def _predictions2df(
        self, predictions: _HiTyperPredictions, file: pathlib.Path
    ) -> pt.DataFrame[InferredSchema]:
        prediction_batches = []

        scopes = predictions.__root__.get(self.project / file, None)
        if scopes is None:
            return InferredSchema.example(size=0)

        src = libcst.parse_module(open(self.project / file).read())

        for topn, batch in enumerate(self._batchify_scopes(scopes), start=1):
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
                        parameters: list[libcst.Param] = [libcst.Param(name=libcst.Name("self"))]
                    else:
                        parameters = []
                    returns: libcst.Annotation | None = None

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

            context = codemod.CodemodContext(
                filename=str(self.project / file),
                metadata_manager=metadata.FullRepoManager(
                    repo_root_dir=str(self.project),
                    paths=[str(self.project / file)],
                    providers=[],
                ),
            )

            annotated = ApplyTypeAnnotationsVisitor(
                context=context,
                annotations=annotations,
                use_future_annotations=True,
            ).transform_module(src)

            collector = TypeCollectorVisitor.strict(context)
            annotated.visit(collector)

            prediction_batches.append(collector.collection.df.assign(topn=topn))

        return (
            pd.concat(prediction_batches)
            .assign(method=self.method)
            .pipe(pt.DataFrame[InferredSchema])
        )

    def _batchify_scopes(self, scopes: _HiTyperScope2Prediction) -> list[_HiTyperScope2Prediction]:
        batches: list[_HiTyperScope2Prediction] = []

        for n in range(self.topn):
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

        print(batches[0])
        return batches


def _derive_qname(scope: str) -> list[str]:
    if scope == "global@global":
        return []

    *funcname, classname = scope.replace(",", ".").split("@")
    if classname == "global":
        return funcname

    return [classname, *funcname]

    # scope = scope.removeprefix("global").removesuffix("global").replace(",", ".")
    # non_empties = list(filter(bool, scope.split("@")))
    # *funcname,
    # return list(filter(bool, reversed(scope.split("@"))))
