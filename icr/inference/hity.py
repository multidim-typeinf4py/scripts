import collections
from dataclasses import dataclass
import enum
import pathlib
import shutil

from ._base import PerFileInference
from common.schemas import (
    InferredSchema,
    InferredSchemaColumns,
    TypeCollectionCategory,
    TypeCollectionSchema,
    TypeCollectionSchemaColumns,
)

import hityper.__main__ as hityper
import libcst as cst
import libcst.metadata as metadata

import pandas as pd
from pandas._libs import missing
import pandera.typing as pt
import pydantic

from common import _helper


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
    type: list[str]

    class Config:
        use_enum_values = True


_HiTyperScope2Prediction = dict[str, list[_HiTyperPrediction]]


class _HiTyperPredictions(pydantic.BaseModel):
    __root__: dict[pathlib.Path, _HiTyperScope2Prediction]


# NOTE: Similarly to Type4Py...
# NOTE: The returned JSON does not discriminate between attributes and variables,
# NOTE: meaning one prediction is overwritten by the other.
# NOTE: Unlike Type4Py, there are no line numbers...

# NOTE: Another bug:
# NOTE: Functions in functions are not recognised, and are marked as func@global


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
                self._predict()

            inferred_types_path = (
                str(self.output_dir)
                + "/"
                + str(self.project).replace("/", "_")
                + "_INFERREDTYPES.json"
            )
            self.predictions = _HiTyperPredictions.parse_file(inferred_types_path)

        return self._predictions2df(self.predictions, relative)

    def _predict(self):
        hityper.findusertype(
            _FindUserTypeArguments(
                repo=str(self.project), core=4, validate=True, output_directory=str(self.output_dir)
            )
        )

        hityper.gentdg(
            _GenTDGArguments(
                repo=str(self.project),
                optimize=True,
                output_directory=str(self.output_dir),
                output_format="json",
            )
        )

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
        df_updates: list[tuple[str, TypeCollectionCategory, str, str, int]] = []

        scopes = predictions.__root__.get(self.project / file, None)
        if scopes is None:
            return InferredSchema.example(size=0)

        src = cst.parse_module(open(self.project / file).read())
        module = metadata.MetadataWrapper(src)

        visitor = HiTyperLocalDisambig(predictions=scopes)
        module.visit(visitor)

        for scope, scope_predictions in visitor.retained.items():
            qname_prefix = scope

            for scope_pred in scope_predictions:
                for n, ty in enumerate(scope_pred.type[: self.topn] or [None]):
                    ty = ty or missing.NA

                    match scope_pred.category:
                        case _HiTyperPredictionCategory.ARG:
                            # Parameters are never global, simply join
                            df_updates.append(
                                (
                                    str(file),
                                    TypeCollectionCategory.CALLABLE_PARAMETER,
                                    f"{qname_prefix}.{scope_pred.name}",
                                    ty,
                                    n,
                                )
                            )

                        case _HiTyperPredictionCategory.RET:
                            # Returns are never global, simply join
                            df_updates.append(
                                (
                                    str(file),
                                    TypeCollectionCategory.CALLABLE_RETURN,
                                    qname_prefix,
                                    ty,
                                    n,
                                )
                            )

                        case _HiTyperPredictionCategory.LOCAL:
                            qname = (
                                scope_pred.name
                                if not qname_prefix
                                else f"{qname_prefix}.{scope_pred.name}"
                            )
                            df_updates.append(
                                (
                                    str(file),
                                    TypeCollectionCategory.VARIABLE,
                                    qname,
                                    ty,
                                    n,
                                )
                            )

        if not df_updates:
            return InferredSchema.example(size=0) 

        wout_ssa = [
            c
            for c in InferredSchemaColumns
            if c not in (InferredSchema.qname_ssa, InferredSchema.method)
        ]
        df = pd.DataFrame(df_updates, columns=wout_ssa)        

        return (
            df.assign(method=self.method)
            .pipe(_helper.generate_qname_ssas_for_file)
            .pipe(pt.DataFrame[InferredSchema])
        )


class HiTyperLocalDisambig(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (metadata.QualifiedNameProvider, metadata.ScopeProvider)

    def __init__(self, predictions: _HiTyperScope2Prediction) -> None:
        super().__init__()

        self._local_preds: _HiTyperScope2Prediction = collections.defaultdict(list)
        self.retained: _HiTyperScope2Prediction = collections.defaultdict(list)

        for scope, preds in predictions.items():
            for pred in preds:
                if pred.category == _HiTyperPredictionCategory.LOCAL:
                    self._local_preds[scope].append(pred)
                else:
                    derived = ".".join(_derive_qname(scope))
                    self.retained[derived].append(pred)

        """ self._local_preds = {
            scope: pred
            for scope, preds in predictions.items()
            for pred in preds
            if pred.category == _HiTyperPredictionCategory.LOCAL
        }

        self.retained: _HiTyperScope2Prediction = collections.defaultdict(
            list,
            (
                (".".join(_derive_qname(scope)), preds)
                for scope, preds in predictions.items()
                for pred in preds
                if pred.category != _HiTyperPredictionCategory.LOCAL
            ),
        ) """
        ...

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool | None:
        return self._handle_assn_tgt(node.target)

    def visit_AssignTarget(self, node: cst.AssignTarget) -> bool | None:
        return self._handle_assn_tgt(node.target)

    def _handle_assn_tgt(self, node: cst.BaseAssignTargetExpression) -> bool | None:
        match node:
            case cst.Name(name):
                full = name

            case cst.Attribute(cst.Name("self"), cst.Name(name)):
                full = f"self.{name}"

            case _:
                return None

        ast_scope = self.get_metadata(metadata.ScopeProvider, node)
        match ast_scope:
            case metadata.GlobalScope():
                scope = "global@global"

            case metadata.FunctionScope():
                qname = next(
                    iter(self.get_metadata(metadata.QualifiedNameProvider, ast_scope.node))
                )
                cleaned = qname.name.replace(".<locals>.", ".")
                if isinstance(ast_scope.parent, metadata.GlobalScope):
                    scope = f"{cleaned}@global"

                else:
                    *prelude, fname = cleaned.split(".")
                    scope = f"{fname}@{','.join(prelude)}"

            case metadata.ClassScope():
                # Do not handle class attributes
                return None

            case _:
                print(f"Unhandled scope {ast_scope}, {ast_scope.name} for {name}")
                return None

        if hints := self._local_preds.get(scope):
            pred = next(filter(lambda p: p.name == name, hints), None)
            if pred is not None:
                pred.name = full
                self._local_preds.pop(scope)
                self.retained[".".join(_derive_qname(scope))].append(pred)


def _derive_qname(scope: str) -> list[str]:
    scope = scope.removeprefix("global").removesuffix("global").replace(",", ".")
    return list(filter(bool, reversed(scope.split("@"))))
