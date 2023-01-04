import collections
from dataclasses import dataclass
import enum
import pathlib
import shutil

from ._base import PerFileInference
from common.schemas import TypeCollectionCategory, TypeCollectionSchema, TypeCollectionSchemaColumns

import hityper.__main__ as hityper
import libcst as cst
import libcst.metadata as metadata

from pandas._libs import missing
import pandera.typing as pt
import pydantic


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

        if self.output_dir.is_dir():
            shutil.rmtree(path=str(self.output_dir))

    def _infer_file(self, relative: pathlib.Path) -> pt.DataFrame[TypeCollectionSchema]:
        if not hasattr(self, "predictions"):
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.predictions = self._predict()

        return self._predictions2df(self.predictions, relative)

    def _predict(self) -> _HiTyperPredictions:
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
                topn=1,
                type4py=True,
            )
        )

        inferred_types_path = (
            str(self.output_dir) + "/" + str(self.project).replace("/", "_") + "_INFERREDTYPES.json"
        )
        return _HiTyperPredictions.parse_file(inferred_types_path)

    def _predictions2df(
        self, predictions: _HiTyperPredictions, file: pathlib.Path
    ) -> pt.DataFrame[TypeCollectionSchema]:
        df_updates: list[tuple(str, TypeCollectionCategory, str, str)] = []

        scopes = predictions.__root__[self.project / file]

        src = cst.parse_module(open(self.project / file).read())
        module = metadata.MetadataWrapper(src)

        visitor = HiTyperLocalDisambig(predictions=scopes)
        module.visit(visitor)

        for scope, scope_predictions in visitor.retained.items():
            qname_prefix = scope

            for scope_pred in scope_predictions:
                ty = scope_pred.type[0] if scope_pred.type else missing.NA

                match scope_pred.category:
                    case _HiTyperPredictionCategory.ARG:
                        # Parameters are never global, simply join
                        df_updates.append(
                            (
                                str(file),
                                TypeCollectionCategory.CALLABLE_PARAMETER,
                                f"{qname_prefix}.{scope_pred.name}",
                                ty,
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
                            )
                        )

        return pt.DataFrame[TypeCollectionSchema](df_updates, columns=TypeCollectionSchemaColumns)


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
                    self.retained[".".join(_derive_qname(scope))].append(pred)

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
