from dataclasses import dataclass
import enum
import pathlib
import shutil

from ._base import ProjectWideInference
from common.schemas import TypeCollectionCategory, TypeCollectionSchema, TypeCollectionSchemaColumns

import hityper.__main__ as hityper

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


class HiTyper(ProjectWideInference):
    method = "HiTyper"

    def _infer_project(self) -> pt.DataFrame[TypeCollectionSchema]:
        output_dir = self.project / ".hityper"
        if output_dir.is_dir():
            shutil.rmtree(path=str(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)

        hityper.findusertype(
            _FindUserTypeArguments(
                repo=str(self.project), core=4, validate=True, output_directory=str(output_dir)
            )
        )

        hityper.gentdg(
            _GenTDGArguments(
                repo=str(self.project),
                optimize=True,
                output_directory=str(output_dir),
                output_format="json",
            )
        )

        hityper.infertypes(
            _InferenceArguments(
                repo=str(self.project),
                output_directory=str(output_dir),
                topn=1,
                type4py=True,
            )
        )

        inferred_types_path = (
            str(output_dir) + "/" + str(self.project).replace("/", "_") + "_INFERREDTYPES.json"
        )
        predictions = _HiTyperPredictions.parse_file(inferred_types_path)
        # print(open(inferred_types_path).read())

        return self._predictions2df(predictions)

    def _predictions2df(
        self, predictions: _HiTyperPredictions
    ) -> pt.DataFrame[TypeCollectionSchema]:

        df_updates: list[tuple(str, TypeCollectionCategory, str, str)] = []

        for fullfile, scopes in predictions.__root__.items():
            file = str(fullfile.relative_to(self.project))

            for scope, scope_predictions in scopes.items():
                qname_prefix = _derive_qname(scope)

                for scope_pred in scope_predictions:
                    ty = scope_pred.type[0] if scope_pred.type else missing.NA

                    match scope_pred.category:
                        case _HiTyperPredictionCategory.ARG:
                            # Parameters are never global, simply join
                            df_updates.append(
                                (
                                    file,
                                    TypeCollectionCategory.CALLABLE_PARAMETER,
                                    f"{'.'.join(qname_prefix)}.{scope_pred.name}",
                                    ty,
                                )
                            )

                        case _HiTyperPredictionCategory.RET:
                            # Returns are never global, simply join
                            df_updates.append(
                                (
                                    file,
                                    TypeCollectionCategory.CALLABLE_RETURN,
                                    ".".join(qname_prefix),
                                    ty,
                                )
                            )

                        case _HiTyperPredictionCategory.LOCAL:
                            qname = (
                                scope_pred.name
                                if not qname_prefix
                                else f"{'.'.join(qname_prefix)}.{scope_pred.name}"
                            )
                            df_updates.append(
                                (
                                    file,
                                    TypeCollectionCategory.VARIABLE,
                                    qname,
                                    ty,
                                )
                            )

        return pt.DataFrame[TypeCollectionSchema](df_updates, columns=TypeCollectionSchemaColumns)


def _derive_qname(scope: str) -> list[str]:
    scope = scope.removeprefix("global").removesuffix("global").replace(",", ".")
    return list(filter(bool, reversed(scope.split("@"))))
