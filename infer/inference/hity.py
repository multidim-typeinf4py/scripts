import abc
import collections
import enum
import json
import logging
import tqdm
import pathlib
from dataclasses import dataclass
from typing import Optional

from hityper.utils import transformType4PyRecommendations
import hityper.__main__ as htm
import libcst
import pandas as pd
import pandera.typing as pt
import pydantic
from libcst import codemod, metadata
from common.annotations import ApplyTypeAnnotationsVisitor
from libcst.codemod.visitors._apply_type_annotations import (
    Annotations,
    FunctionKey,
    FunctionAnnotation,
)

from libsa4py.cst_extractor import Extractor
from type4py.deploy.infer import (
    get_dps_single_file,
    get_type_preds_single_file,
)

from infer.inference.t4py import PTType4Py


import utils
from common.schemas import InferredSchema
from symbols.collector import build_type_collection
from ._base import Inference, ProjectWideInference


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


MLVarPred4HiTyper = dict[str, list[tuple[str, float]]]


class MLFuncPred4HiTyper(pydantic.BaseModel):
    q_name: str
    params_p: Optional[dict[str, list[tuple[str, float]]]]
    ret_type_p: Optional[list[tuple[str, float]]]
    variables_p: MLVarPred4HiTyper


class MLClassPred4HiTyper(pydantic.BaseModel):
    q_name: str
    funcs: list[MLFuncPred4HiTyper]
    variables_p: MLVarPred4HiTyper


class MLPreds4HiTyper(pydantic.BaseModel):
    classes: list[MLClassPred4HiTyper]
    funcs: list[MLFuncPred4HiTyper]
    variables_p: MLVarPred4HiTyper


class ML4HiTyper(pydantic.BaseModel):
    __root__: dict[str, MLPreds4HiTyper]


class HiTyperModelAdaptor(abc.ABC):
    def __init__(self, model_path: pathlib.Path) -> None:
        self.model_path = model_path
        super().__init__()

    @abc.abstractmethod
    def topn(self) -> int:
        ...

    @abc.abstractmethod
    def predict(self, project: pathlib.Path, subset: set[pathlib.Path]) -> MLPreds4HiTyper:
        ...


class Type4Py2HiTyper(HiTyperModelAdaptor):
    def __init__(self, model_path: pathlib.Path, topn: int) -> None:
        super().__init__(model_path)
        self.type4py = PTType4Py(pre_trained_model_path=model_path, topn=topn)

    def topn(self) -> int:
        return self.type4py.topn

    def predict(self, project: pathlib.Path, subset: set[pathlib.Path]) -> ML4HiTyper:
        r = ML4HiTyper(__root__=dict())
        for file in subset:
            with (project / file).open() as f:
                src_f_read = f.read()
            type_hints = Extractor.extract(src_f_read, include_seq2seq=False).to_dict()

            (
                all_type_slots,
                vars_type_hints,
                params_type_hints,
                rets_type_hints,
            ) = get_dps_single_file(type_hints)

            if not any(h for h in (vars_type_hints, params_type_hints, rets_type_hints)):
                continue

            p = get_type_preds_single_file(
                type_hints,
                all_type_slots,
                (vars_type_hints, params_type_hints, rets_type_hints),
                self.type4py,
                filter_pred_types=False,
            )

            parsed = MLPreds4HiTyper.parse_obj(p)
            r.__root__[str(project.resolve() / file)] = parsed

        return r


class HiTyper(ProjectWideInference):
    method = "HiTyper"

    def __init__(self, model: HiTyperModelAdaptor) -> None:
        super().__init__()
        self.model = model
        # logging.getLogger(hityper.__name__).setLevel(logging.ERROR)

    def _infer_project(
        self, mutable: pathlib.Path, subset: set[pathlib.Path]
    ) -> pt.DataFrame[InferredSchema]:
        output_dir = mutable / ".hityper"
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True, exist_ok=True)

        # Create model predictions and pass to HiTyper
        self.logger.info(
            f"Inferring over {len(subset)} files in {output_dir} with {self.model.__class__.__qualname__}"
        )
        model_preds = self.model.predict(mutable, subset)

        outpath = output_dir / f"__{self.model.__class__.__qualname__}__.json"
        assert not outpath.is_file()

        self.logger.info(f"Writing model predictions to {outpath} for HiTyper to use")
        with outpath.open("w") as f:
            json.dump(model_preds.dict(exclude_none=True)["__root__"], f)

        htm.infertypes(
            _InferenceArguments(
                repo=str(mutable),
                output_directory=str(output_dir),
                topn=self.model.topn(),
                recommendations=str(outpath),
            )
        )

        inferred_types_path = (
            str(output_dir) + "/" + str(mutable).replace("/", "_") + "_INFERREDTYPES.json"
        )
        repo_predictions = _HiTyperPredictions.parse_file(inferred_types_path)
        predictions = self._parse_predictions(repo_predictions, mutable)

        collections = []
        for topn in range(1, self.model.topn() + 1):
            with utils.scratchpad(mutable) as sc:
                tw_hint_res = codemod.parallel_exec_transform_with_prettyprint(
                    transform=ParallelTypeApplier(
                        context=codemod.CodemodContext(),
                        path2batches=predictions,
                        topn=topn - 1,
                    ),
                    jobs=utils.worker_count(),
                    repo_root=str(sc),
                    files=[sc / pathlib.Path(f).relative_to(mutable) for f in model_preds.__root__],
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
            .assign(method=self.method)
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

        for n in range(self.model.topn()):
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


class _HiTyperType4PyTopN(HiTyper):
    def __init__(self, topn: int) -> None:
        super().__init__(
            Type4Py2HiTyper(model_path=pathlib.Path("/home/benji/Documents/Uni/heidelberg/05/masterarbeit/impls/scripts/models/type4py"), topn=topn)
        )


class HiTyperType4PyTop1(_HiTyperType4PyTopN):
    def __init__(self) -> None:
        super().__init__(topn=1)


class HiTyperType4PyTop3(_HiTyperType4PyTopN):
    def __init__(self) -> None:
        super().__init__(topn=3)


class HiTyperType4PyTop5(_HiTyperType4PyTopN):
    def __init__(self) -> None:
        super().__init__(topn=5)


class HiTyperType4PyTop10(_HiTyperType4PyTopN):
    def __init__(self) -> None:
        super().__init__(topn=10)
