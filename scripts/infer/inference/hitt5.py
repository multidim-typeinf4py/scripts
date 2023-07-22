import collections
import dataclasses
import itertools
import operator
import os
import pathlib

from typet5.function_decoding import RolloutPrediction
from typet5.static_analysis import (
    ProjectPath,
    SignatureMap,
    PythonProject,
    VariableSignature,
    ElemSignature,
    FunctionSignature,
)

from ._hityper import ModelAdaptor, HiTyper
from ._utils import wrapped_partial
from ..structure import DatasetFolderStructure


class TT5Adaptor(ModelAdaptor):
    def __init__(self, topn: int) -> None:
        super().__init__()
        self._topn = topn

    def topn(self) -> int:
        return self._topn

    def predict(
        self, project: pathlib.Path, subset: set[pathlib.Path]
    ) -> ModelAdaptor.ProjectPredictions:
        from scripts.common.output import InferenceArtifactIO

        io = InferenceArtifactIO(
            artifact_root=pathlib.Path(os.environ["ARTIFACT_ROOT"]).parent
            / f"typet5topn{self.topn()}",
            dataset=DatasetFolderStructure(pathlib.Path(os.environ["DATASET_ROOT"])),
            repository=pathlib.Path(os.environ["REPOSITORY"]),
            tool_name=f"TypeT5TopN{self.topn()}",
            task=os.environ["TASK"],
        )

        # No need to trim predictions; HiTyper does this for us via
        # tdg.recommendType(self, ..., topn)
        cached_predictions: RolloutPrediction
        (cached_predictions,) = io.read()
        pred_assignments: SignatureMap = cached_predictions.final_sigmap

        # Make relative to temporary project root

        root = dict[str, ModelAdaptor.FilePredictions]()

        #
        for file in subset:
            module = PythonProject.rel_path_to_module_name(file)
            module_predictions = {
                project_path.path: signature
                for project_path, signature in pred_assignments.items()
                if project_path.module == module
            }

            root[str(project.resolve() / file)] = signatures_to_type4py_format(module_predictions)

        return ModelAdaptor.ProjectPredictions(__root__=root)


@dataclasses.dataclass
class FunctionPrediction:
    fname: str
    signature: FunctionSignature


@dataclasses.dataclass
class MethodPrediction:
    clazz: str
    function: FunctionPrediction


@dataclasses.dataclass
class VariablePrediction:
    vname: str
    signature: VariableSignature


@dataclasses.dataclass
class AttributePrediction:
    clazz: str
    variable: VariablePrediction


def signatures_to_type4py_format(
    predictions: dict[str, ElemSignature]
) -> ModelAdaptor.FilePredictions:
    from scripts.common.ast_helper import _stringify

    classes = list()
    funcs = list()
    variables = list()

    methods = collections.defaultdict[str, dict[str, MethodPrediction]](dict)
    functions = dict[str, FunctionPrediction]()

    attributes = collections.defaultdict[str, dict[str, AttributePrediction]](dict)
    globls = dict[str, VariablePrediction]()

    for symbol_path, signature in sorted(predictions.items()):
        match signature:
            case VariableSignature(_, True):
                clazz, vname = symbol_path.split(".")
                qname = f"{clazz}.{vname}"
                attributes[clazz][qname] = AttributePrediction(
                    clazz=clazz,
                    variable=VariablePrediction(vname=qname, signature=signature),
                )

            case VariableSignature(_, False):
                (vname,) = symbol_path.split(".")
                globls[vname] = VariablePrediction(vname=vname, signature=signature)

            case FunctionSignature(_, _, True):
                *clazz, fname = symbol_path.split(".")
                clazz = ".".join(clazz)

                qname = f"{clazz}.{fname}"
                methods[clazz][qname] = MethodPrediction(
                    clazz=clazz, function=FunctionPrediction(fname=qname, signature=signature)
                )

            case FunctionSignature(_, _, False):
                (fname,) = symbol_path.split(".")
                functions[fname] = FunctionPrediction(fname=fname, signature=signature)

    for class_qname in methods | attributes:
        class_methods = list()
        for method_qname, method_prediction in methods.get(class_qname, {}).items():
            class_methods.append(
                ModelAdaptor.FuncPrediction(
                    q_name=method_qname,
                    params_p={
                        param: [(_stringify(anno), 0.9)]
                        for param, anno in method_prediction.function.signature.params.items()
                        if param is not None
                    },
                    ret_type_p=(
                        [(_stringify(ret), 0.9)]
                        if (ret := method_prediction.function.signature.returns) is not None
                        else None
                    ),
                    variables_p={},
                )
            )

        attrs = dict()
        for attr_qname, attr_prediction in attributes.get(class_qname, {}).items():
            anno = attr_prediction.variable.signature.annot
            attrs[attr_qname] = [(_stringify(anno), 0.9)]

        classes.append(
            ModelAdaptor.ClassPrediction(q_name=class_qname, funcs=class_methods, variables_p=attrs)
        )

    return ModelAdaptor.FilePredictions(
        classes=classes,
        funcs=funcs,
        variables_p=variables,
    )


class HiTT5TopN(HiTyper):
    def __init__(self, topn: int) -> None:
        super().__init__(TT5Adaptor(topn=topn))

    def method(self) -> str:
        return f"HiType4PyN{self.adaptor.topn()}"


HiTT5TopNTop1 = wrapped_partial(HiTT5TopN, topn=1)
HiTT5TopNTop3 = wrapped_partial(HiTT5TopN, topn=3)
HiTT5TopNTop5 = wrapped_partial(HiTT5TopN, topn=5)
HiTT5TopNTop10 = wrapped_partial(HiTT5TopN, topn=10)
