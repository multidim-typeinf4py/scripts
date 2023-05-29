from __future__ import annotations

import codecs
import dataclasses
import enum
import gzip
import itertools
import json
import operator
import pathlib
from typing import Iterator

import astunparse
import typed_ast.ast3
from type_check.annotater import AnnotationKind
from typed_ast.ast3 import (
    AnnAssign,
    Assign,
    Attribute,
    FunctionDef,
    Name,
    Tuple,
    arg,
    NodeVisitor,
    ClassDef,
)

from infer.inference._hityper import ModelAdaptor, HiTyper
from infer.inference.typilus import TypilusPrediction, Typilus


@dataclasses.dataclass
class FuncPred:
    q_name: str
    params_p: ModelAdaptor.VarPrediction = dataclasses.field(
        default_factory=dict
    )
    ret_type_p: list[ModelAdaptor.Prediction] = dataclasses.field(default_factory=list)
    variables_p: ModelAdaptor.VarPrediction = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass
class ClassPred:
    q_name: str
    funcs: list[FuncPred] = dataclasses.field(default_factory=list)
    variables_p: ModelAdaptor.VarPrediction = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass
class FilePredictions:
    classes: list[ClassPred] = dataclasses.field(default_factory=list)
    funcs: list[FuncPred] = dataclasses.field(default_factory=list)
    variables_p: ModelAdaptor.VarPrediction = dataclasses.field(
        default_factory=dict
    )


class TypilusHiTyperVisitor(NodeVisitor):
    class Context(enum.Enum):
        FUNCTION = enum.auto()
        CLASS = enum.auto()

    def __init__(self, predictions: list[TypilusPrediction]) -> None:
        self.predictions = predictions
        self.hityper_json = FilePredictions()

        self.qnames = list[str]()
        self.insertion_point_stack: list[FilePredictions | ClassPred | FuncPred] = [
            self.hityper_json
        ]

    @property
    def insertion_point(self):
        return self.insertion_point_stack[-1]

    def qname(self, symbol: str) -> str:
        return ".".join((*self.qnames, symbol))

    def add_inferred(
        self, symbol: str, prediction: str, probability: float, kind: AnnotationKind
    ) -> None:
        match kind:
            case AnnotationKind.VAR:
                assert hasattr(self.insertion_point, "variables_p")
                if symbol not in self.insertion_point.variables_p:
                    self.insertion_point.variables_p[symbol] = []
                self.insertion_point.variables_p[symbol].append((prediction, probability))

            case AnnotationKind.PARA:
                assert hasattr(self.insertion_point, "funcs")
                if symbol not in self.insertion_point.funcs[-1].params_p:
                    self.insertion_point.funcs[-1].params_p[symbol] = []
                self.insertion_point.funcs[-1].params_p[symbol].append((prediction, probability))

            case AnnotationKind.FUNC:
                assert hasattr(self.insertion_point, "funcs")
                assert symbol == self.insertion_point.funcs[-1].q_name
                self.insertion_point.funcs[-1].ret_type_p.append((prediction, probability))

    # ! arg exists in only Python 3; Python 2 uses Name.
    # ! See https://greentreesnakes.readthedocs.io/en/latest/nodes.html#arg
    def visit_arg(self, node: arg):
        for pred_type, pred_prob in (
            self.__extract_types_and_probs(node.arg, node.lineno, AnnotationKind.PARA) or []
        ):
            self.add_inferred(
                symbol=node.arg,
                prediction=pred_type,
                probability=pred_prob,
                kind=AnnotationKind.PARA,
            )

    def visit_ClassDef(self, node: ClassDef):
        if not hasattr(self.insertion_point, "classes"):
            # HiTyper does not support children at this insertion point, so do not recurse
            # and continue
            return

        self.insertion_point.classes.append(
            ClassPred(q_name=self.qname(node.name))
        )

        self.insertion_point_stack.append(self.insertion_point.classes[-1])
        self.qnames.append(node.name)

        self.generic_visit(node)

        self.insertion_point_stack.pop()
        self.qnames.pop()

    def visit_FunctionDef(self, node: FunctionDef):
        if not hasattr(self.insertion_point, "funcs"):
            # HiTyper does not support children at this insertion point, so do not recurse
            # and continue
            return

        fqname = self.qname(node.name)
        self.insertion_point.funcs.append(FuncPred(q_name=fqname))

        for pred_type, pred_prob in (
            self.__extract_types_and_probs(node.name, node.lineno, AnnotationKind.FUNC) or []
        ):
            self.add_inferred(
                symbol=fqname,
                prediction=pred_type,
                probability=pred_prob,
                kind=AnnotationKind.FUNC,
            )

        for a in node.args.args:
            self.visit(a)

        self.insertion_point_stack.append(self.insertion_point.funcs[-1])
        self.qnames.append(node.name)

        for s in node.body:
            self.visit(s)

        self.insertion_point_stack.pop()
        self.qnames.pop()

    def visit_AnnAssign(self, node: AnnAssign):
        self.generic_visit(node)

        varname = astunparse.unparse(node.target).strip()
        match node.target:
            case Name():
                symbol = node.target.id
            case Attribute() if node.value.id == "self":
                symbol = node.attr
            case _:
                return

        for pred_type, pred_prob in (
            self.__extract_types_and_probs(varname, node.lineno, AnnotationKind.VAR) or []
        ):
            self.add_inferred(
                symbol=symbol, prediction=pred_type, probability=pred_prob, kind=AnnotationKind.VAR
            )

    def visit_Assign(self, node: Assign):
        self.generic_visit(node)

        targets = node.targets
        # ! Consider only the case when "targets" has only one non-Tuple element
        if len(targets) > 1 or isinstance(targets[0], Tuple):
            return

        target = node.targets[0]
        varname = astunparse.unparse(target).strip()
        match target:
            case Name():
                symbol = target.id
            case Attribute() if node.value.id == "self":
                symbol = target.attr
            case _:
                return

        for pred_type, pred_prob in (
            self.__extract_types_and_probs(varname, node.lineno, AnnotationKind.VAR) or []
        ):
            self.add_inferred(
                symbol=symbol, prediction=pred_type, probability=pred_prob, kind=AnnotationKind.VAR
            )

    def __extract_types_and_probs(
        self, identifier, lineno, kind
    ) -> list[ModelAdaptor.Prediction] | None:
        pred_idx = self.__get_index(identifier, lineno, kind)
        if pred_idx == -1:
            return None

        pred_info = self.predictions[pred_idx]
        preds = pred_info["predicted_annotation_logprob_dist"]

        return preds

    def __get_index(self, name, lineno, kind):
        for idx, predline in enumerate(self.predictions):
            if (
                predline["name"] == name
                and predline["location"][0] == lineno
                and predline["annotation_type"] == kind.value
            ):
                return idx
        return -1


class Typilus2HiTyper(ModelAdaptor):
    def __init__(self, model_folder: pathlib.Path, topn: int) -> None:
        super().__init__(model_folder)
        self.typilus = Typilus(model_folder=model_folder, topn=topn)

    def topn(self) -> int:
        return self.typilus.topn

    def predict(
        self, project: pathlib.Path, subset: set[pathlib.Path]
    ) -> ModelAdaptor.ProjectPredictions:
        dataset = self.typilus.repo_to_dataset(project)
        predictions = self.typilus.predict(dataset, project / "typilus-predictions.json.gz")

        # Load predictions from disk and perform same sifting
        # that annotator from typilus does, i.e. select using fpath
        sifted = list[TypilusPrediction](
            filter(
                lambda p: any(f"/{s}" == p["provenance"] for s in subset),
                (prediction for batch in _load_json_gz(predictions.path) for prediction in batch),
            )
        )

        project_predictions = dict[str, ModelAdaptor.FilePredictions]()

        for provenance, predictions in itertools.groupby(sifted, key=operator.itemgetter("provenance")):
            provenance = pathlib.Path(provenance[1:])
            predictions = list[TypilusPrediction](predictions)

            fullpath = project / provenance

            visitor = TypilusHiTyperVisitor(predictions)

            typilus_ast = typed_ast.ast3.parse(source=fullpath.read_text())
            visitor.visit(typilus_ast)

            hity_json = dataclasses.asdict(visitor.hityper_json)
            project_predictions[str(fullpath.resolve())] = ModelAdaptor.FilePredictions.parse_obj(
                hity_json
            )

        return ModelAdaptor.ProjectPredictions(__root__=project_predictions)



def _load_json_gz(filename: str) -> Iterator[TypilusPrediction]:
    reader = codecs.getreader("utf-8")
    with gzip.open(filename) as f:
        for line in reader(f):
            yield json.loads(line, object_pairs_hook=TypilusPrediction)


class _HiTyperTypilusTopN(HiTyper):
    def __init__(self, topn: int) -> None:
        super().__init__(
            Typilus2HiTyper(
                model_folder=pathlib.Path("models") / "typilus",
                topn=topn,
            )
        )

    def method(self) -> str:
        return f"HiTyperTypilusN{self.adaptor.topn()}"


class HiTyperTypilusTop1(_HiTyperTypilusTopN):
    def __init__(self):
        super().__init__(topn=1)


class HiTyperTypilusTop3(_HiTyperTypilusTopN):
    def __init__(self):
        super().__init__(topn=3)


class HiTyperTypilusTop5(_HiTyperTypilusTopN):
    def __init__(self):
        super().__init__(topn=5)


class HiTyperTypilusTop10(_HiTyperTypilusTopN):
    def __init__(self):
        super().__init__(topn=10)
