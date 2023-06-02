import dataclasses
import pathlib

import libcst
from libcst import metadata
from typewriter.dltpy.preprocessing.pipeline import preprocessor

from src.infer.inference._hityper import ModelAdaptor, HiTyper
from src.infer.inference.typewriter import _TypeWriter, Parameter, Return


@dataclasses.dataclass
class FuncPred:
    q_name: str
    params_p: ModelAdaptor.VarPrediction = dataclasses.field(default_factory=dict)
    ret_type_p: list[ModelAdaptor.Prediction] = dataclasses.field(default_factory=list)
    variables_p: dict = dataclasses.field(default_factory=dict)

@dataclasses.dataclass
class ClassPred:
    q_name: str
    funcs: list[FuncPred] = dataclasses.field(default_factory=list)
    variables_p: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class FilePredictions:
    classes: list[ClassPred] = dataclasses.field(default_factory=list)
    funcs: list[FuncPred] = dataclasses.field(default_factory=list)
    variables_p: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TypeWriterFuncPreds:
    fname: str
    param_types: list[list[tuple[str, str]]]
    ret_types: list[list[str]]


class _TypeWriter2HiTyper(libcst.CSTVisitor):
    METADATA_DEPENDENCIES = (
        metadata.ScopeProvider,
        metadata.QualifiedNameProvider,
    )

    def __init__(
        self, parameters: list[list[Parameter]], returns: list[list[Return]], topn: int
    ) -> None:
        super().__init__()

        self.parameters = parameters
        self.param_cursor = 0

        self.returns = returns
        self.ret_cursor = 0

        self.file_predictions = FilePredictions()
        self.insertion_point_stack: list[ClassPred | FilePredictions] = [
            self.file_predictions
        ]

        self.topn = topn

    def visit_ClassDef(self, node: libcst.ClassDef) -> None:
        self.file_predictions.classes.append(ClassPred(self._retrieve_qname(node)))
        self.insertion_point_stack.append(self.file_predictions.classes[-1])

    def leave_ClassDef(self, node: libcst.ClassDef) -> None:
        self.insertion_point_stack.pop()

    def visit_FunctionDef(self, f: libcst.FunctionDef) -> None:
        params_p = ModelAdaptor.VarPrediction()
        ret_type_p = self._visit_rettype(f)

        # TypeWriter only looks at normal arguments, as demonstrated by extraction code:
        # arg_names: List[str] = [arg.arg for arg in node.args.args]
        # arg_types: List[str] = [self.pretty_print(arg.annotation) for arg in node.args.args]
        for param in f.params.params:
            params_p[param.name.value] = self._visit_param(param)

        self.insertion_point_stack[-1].funcs.append(
            FuncPred(
                q_name=self._retrieve_qname(f),
                params_p=params_p,
                ret_type_p=ret_type_p,
            )
        )

    def _visit_param(self, param: libcst.Param) -> list[ModelAdaptor.Prediction]:
        if param.name.value == "self":
            return []

        name = preprocessor.process_identifier(param.name.value)

        pc = self.param_cursor
        try:
            while (p := self.parameters[pc])[0].pname != name:
                pc += 1
        except IndexError:
            return []
        self.param_cursor = pc + 1

        predictions = [
            (self._read_tw_pred(pred.ty), 1 / (2**frac))
            for pred, frac in zip(p, range(1, self.topn + 1))
        ]
        return list(filter(lambda prediction: prediction[0] is not None, predictions))

    def _visit_rettype(self, f: libcst.FunctionDef) -> list[ModelAdaptor.Prediction]:
        name = preprocessor.process_identifier(f.name.value)

        rc = self.ret_cursor
        try:
            while (rets := self.returns[rc])[0].fname != name:
                rc += 1
        except IndexError:
            return []
        self.ret_cursor = rc + 1

        predictions = [
            (self._read_tw_pred(pred.ty), 1 / (2**frac))
            for pred, frac in zip(rets, range(1, self.topn + 1))
        ]
        return list(filter(lambda prediction: prediction[0] is not None, predictions))

    def _read_tw_pred(self, annotation: str | None) -> str | None:
        if annotation is None or annotation == "other":
            return None
        return annotation

    def _retrieve_qname(self, node: libcst.FunctionDef | libcst.ClassDef) -> str:
        qnames = self.get_metadata(metadata.QualifiedNameProvider, node)
        qname = next(iter(qnames))

        return qname.name.replace(".<locals>.", ".")


class TypeWriterAdaptor(ModelAdaptor):
    def __init__(self, model_path: pathlib.Path, topn: int):
        super().__init__(model_path)
        self.typewriter = _TypeWriter(model_path, topn=topn)

    def topn(self) -> int:
        return self.typewriter.topn

    def predict(
        self, project: pathlib.Path, subset: set[pathlib.Path]
    ) -> ModelAdaptor.ProjectPredictions:
        file2predictions: dict[pathlib.Path, tuple | None] = {
            project / s: self.typewriter.infer_for_file(project, s) for s in subset
        }

        hityper_predictions = dict[str, FilePredictions]()

        for path, model_preds in file2predictions.items():
            if not model_preds:
                continue

            parameters, returns = self.transform_predictions(*model_preds)
            visitor = _TypeWriter2HiTyper(parameters, returns, self.topn())

            metadata.MetadataWrapper(
                module=libcst.parse_module(path.read_text()), unsafe_skip_copy=True
            ).visit(visitor)

            file_predictions = dataclasses.asdict(visitor.file_predictions)
            hityper_predictions[
                str(path.resolve())
            ] = ModelAdaptor.FilePredictions.parse_obj(file_predictions)

        return ModelAdaptor.ProjectPredictions(__root__=hityper_predictions)

    def transform_predictions(
        self,
        topn_parameters: list[list[Parameter]],
        topn_returns: list[list[Return]],
    ) -> tuple[list[list[Parameter]], list[list[Return]]]:
        param_types = [
            [topn_parameters[n][x] for n in range(self.topn())]
            for x in range(len(topn_parameters[0]))
        ]
        ret_types = [
            [topn_returns[n][x] for n in range(self.topn())]
            for x in range(len(topn_returns[0]))
        ]

        return param_types, ret_types


class _HiTyperTypeWriterTopN(HiTyper):
    def __init__(self, topn: int) -> None:
        super().__init__(
            TypeWriterAdaptor(
                model_path=pathlib.Path("models") / "typewriter",
                topn=topn,
            )
        )

    def method(self) -> str:
        return f"HiTyperTypeWriterN{self.adaptor.topn()}"


class HiTyperTypeWriterTop1(_HiTyperTypeWriterTopN):
    def __init__(self) -> None:
        super().__init__(topn=1)


class HiTyperTypeWriterTop3(_HiTyperTypeWriterTopN):
    def __init__(self) -> None:
        super().__init__(topn=3)


class HiTyperTypeWriterTop5(_HiTyperTypeWriterTopN):
    def __init__(self) -> None:
        super().__init__(topn=5)


class HiTyperTypeWriterTop10(_HiTyperTypeWriterTopN):
    def __init__(self) -> None:
        super().__init__(topn=10)
