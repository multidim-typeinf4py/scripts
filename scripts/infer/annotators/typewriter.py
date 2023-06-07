import dataclasses
import logging
import pathlib
import typing

import libcst
from libcst import codemod
from typewriter.dltpy.preprocessing.pipeline import preprocessor

from scripts.infer.annotators import ParallelTopNAnnotator


@dataclasses.dataclass
class Parameter:
    fname: str
    pname: str
    ty: str


@dataclasses.dataclass
class Return:
    fname: str
    ty: str


class TWProjectApplier(
    ParallelTopNAnnotator[
        typing.Mapping[pathlib.Path, tuple[list[list[Parameter]], list[list[Return]]]],
        tuple[list[Parameter], list[Return]],
    ]
):
    def extract_predictions_for_file(
        self,
        path2topn: typing.Mapping[
            pathlib.Path, tuple[list[list[Parameter]], list[list[Return]]]
        ],
        path: pathlib.Path,
        topn: int,
    ) -> tuple[list[Parameter], list[Return]]:
        path = pathlib.Path(self.context.filename).relative_to(
            self.context.metadata_manager.root_path
        )

        if path not in path2topn:
            return [], []

        topn_parameters, topn_returns = path2topn[path]
        parameters, returns = topn_parameters[self.topn], topn_returns[self.topn]

        return parameters, returns

    def annotator(
        self, annotations: tuple[list[Parameter], list[Return]]
    ) -> codemod.Codemod:
        parameters, returns = annotations
        return TWFileApplier(
            context=self.context,
            parameters=parameters,
            returns=returns,
        )


class TWFileApplier(codemod.ContextAwareTransformer):
    def __init__(
        self,
        context: codemod.CodemodContext,
        parameters: list[Parameter],
        returns: list[Return],
    ) -> None:
        super().__init__(context)

        self.parameters = parameters
        self.param_cursor = 0

        self.returns = returns
        self.ret_cursor = 0

        self.logger = logging.getLogger(type(self).__qualname__)

    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        if not self.parameters and not self.returns:
            self.logger.warning(
                f"No predictions made for {self.context.filename}; returning unchanged tree"
            )
            return tree

        return tree.visit(self)

    def leave_FunctionDef(self, _, f: libcst.FunctionDef) -> libcst.FunctionDef:
        name = preprocessor.process_identifier(f.name.value)

        rc = self.ret_cursor
        try:
            while (r := self.returns[rc]).fname != name:
                rc += 1
        except IndexError:
            self.logger.warning(
                f"Cannot find prediction for function {f.name.value} in {self.context.filename}, assuming no prediction made"
            )
            return f

        if rc - self.ret_cursor > 1:
            self.logger.warning(
                f"Had to skip {rc - self.ret_cursor} ret entries to find {f.name.value}  in {self.context.filename}"
            )
        self.ret_cursor = rc + 1
        return f.with_changes(returns=self._read_tw_pred(r.ty))

    def leave_Param(self, _, param: libcst.Param) -> libcst.Param:
        if param.name.value == "self":
            # TypeWriter simply ignores self, with no further context checking
            return param

        name = preprocessor.process_identifier(param.name.value)

        pc = self.param_cursor
        try:
            while (p := self.parameters[pc]).pname != name:
                pc += 1
        except IndexError:
            self.logger.warning(
                f"Cannot find prediction for parameter {param.name.value} in {self.context.filename}, assuming no prediction made"
            )
            return param

        if pc - self.param_cursor > 1:
            self.logger.warning(
                f"Had to skip {pc - self.param_cursor} ret entries to find {param.name.value} in {self.context.filename}"
            )
        self.param_cursor = pc + 1
        return param.with_changes(annotation=self._read_tw_pred(p.ty))

    def _read_tw_pred(self, annotation: str | None) -> libcst.Annotation | None:
        if annotation is None or annotation in ("other", "unknown"):
            return None

        else:
            return libcst.Annotation(annotation=libcst.parse_expression(annotation))
