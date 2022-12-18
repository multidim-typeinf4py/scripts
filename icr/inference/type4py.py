import functools
import itertools
import operator
import pathlib
import typing
import requests

from common.schemas import TypeCollectionSchema, TypeCollectionSchemaColumns
from common.storage import TypeCollection
from ._base import PerFileInference

from libcst.codemod.visitors._apply_type_annotations import (
    Annotations,
    FunctionKey,
    FunctionAnnotation,
)
import libcst as cst
import libcst.matchers as m
import libcst.metadata as metadata
from libcst.metadata.scope_provider import LocalScope

import pandera.typing as pt
import pydantic

# Based on https://github.com/pydantic/pydantic/issues/3295#issuecomment-936594175
class NullifyEmptyDictModel(pydantic.BaseModel):
    @pydantic.root_validator(pre=True)
    def remove_empty(cls, values):
        fields = list(values.keys())
        for field in fields:
            value = values[field]
            if isinstance(value, dict):
                if not values[field]:
                    values[field] = None
        return values


_Type4PyName2Hint = dict[str, list[tuple[str, float]]]


class _Type4PyFunc(NullifyEmptyDictModel):
    q_name: str
    fn_var_ln: dict[str, tuple[tuple[int, int], tuple[int, int]]]
    params_p: _Type4PyName2Hint | None
    ret_type_p: list[tuple[str, float]] | None
    variables_p: _Type4PyName2Hint | None


class _Type4PyClass(NullifyEmptyDictModel):
    name: str
    q_name: str
    funcs: list[_Type4PyFunc]
    variables_p: _Type4PyName2Hint | None


class _Type4PyResponse(NullifyEmptyDictModel):
    classes: list[_Type4PyClass]
    funcs: list[_Type4PyFunc]
    mod_var_ln: dict[str, tuple[tuple[int, int], tuple[int, int]]]
    variables_p: _Type4PyName2Hint | None


class _Type4PyAnswer(pydantic.BaseModel):
    error: str | None
    response: _Type4PyResponse


# NOTE: THERE IS A BUG / THERE IS AMBIGUITY IN THE PREDICTIONS OF TYPE4PY!!
# NOTE: The returned JSON does not discriminate between attributes and variables,
# NOTE: meaning one prediction is overwritten by the other. MRE follows:

# NOTE: class C:
# NOTE:     def f(self, i):
# NOTE:         self.very_specific_name = 5
# NOTE:         very_specific_name = "String"

# NOTE: This is "handled" by referencing the span of the variable stated in Type4Py's JSON


class Type4Py(PerFileInference):
    method = "type4py"

    def _infer_file(self, relative: pathlib.Path) -> pt.DataFrame[TypeCollectionSchema]:
        with (self.project / relative).open() as f:
            r = requests.post("http://localhost:5001/api/predict?tc=0", f.read())
            print(r.text)

        answer = _Type4PyAnswer.parse_raw(r.text)

        if answer.error is not None:
            print(
                f"WARNING: {Type4Py.__qualname__} failed for {self.project / relative} - {answer.error}"
            )
            return pt.DataFrame[TypeCollectionSchema](columns=TypeCollectionSchemaColumns)

        src = (self.project / relative).open().read()
        module = cst.MetadataWrapper(cst.parse_module(src))

        # Callables
        anno_maker = Type4Py2Annotations(answer=answer)
        module.visit(anno_maker)

        annotations = anno_maker.annotations
        collection = TypeCollection.from_annotations(
            file=relative, annotations=annotations, strict=True
        )
        return collection.df


class Type4Py2Annotations(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (
        metadata.PositionProvider,
        metadata.QualifiedNameProvider,
        metadata.ScopeProvider,
    )
    """Type4Py's predictions for callables are ordered alphabetically...
    And reassigning simply overwrites the previous prediction
    Parse files to determine correct annotating"""

    def __init__(self, answer: _Type4PyAnswer) -> None:
        super().__init__()
        self.annotations = Annotations.empty()
        self._answer = answer.copy(deep=True)

        # qualified callable name and the name of the class's this (None in functions)
        self._method_callable: tuple[str, str | None] | None = None

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        assert (fqname := self.get_metadata(metadata.QualifiedNameProvider, node)) is not None
        fqnames = [fn.name for fn in fqname]
        assert len(fqnames) == 1

        # Functions
        indirect_clazz_scope = self._class_scope(node)
        if indirect_clazz_scope is None and (f := self._func(func_qname=fqnames[0])) is not None:
            key = FunctionKey.make(f.q_name, node.params)
            self.annotations.functions[key] = self._handle_func(f, node.params)

        # Methods
        if (
            indirect_clazz_scope is not None
            and (class_qname := indirect_clazz_scope.name) is not None
            and (f := self._method(clazz_qname=class_qname, method_qname=fqnames[0])) is not None
        ):
            key = FunctionKey.make(f.q_name, node.params)
            self.annotations.functions[key] = self._handle_func(f, node.params)

        # Mark callables when trying to assign variables
        if (scope := self.get_metadata(metadata.ScopeProvider, node)) is not None:
            match scope:
                # Method
                case metadata.ClassScope():
                    method_self = next(map(lambda p: p.name.value, node.params.params), None)

                # Function
                case _:
                    method_self = None

            self._method_callable = (f.q_name, method_self)

        else:
            self._method_callable = None

        return None

    def leave_FunctionDef(self, _: cst.FunctionDef) -> None:
        self._method_callable = None
        return None

    def visit_AssignTarget(self, node: cst.AssignTarget) -> bool | None:
        scope = self.get_metadata(metadata.ScopeProvider, node)

        # Attempt to assign to variables and handle bug with identically named vars
        # code = cst.Module([node]).code
        match (self._method_callable, scope.parent, scope):
            # Method
            case (str(qmethod), str(method_self)), metadata.ClassScope(), metadata.FunctionScope():
                # print(f"{code} is being treated as assignment in method")
                self._handle_assgn_in_method(
                    node,
                    clazz_qname=scope.parent.name,
                    method_qname=qmethod,
                    method_self=method_self,
                )

            # Function
            case (str(qfunc), _), _, metadata.FunctionScope():
                # print(f"{code} is being treated as assignment in function")
                self._handle_assgn_in_function(node, funcqname=qfunc)

            # Class attr
            case None, _, metadata.ClassScope():
                # print(f"{code} is being treated as assignment to class attr")
                self._handle_class_attr(node, clazz_qname=scope.name)

            # Global variable
            case None, _, metadata.GlobalScope():
                self._handle_global_var(node)

            case _:
                # print(f"{code} is being ignored")
                return None

    def _handle_class_attr(self, node: cst.AssignTarget, clazz_qname: str) -> None:
        # Class attributes
        if not isinstance(node.target, cst.Name):
            return None

        attr = self._clazz_attr(clazz_qname=clazz_qname, attr_name=node.target.value)
        if attr is None:
            return

        variable, hints = attr
        (hint, _) = hints[0]

        clazz_name = clazz_qname.split(".")[-1]
        if clazz_qname not in self.annotations.class_definitions:
            self.annotations.class_definitions[clazz_qname] = cst.ClassDef(
                name=cst.Name(clazz_name), body=cst.IndentedBlock(body=[])
            )

        body = self.annotations.class_definitions[f"{clazz_qname}"].body
        body = cst.IndentedBlock(
            body=[
                *body.body,
                cst.SimpleStatementLine(
                    body=[
                        cst.AnnAssign(
                            target=cst.Name(variable),
                            annotation=cst.Annotation(cst.parse_expression(hint)),
                        )
                    ]
                ),
            ]
        )

        self.annotations.class_definitions[clazz_qname] = cst.ClassDef(
            name=cst.Name(clazz_name), body=body
        )
        return None

    def _handle_func(self, f: _Type4PyFunc, params: cst.Parameters) -> FunctionAnnotation:
        num_params: list[cst.Param] = list()
        kwonly_params: list[cst.Param] = list()
        posonly_params: list[cst.Param] = list()

        posonly_param_names = set(map(lambda p: p.name.value, params.posonly_params))
        kwonly_param_names = set(map(lambda p: p.name.value, params.kwonly_params))
        param_names = set(map(lambda p: p.name.value, params.params))

        for variable, hints in (f.params_p or dict()).items():
            if not hints:
                continue

            hint, _ = hints[0]
            param = cst.Param(
                name=cst.Name(variable), annotation=cst.Annotation(cst.parse_expression(hint))
            )

            if variable in param_names:
                num_params.append(param)
            elif variable in kwonly_param_names:
                kwonly_params.append(param)
            elif variable in posonly_param_names:
                posonly_params.append(param)
            else:
                raise RuntimeError(
                    f"{variable} is neither a parameter, nor a pos-only, not a kw-only parameter of {f.q_name}"
                )

        ps = cst.Parameters(
            params=num_params,
            kwonly_params=kwonly_params,
            posonly_params=posonly_params,
        )

        match f.ret_type_p:
            case [(hint, _), *_]:
                annoexpr: cst.Annotation | None = cst.Annotation(cst.parse_expression(hint))
            case _:
                annoexpr = None

        return FunctionAnnotation(parameters=ps, returns=annoexpr)

    def _handle_assgn_in_method(
        self, node: cst.AssignTarget, clazz_qname: str, method_qname: str, method_self: str
    ) -> None:
        match node.target:
            case cst.Name(value=ident):
                span = self.get_metadata(metadata.PositionProvider, node.target)

            case cst.Attribute(value=cst.Name(method_self), attr=cst.Name(ident)):
                span = self.get_metadata(metadata.PositionProvider, node.target.attr)

            case _:
                return None

        if (func := self._method(clazz_qname=clazz_qname, method_qname=method_qname)) is None:
            print(f"cannot find {clazz_qname=} @ {method_qname=}")
            return
        for variable, hints in (func.variables_p or dict()).items():
            if not hints or ident != variable:
                continue

            if (resp_span := func.fn_var_ln.get(variable)) is None:
                continue

            if not self._span_matches(node.target, resp_span, span):
                continue

            (hint, _) = hints[0]

            self.annotations.attributes[
                f"{func.q_name}.{method_self}.{variable}"
                if isinstance(node.target, cst.Attribute)
                else f"{func.q_name}.{variable}"
            ] = cst.Annotation(cst.parse_expression(hint))
            return None

    def _handle_assgn_in_function(self, node: cst.AssignTarget, funcqname: str) -> None:
        if not isinstance(node.target, cst.Name):
            return None

        if (func := self._func(func_qname=funcqname)) is None:
            return None

        for variable, hints in (func.variables_p or dict()).items():
            if not hints or node.target.value != variable:
                continue

            if (resp_span := func.fn_var_ln.get(variable)) is None:
                continue

            if not self._span_matches(node.target, resp_span):
                continue

            (hint, _) = hints[0]
            self.annotations.attributes[
                f"{func.q_name}.{variable}"
                if isinstance(node.target, cst.Attribute)
                else f"{func.q_name}.{variable}"
            ] = cst.Annotation(cst.parse_expression(hint))

            return None

    def _handle_global_var(self, node: cst.AssignTarget) -> None:
        if not isinstance(node.target, cst.Name) or not self._answer.response.variables_p:
            return None

        if not (hints := self._answer.response.variables_p[node.target.value]):
            return None

        if not self._span_matches(node.target, self._answer.response.mod_var_ln[node.target.value]):
            return None

        (hint, _) = hints[0]
        self.annotations.attributes[node.target.value] = cst.Annotation(cst.parse_expression(hint))

        return None

    def _clazz_attr(
        self, clazz_qname: str, attr_name: str
    ) -> tuple[str, list[tuple[str, float]]] | None:
        cs = [
            (variable, hints)
            for clazz in self._answer.response.classes
            if clazz.variables_p
            for variable, hints in clazz.variables_p.items()
            if variable == attr_name
            if clazz.variables_p is not None and clazz.q_name == clazz_qname
        ]

        assert len(cs) <= 1
        return cs[0] if cs else None

    def _func(self, func_qname: str) -> _Type4PyFunc | None:
        fs = [func for func in self._answer.response.funcs if func.q_name == func_qname]

        assert len(fs) <= 1
        return fs[0] if fs else None

    def _method(self, clazz_qname: str, method_qname: str) -> _Type4PyFunc | None:
        ms = [
            func
            for clazz in self._answer.response.classes
            for func in clazz.funcs
            if func.q_name == method_qname
            if clazz.q_name == clazz_qname
        ]

        assert len(ms) <= 1

        return ms[0] if ms else None

    def _class_scope(self, node: cst.CSTNode) -> metadata.ClassScope | None:
        def _scope_recursion(scope: metadata.Scope) -> metadata.ClassScope | None:
            if isinstance(scope, metadata.BuiltinScope):
                return None

            if isinstance(scope, metadata.ClassScope):
                return scope

            return _scope_recursion(scope.parent)

        scope = self.get_metadata(metadata.ScopeProvider, node)
        if scope is None:
            return None

        if isinstance(scope, metadata.ClassScope):
            return scope

        return _scope_recursion(scope.parent)

    def _span_matches(
        self,
        node: cst.CSTNode,
        resp_span: tuple[tuple[int, int], tuple[int, int]],
        rnge: metadata.CodeRange | None = None,
    ) -> bool:
        span = rnge or self.get_metadata(metadata.PositionProvider, node)

        (resp_start_line, resp_start_col), (resp_stop_line, resp_stop_col) = resp_span
        align = [
            (resp_start_line, span.start.line),
            (resp_start_col, span.start.column),
            (resp_stop_line, span.end.line),
            (resp_stop_col, span.end.column),
        ]

        return all(itertools.starmap(operator.eq, align))
