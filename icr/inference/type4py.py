import pathlib
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


class Type4Py(PerFileInference):
    method = "type4py"

    def _infer_file(self, relative: pathlib.Path) -> pt.DataFrame[TypeCollectionSchema]:
        with (self.project / relative).open() as f:
            r = requests.post("http://localhost:5001/api/predict?tc=0", f.read())

        answer = _Type4PyAnswer.parse_raw(r.text)

        if answer.error is not None:
            print(
                f"WARNING: {Type4Py.__qualname__} failed for {self.project / relative} - {answer.error}"
            )
            return pt.DataFrame[TypeCollectionSchema](columns=TypeCollectionSchemaColumns)

        src = (self.project / relative).open().read()
        module = cst.MetadataWrapper(cst.parse_module(src))

        # Callables
        anno_maker = Type4Py2CallableAnnotations(answer=answer)
        module.visit(anno_maker)

        annotations = anno_maker.annotations

        # Variables
        for variable, hints in (answer.response.variables_p or dict()).items():
            (hint, _) = hints[0]
            annotations.attributes[variable] = cst.Annotation(cst.parse_expression(hint))

        # Variables in functions
        for func in answer.response.funcs:
            for variable, hints in (func.variables_p or dict()).items():
                if not hints:
                    continue
                (hint, _) = hints[0]
                annotations.attributes[f"{func.q_name}.{variable}"] = cst.Annotation(
                    cst.parse_expression(hint)
                )

        collection = TypeCollection.from_annotations(
            file=relative, annotations=annotations, strict=True
        )
        return collection.df


class Type4Py2CallableAnnotations(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (
        metadata.QualifiedNameProvider,
        metadata.ScopeProvider,
    )
    """Type4Py's predictions for callables are ordered alphabetically...
    Parse files to determine correct order"""

    def __init__(self, answer: _Type4PyAnswer) -> None:
        super().__init__()
        self.annotations = Annotations.empty()
        self._answer = answer.copy(deep=True)

        self._method_self: str | None = None

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        assert (fqname := self.get_metadata(metadata.QualifiedNameProvider, node)) is not None
        fqnames = [fn.name for fn in fqname]

        # Functions
        if any((f := func).q_name in fqnames for func in self._answer.response.funcs):
            key = FunctionKey.make(f.q_name, node.params)
            self.annotations.functions[key] = self._handle_func(f, node.params)

        # Methods
        elif any(
            (f := func).q_name in fqnames
            for clazz in self._answer.response.classes
            for func in clazz.funcs
        ):
            key = FunctionKey.make(f.q_name, node.params)
            self.annotations.functions[key] = self._handle_func(f, node.params)

        # Mark viable method
        if (scope := self.get_metadata(metadata.ScopeProvider, node)) is not None and isinstance(
            scope, metadata.ClassScope
        ):
            self._method_self = next(map(lambda p: p.name.value, node.params.params), None)

        else:
            self._method_self = None

    def leave_FunctionDef(self, _: cst.FunctionDef) -> None:
        self._method_self = None

    def visit_AssignTarget(self, node: cst.AssignTarget) -> bool | None:
        # Attempt to assign to class attributes
        if self._method_self is None:
            self._handle_class_attr(node)

        # Attempt to assign to instance attribute
        else:
            self._handle_inst_attr(node)

    def _handle_class_attr(self, node: cst.AssignTarget) -> None:
        # Class attributes
        if not m.matches(node.target, m.Name()):
            return

        for clazz in self._answer.response.classes:
            for variable, hints in (clazz.variables_p or dict()).items():
                if not hints or node.target.value != variable:
                    continue
                (hint, _) = hints[0]

                if clazz.q_name not in self.annotations.class_definitions:
                    self.annotations.class_definitions[clazz.q_name] = cst.ClassDef(
                        name=cst.Name(clazz.name), body=cst.IndentedBlock(body=[])
                    )

                body = self.annotations.class_definitions[f"{clazz.q_name}"].body
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

                self.annotations.class_definitions[clazz.q_name] = cst.ClassDef(
                    name=cst.Name(clazz.name), body=body
                )
                return None

    def _handle_inst_attr(self, node: cst.AssignTarget) -> None:
        assert self._method_self is not None
        if m.matches(node.target, m.Attribute(value=m.Name(self._method_self), attr=m.Name())):
            for clazz in self._answer.response.classes:
                for func in clazz.funcs:
                    for variable, hints in (func.variables_p or dict()).items():
                        if not hints or node.target.attr.value != variable:
                            continue
                        (hint, _) = hints[0]

                        self.annotations.attributes[
                            f"{func.q_name}.{self._method_self}.{variable}"
                        ] = cst.Annotation(cst.parse_expression(hint))
                        return None

        elif m.matches(node.target, m.Name()):
            for clazz in self._answer.response.classes:
                for func in clazz.funcs:
                    for variable, hints in (func.variables_p or dict()).items():
                        if not hints or node.target.value != variable:
                            continue
                        (hint, _) = hints[0]

                        self.annotations.attributes[f"{func.q_name}.{variable}"] = cst.Annotation(
                            cst.parse_expression(hint)
                        )
                        return None

    def _handle_func(self, f: _Type4PyFunc, params: cst.Parameters) -> FunctionAnnotation:
        if kp := f.params_p.get("args"):
            f.params_p.pop("args")

            hint, _ = kp[0]
            hint = cst.parse_expression(hint)
            star_arg = cst.Param(name=cst.Name("args"), annotation=cst.Annotation(hint))
        else:
            star_arg = None

        if kp := f.params_p.get("kwargs"):
            f.params_p.pop("kwargs")

            hint, _ = kp[0]
            hint = cst.parse_expression(hint)
            star_kwarg = cst.Param(name=cst.Name("kwargs"), annotation=cst.Annotation(hint))
        else:
            star_kwarg = None

        num_params: list[cst.Param] = list()
        kwonly_params: list[cst.Param] = list()
        posonly_params: list[cst.Param] = list()

        num_param_names = set(map(lambda p: p.name.value, params.params))
        kwonly_param_names = set(map(lambda p: p.name.value, params.kwonly_params))
        posonly_param_names = set(map(lambda p: p.name.value, params.posonly_params))

        for variable, hints in f.params_p.items():
            if not hints:
                continue

            hint, _ = hints[0]
            param = cst.Param(
                name=cst.Name(variable), annotation=cst.Annotation(cst.parse_expression(hint))
            )

            if variable in num_param_names:
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
            star_arg=star_arg,
            kwonly_params=kwonly_params,
            star_kwarg=star_kwarg,
            posonly_params=posonly_params,
        )

        match f.ret_type_p:
            case [(hint, _), *_]:
                annoexpr = cst.Annotation(cst.parse_expression(hint))
            case _:
                annoexpr = None

        return FunctionAnnotation(parameters=ps, returns=annoexpr)
