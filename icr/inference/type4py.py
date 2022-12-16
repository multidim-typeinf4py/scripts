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
    # params: dict[str, str] | None
    ret_type_p: list[tuple[str, float]] | None


class _Type4PyClass(NullifyEmptyDictModel):
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


class Type4Py(PerFileInference):
    method = "type4py"

    def _infer_file(self, relative: pathlib.Path) -> pt.DataFrame[TypeCollectionSchema]:
        with (self.project / relative).open() as f:
            r = requests.post("http://localhost:5001/api/predict?tc=0", f.read())
            # print(r.text)

        answer = _Type4PyAnswer.parse_raw(r.text)
        if answer.error is not None:
            print(
                f"WARNING: {Type4Py.__qualname__} failed for {self.project / relative} - {answer.error}"
            )
            return pt.DataFrame[TypeCollectionSchema](columns=TypeCollectionSchemaColumns)

        src = (self.project / relative).open().read()
        module = cst.MetadataWrapper(cst.parse_module(src))

        anno_maker = Type4Py2Annotations(answer=answer)
        module.visit(anno_maker)

        collection = TypeCollection.from_annotations(
            file=relative, annotations=anno_maker.annotations, strict=True
        )
        return collection.df


class Type4Py2Annotations(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (metadata.QualifiedNameProvider,)
    """Type4Py predictions are ordered alphabetically...
    Parse files to determine correct order"""

    def __init__(self, answer: _Type4PyAnswer) -> None:
        super().__init__()
        self.annotations = Annotations.empty()
        self._answer = answer.copy(deep=True)

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

        for variable, hint in f.params_p.items():
            if not hint:
                continue
            param = cst.Param(
                name=cst.Name(variable), annotation=cst.Annotation(cst.parse_expression(hint[0][0]))
            )

            if variable in num_param_names:
                num_params.append(param)
            elif variable in kwonly_param_names:
                kwonly_params.append(param)
            elif variable in posonly_param_names:
                posonly_params.append(param)
            else:
                raise RuntimeError(
                    f"{variable} is neither a parameter, nor a kw-only parameter of {f.q_name}"
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
