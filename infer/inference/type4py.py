import itertools
import operator
import pathlib
import requests

import json

from common.schemas import InferredSchema
from symbols.collector import TypeCollectorVistor
from ._base import PerFileInference

from common.annotations import (
    MultiVarAnnotations,
    FunctionKey,
    FunctionAnnotation,
)
import libcst as cst
from libcst import codemod
import libcst.metadata as metadata

from libsa4py.cst_transformers import TypeApplier

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
    fn_var_ln: dict[str, tuple[tuple[int, int], tuple[int, int]]] | None
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
    mod_var_ln: dict[str, tuple[tuple[int, int], tuple[int, int]]] | None
    variables_p: _Type4PyName2Hint | None


class _Type4PyAnswer(pydantic.BaseModel):
    error: str | None
    response: dict | None


class Type4Py(PerFileInference):
    method = "type4py"

    def _infer_file(self, relative: pathlib.Path) -> pt.DataFrame[InferredSchema]:
        with (self.project / relative).open() as f:
            r = requests.post("http://localhost:5001/api/predict?tc=0", f.read().encode("utf-8"))
            # print(r.text)

        answer = _Type4PyAnswer.parse_raw(r.text)
        # print(answer)

        if answer.error is not None:
            print(
                f"WARNING: {Type4Py.__qualname__} failed for {self.project / relative} - {answer.error}"
            )
            return InferredSchema.example(size=0)

        if answer.response is None:
            print(
                f"WARNING: {Type4Py.__qualname__} couldnt infer anything for {self.project / relative}"
            )
            return InferredSchema.example(size=0)

        src = (self.project / relative).open().read()
        module = cst.MetadataWrapper(cst.parse_module(src))

        # Callables
        # anno_maker = Type4Py2Annotations(answer=answer)
        inferred = module.visit(TypeApplier(f_processeed_dict=answer.response, apply_nlp=False))

        collector = TypeCollectorVistor.strict(
            context=codemod.CodemodContext(
                filename=str(self.project / relative),
                metadata_manager=metadata.FullRepoManager(
                    repo_root_dir=str(self.project),
                    paths=[str(self.project / relative)],
                    providers=[],
                ),
            )
        )
        inferred = collector.transform_module(inferred)

        return collector.collection.df.assign(method="type4py", topn=0).pipe(
            pt.DataFrame[InferredSchema]
        )
