import pathlib
from typing import Optional

import requests


from common.schemas import InferredSchema
from symbols.collector import TypeCollectorVisitor
from ._base import PerFileInference

import libcst as cst
from libcst import codemod
import libcst.metadata as metadata

from libsa4py.cst_transformers import TypeApplier

import pandera.typing as pt
import pydantic


class _Type4PyAnswer(pydantic.BaseModel):
    error: Optional[str]
    response: Optional[dict]


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

        collector = TypeCollectorVisitor.strict(
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

        return collector.collection.df.assign(method="type4py", topn=1).pipe(
            pt.DataFrame[InferredSchema]
        )
