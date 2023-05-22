import asyncio
import pathlib
import pprint
import torch

from typet5.model import ModelWrapper
from typet5.train import PreprocessArgs
from typet5.utils import *
from typet5.function_decoding import (
    RolloutCtx,
    PreprocessArgs,
    DecodingOrders,
    RolloutPrediction,
    SignatureMap,
)
from typet5.static_analysis import (
    FunctionSignature,
    VariableSignature,
    ProjectPath,
)

import libcst
from libcst.codemod.visitors._apply_type_annotations import (
    Annotations,
    FunctionKey,
    FunctionAnnotation,
)
from libcst import codemod, helpers as h, metadata, matchers as m
from typet5.static_analysis import PythonProject
import pandera.typing as pt

from common.schemas import InferredSchema
from infer.inference._base import ProjectWideInference
from symbols.collector import build_type_collection
import utils


class TypeT5Applier(codemod.ContextAwareTransformer):
    METADATA_DEPENDENCIES = (metadata.QualifiedNameProvider,)

    def __init__(self, context: codemod.CodemodContext, predictions: SignatureMap) -> None:
        super().__init__(context)
        self.predictions = predictions

    def leave_FunctionDef(
        self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef
    ) -> libcst.FunctionDef:
        qname = next(iter(self.get_metadata(metadata.QualifiedNameProvider, original_node)))
        prediction = self.predictions.get(
            ProjectPath(module=self.context.full_module_name, path=qname.name)
        )

        return prediction.apply(updated_node) if prediction is not None else updated_node

    def leave_Assign(
        self, original_node: libcst.Assign, updated_node: libcst.Assign
    ) -> libcst.Assign | libcst.AnnAssign:
        if len(original_node.targets) != 1 or not self.matches(
            original_node.targets[0], m.AssignTarget(m.Name() | m.Attribute(value=m.Name("self")))
        ):
            return original_node

        t = original_node.targets[0].target

        if self.matches(t, m.Attribute()):
            path = h.get_full_name_for_node_or_raise(t.attr)
        else:
            path = h.get_full_name_for_node_or_raise(t)
        prediction = self.predictions.get(
            ProjectPath(module=self.context.full_module_name, path=path)
        )
        if not prediction:
            return original_node

        assert isinstance(prediction, VariableSignature)
        return libcst.AnnAssign(target=t, annotation=prediction.annot)


class TypeT5(ProjectWideInference):
    def __init__(self) -> None:
        super().__init__()
        self.wrapper = ModelWrapper.load_from_hub("MrVPlusOne/TypeT5-v7")
        self.wrapper.to(torch.device(f"cuda" if torch.cuda.is_available() else "cpu"))

    def method(self) -> str:
        return "typet5"

    def _infer_project(
        self, mutable: pathlib.Path, subset: set[pathlib.Path]
    ) -> pt.DataFrame[InferredSchema]:
        project = PythonProject.parse_from_root(root=mutable)
        rctx = RolloutCtx(model=self.wrapper)

        rollout: RolloutPrediction = asyncio.run(
            rctx.run_on_project(
                project, pre_args=PreprocessArgs(), decode_order=DecodingOrders.DoubleTraversal()
            )
        )
        self.logger.debug(pprint.pprint(rollout.final_sigmap))

        res = codemod.parallel_exec_transform_with_prettyprint(
            transform=TypeT5Applier(
                context=codemod.CodemodContext(),
                predictions=rollout.final_sigmap,
            ),
            jobs=utils.worker_count(),
            repo_root=str(mutable),
            files=[str(mutable / f) for f in subset],
        )
        self.logger.info(
            utils.format_parallel_exec_result(f"Annotated with TypeT5 @ topn={1}", result=res)
        )

        return (
            build_type_collection(root=mutable, allow_stubs=False, subset=subset)
            .df.assign(method=self.method(), topn=1)
            .pipe(pt.DataFrame[InferredSchema])
        )
