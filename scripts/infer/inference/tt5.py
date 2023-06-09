import pathlib

import libcst
import pandera.typing as pt
import torch
from libcst import codemod
from typet5.experiments import utils as typet5_utils
from typet5.function_decoding import (
    RolloutCtx,
    PreprocessArgs,
    DecodingOrders,
    RolloutPredictionTopN,
    SignatureMap,
)
from typet5.model import ModelWrapper
from typet5.static_analysis import (
    PythonProject,
)
from typet5.train import (
    TrainingConfig,
    DecodingArgs,
)
from typet5.utils import *

from scripts import utils
from scripts.common.schemas import InferredSchema
from scripts.infer.annotators import TT5ProjectApplier
from scripts.infer.inference._base import ProjectWideInference
from scripts.infer.inference._utils import wrapped_partial


class TypeT5Applier(codemod.ContextAwareTransformer):
    def __init__(self, context: codemod.CodemodContext, predictions: SignatureMap) -> None:
        super().__init__(context)
        self.predictions = predictions

    def leave_Module(
        self, original_node: libcst.Module, updated_node: libcst.Module
    ) -> libcst.Module:
        return typet5_utils.apply_sigmap(
            m=original_node,
            sigmap=self.predictions,
            module_name=self.context.full_module_name,
        )


class TypeT5Configs:
    Default = TrainingConfig(
        func_only=True,
        pre_args=PreprocessArgs(
            drop_env_types=False,
            add_implicit_rel_imports=True,
        ),
        left_margin=2048,
        right_margin=2048 - 512,
        preamble_size=1000,
    )


class TypeT5TopN(ProjectWideInference):
    def __init__(self, topn: int) -> None:
        super().__init__()
        self.wrapper = ModelWrapper.load_from_hub("MrVPlusOne/TypeT5-v7")
        self.wrapper.args = DecodingArgs(
            sampling_max_tokens=TypeT5Configs.Default.ctx_size,
            ctx_args=TypeT5Configs.Default.dec_ctx_args(),
            do_sample=False,
            top_p=1.0,
            num_beams=16,
        )
        self.wrapper.to(torch.device(f"cuda" if torch.cuda.is_available() else "cpu"))

        self.topn = topn

    def method(self) -> str:
        return f"TypeT5TopN{self.topn}"

    def _infer_project(
        self, mutable: pathlib.Path, subset: set[pathlib.Path]
    ) -> pt.DataFrame[InferredSchema]:
        project = PythonProject.parse_from_root(root=mutable)
        rctx = RolloutCtx(model=self.wrapper)

        rollout: RolloutPredictionTopN = asyncio.run(
            rctx.run_on_project(
                project=project,
                pre_args=PreprocessArgs(),
                decode_order=DecodingOrders.DoubleTraversal(),
                concurrency=utils.worker_count(),
                num_return_sequences=self.topn,
            )
        )

        return TT5ProjectApplier.collect_topn(
            project=mutable,
            subset=subset,
            predictions=rollout.final_sigmap,
            topn=self.topn,
            tool=self,
        )


TypeT5Top1 = wrapped_partial(TypeT5TopN, topn=1)
TypeT5Top3 = wrapped_partial(TypeT5TopN, topn=3)
TypeT5Top5 = wrapped_partial(TypeT5TopN, topn=5)
TypeT5Top10 = wrapped_partial(TypeT5TopN, topn=10)
