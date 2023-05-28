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
    RolloutPredictionTopN,
    SignatureMap,
    SignatureMapTopN,
)
from typet5.train import (
    TrainingConfig,
    DecodingArgs,
)
from typet5.static_analysis import (
    FunctionSignature,
    VariableSignature,
    ProjectPath,
    PythonProject,
)

import libcst
from libcst import codemod, helpers as h, metadata, matchers as m
import pandera.typing as pt

from common.schemas import InferredSchema
from infer.inference._base import ProjectWideInference
from symbols.collector import build_type_collection
import utils


class TypeT5Applier(codemod.ContextAwareTransformer):
    METADATA_DEPENDENCIES = (metadata.QualifiedNameProvider, metadata.ScopeProvider, )

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
        if not prediction:
            return original_node

        assert isinstance(prediction, FunctionSignature), f"{type(prediction)=}"
        return prediction.apply(updated_node) if prediction is not None else updated_node

    def leave_Assign(
        self, original_node: libcst.Assign, updated_node: libcst.Assign
    ) -> libcst.Assign | libcst.AnnAssign:
        if (scope := self.get_metadata(metadata.ScopeProvider, original_node)) is None:
            return original_node

        if not isinstance(scope, metadata.GlobalScope | metadata.ClassScope):
            return original_node

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
        return libcst.AnnAssign(target=t, annotation=prediction.annot, value=original_node.value)


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


class _TypeT5(ProjectWideInference):
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
                project,
                pre_args=PreprocessArgs(),
                decode_order=DecodingOrders.DoubleTraversal(),
                num_return_sequences=self.topn,
            )
        )

        pprint.pprint(rollout.final_sigmap)

        collections = []
        for topn, batch in enumerate(_batchify(rollout.final_sigmap, self.topn), start=1):
            with utils.scratchpad(mutable) as sc:
                res = codemod.parallel_exec_transform_with_prettyprint(
                    transform=TypeT5Applier(
                        context=codemod.CodemodContext(),
                        predictions=batch,
                    ),
                    jobs=utils.worker_count(),
                    repo_root=str(sc),
                    files=[str(sc / f) for f in subset],
                )
                self.logger.info(
                    utils.format_parallel_exec_result(
                        f"Annotated with TypeT5 @ topn={topn}", result=res
                    )
                )

                collected = build_type_collection(root=sc, allow_stubs=False, subset=subset).df
                collections.append(collected.assign(topn=topn))
        return (
            pd.concat(collections, ignore_index=True)
            .assign(method=self.method())
            .pipe(pt.DataFrame[InferredSchema])
        )


def _batchify(predictions: SignatureMapTopN, maxn: int) -> Generator[SignatureMap, None, None]:
    for n in range(maxn):
        yield SignatureMap({
            project_path: signatures[n]
            for project_path, signatures in predictions.items()
        })


class TypeT5Top1(_TypeT5):
    def __init__(self) -> None:
        super().__init__(topn=1)


class TypeT5Top3(_TypeT5):
    def __init__(self) -> None:
        super().__init__(topn=3)


class TypeT5Top5(_TypeT5):
    def __init__(self) -> None:
        super().__init__(topn=5)


class TypeT5Top10(_TypeT5):
    def __init__(self) -> None:
        super().__init__(topn=10)