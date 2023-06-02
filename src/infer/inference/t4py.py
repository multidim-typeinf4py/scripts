import dataclasses
import json
import pathlib
import pickle
import typing

import libcst
import numpy as np
import onnxruntime
import pandas as pd
import pandera.typing as pt
import torch
import tqdm
from annoy import AnnoyIndex
from gensim.models import Word2Vec
from libcst import codemod
from libcst import metadata
from libsa4py.cst_extractor import Extractor
from libsa4py.cst_transformers import TypeApplier
from type4py.deploy.infer import (
    get_dps_single_file,
    get_type_preds_single_file,
)

import utils
from src.common.schemas import InferredSchema
from src.symbols.collector import build_type_collection
from ._base import ProjectWideInference


@dataclasses.dataclass
class FileDatapoints:
    ext_type_hints: dict
    all_type_slots: list
    vars_type_hints: list
    param_type_hints: list
    rets_type_hints: list


class PTType4Py:
    def __init__(self, pre_trained_model_path: pathlib.Path, topn: int):
        self.model_path = pre_trained_model_path

        self.type4py_model = onnxruntime.InferenceSession(
            str(self.model_path / f"type4py_complete_model.onnx"),
            providers=[
                "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
            ],
        )
        self.type4py_model_params = json.load((self.model_path / "model_params.json").open())
        self.type4py_model_params["k"] = topn

        self.w2v_model = Word2Vec.load(fname=str(self.model_path / "w2v_token_model.bin"))
        self.type_clusters_idx = AnnoyIndex(self.type4py_model_params["output_size"], "euclidean")
        self.type_clusters_idx.load(str(self.model_path / "type4py_complete_type_cluster"))
        self.type_clusters_labels = np.load(str(self.model_path / f"type4py_complete_true.npy"))
        self.label_enc = pickle.load((self.model_path / "label_encoder_all.pkl").open("rb"))

        self.topn = topn
        self.vths = None

        self.use_pca = False

    def load_pretrained_model(self):
        ...


def _batchify(predictions: dict, topn: int) -> list[dict]:
    def read_or_null(l: list, n: int) -> str:
        if n < len(l):
            return l[n][0]
        return ""

    def variables_read(d: dict, n: int) -> dict:
        return {v: read_or_null(d["variables_p"][v], n) for v in d["variables"]}

    def funcs_read(d: dict, n: int) -> list:
        return [
            {
                "name": fn["name"],
                "q_name": fn["q_name"],
                "params": {p: read_or_null(fn["params_p"][p], n) for p in fn["params"]},
                "ret_type": (read_or_null(fn["ret_type_p"], n) if "ret_type_p" in fn else ""),
                "variables": variables_read(fn, n),
            }
            for fn in d["funcs"]
        ]

    def classes_read(d: dict, n: int) -> list:
        return [
            {
                "name": clazz["name"],
                "q_name": clazz["q_name"],
                "funcs": funcs_read(clazz, n),
                "variables": variables_read(clazz, n),
            }
            for clazz in d["classes"]
        ]

    batches: list[dict] = []

    # Transfer variables_p into variables, same with params_p and ret_type_p
    for n in range(topn):
        batches.append(
            {
                "variables": variables_read(predictions, n),
                "funcs": funcs_read(predictions, n),
                "classes": classes_read(predictions, n),
            }
        )

    return batches


class ParallelTypeApplier(codemod.ContextAwareTransformer):
    def __init__(
        self,
        context: codemod.CodemodContext,
        path2batches: dict[pathlib.Path, list[dict]],
        topn: int,
    ) -> None:
        super().__init__(context)
        self.path2batches = path2batches
        self.topn = topn

    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        assert self.context.filename is not None
        assert self.context.metadata_manager is not None

        path = pathlib.Path(self.context.filename).relative_to(
            self.context.metadata_manager.root_path
        )

        batch = self.path2batches[path][self.topn]

        return metadata.MetadataWrapper(
            module=tree,
            unsafe_skip_copy=True,
            cache=self.context.metadata_manager.get_cache_for_path(path=self.context.filename),
        ).visit(TypeApplier(f_processeed_dict=batch, apply_nlp=False))


class _Type4Py(ProjectWideInference):
    def __init__(
        self,
        model_path: pathlib.Path,
        topn: int,
    ):
        super().__init__()

        self.topn = topn
        self.pretrained = PTType4Py(model_path, topn=topn)

    def method(self) -> str:
        return f"type4pyN{self.topn}"

    def _infer_project(
        self, mutable: pathlib.Path, subset: set[pathlib.Path]
    ) -> pt.DataFrame[InferredSchema]:
        proj_files = subset

        paths2datapoints = self._create_or_load_datapoints(mutable, proj_files)
        paths_with_predictions = {
            p: dp
            for p, dp in paths2datapoints.items()
            if dp is not None
            and any(
                dp_hint
                for dp_hint in (
                    dp.vars_type_hints,
                    dp.param_type_hints,
                    dp.rets_type_hints,
                )
            )
        }

        paths2batches = {
            p: _batchify(
                get_type_preds_single_file(
                    dps.ext_type_hints,
                    dps.all_type_slots,
                    (dps.vars_type_hints, dps.param_type_hints, dps.rets_type_hints),
                    self.pretrained,
                    filter_pred_types=False,
                ),
                topn=self.topn,
            )
            for p, dps in paths_with_predictions.items()
        }

        collections = []
        for topn in range(1, self.topn + 1):
            with utils.scratchpad(mutable) as sc:
                t4p_hint_res = codemod.parallel_exec_transform_with_prettyprint(
                    transform=ParallelTypeApplier(
                        context=codemod.CodemodContext(),
                        path2batches=paths2batches,
                        topn=topn - 1,
                    ),
                    jobs=utils.worker_count(),
                    repo_root=str(sc),
                    files=[str(sc / p) for p in subset],
                )
                self.logger.info(
                    utils.format_parallel_exec_result(
                        f"Annotated with Type4Py @ topn={topn}", result=t4p_hint_res
                    )
                )
                collections.append(
                    build_type_collection(
                        root=sc, allow_stubs=False, subset=set(proj_files)
                    ).df.assign(topn=topn)
                )

        return (
            pd.concat(collections, ignore_index=True)
            .assign(method=self.method())
            .pipe(pt.DataFrame[InferredSchema])
        )

    def _create_or_load_datapoints(
        self, project: pathlib.Path, proj_files: set[pathlib.Path]
    ) -> dict[pathlib.Path, FileDatapoints]:
        datapoints = dict[pathlib.Path, FileDatapoints]()

        for file in (pbar := tqdm.tqdm(proj_files)):
            pbar.set_description(desc=f"Computing datapoints for {file}")
            filepath = project / file

            try:
                with filepath.open() as f:
                    src_f_read = f.read()
                type_hints = Extractor.extract(src_f_read, include_seq2seq=False).to_dict()

                (
                    all_type_slots,
                    vars_type_hints,
                    params_type_hints,
                    rets_type_hints,
                ) = get_dps_single_file(type_hints)

                datapoints[file] = FileDatapoints(
                    ext_type_hints=type_hints,
                    all_type_slots=all_type_slots,
                    vars_type_hints=vars_type_hints,
                    param_type_hints=params_type_hints,
                    rets_type_hints=rets_type_hints,
                )

            except (
                UnicodeDecodeError,
                FileNotFoundError,
                IsADirectoryError,
                SyntaxError,
            ) as e:
                self.logger.warning(f"Skipping {file} during datapoint calculation - {e}")

        return datapoints


class _Type4PyTopN(_Type4Py):
    def __init__(self, topn: int):
        super().__init__(model_path=pathlib.Path.cwd() / "models" / "type4py", topn=topn)


class Type4PyTop1(_Type4PyTopN):
    def __init__(self):
        super().__init__(topn=1)


class Type4PyTop3(_Type4PyTopN):
    def __init__(self):
        super().__init__(topn=3)


class Type4PyTop5(_Type4PyTopN):
    def __init__(self):
        super().__init__(topn=5)


class Type4PyTop10(_Type4PyTopN):
    def __init__(self):
        super().__init__(topn=10)
