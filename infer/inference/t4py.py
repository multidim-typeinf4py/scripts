import collections
import dataclasses
import json
import pathlib
import pickle
import typing

import libcst
import numpy as np
import pandas as pd
import onnxruntime
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

from common.schemas import InferredSchema
from symbols.collector import build_type_collection
from ._base import PerFileInference


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


def _batchify(predictions: dict, topn: int) -> typing.Generator[dict, None, None]:

    # Transfer variables_p into variables, same with params_p and ret_type_p

    for n in range(topn):
        for m_v, _ in predictions["variables"].items():
            # The predictions for module-level vars

            if len(predictions["variables_p"][m_v]) <= n:
                t = ""
            else:
                t, _ = predictions["variables_p"][m_v][n]
            predictions["variables"][m_v] = t

            for i, fn in enumerate(predictions["funcs"]):
                for p_n, _ in fn["params"].items():
                    # The predictions for arguments for module-level functions
                    if len(predictions["funcs"][i]["params_p"][p_n]) <= n:
                        t = ""
                    else:
                        t, _ = predictions["funcs"][i]["params_p"][p_n][n]
                    predictions["funcs"][i]["params"][p_n] = t

                # The predictions local variables for module-level functions
                for fn_v, _ in fn["variables"].items():
                    if len(predictions["funcs"][i]["variables_p"][fn_v]) <= n:
                        t = ""
                    else:
                        t, _ = predictions["funcs"][i]["variables_p"][fn_v][n]
                    predictions["funcs"][i]["variables"][fn_v] = t

                # The return type for module-level functions
                if predictions["funcs"][i]["ret_exprs"] != []:
                    if len(predictions["funcs"][i]["ret_type_p"]) <= n:
                        t = ""
                    else:
                        t, _ = predictions["funcs"][i]["ret_type_p"][n]
                    predictions["funcs"][i]["ret_type"] = t

            # The type of class-level vars
            for c_i, c in enumerate(predictions["classes"]):
                for c_v, _ in c["variables"].items():
                    if len(c["variables_p"][c_v]) <= n:
                        t = ""
                    else:
                        t, _ = c["variables_p"][c_v][n]
                    predictions["classes"][c_i]["variables"][c_v] = t

                # The type of arguments for class-level functions
                for fn_i, fn in enumerate(c["funcs"]):
                    for p_n, _ in fn["params"].items():
                        if len(fn["params_p"][p_n]) <= n:
                            t = ""
                        else:
                            t, _ = fn["params_p"][p_n][n]
                        predictions["classes"][c_i]["funcs"][fn_i]["params"][p_n] = t

                    # The type of local variables for class-level functions
                    for fn_v, _ in fn["variables"].items():
                        if len(fn["variables_p"][fn_v]) <= n:
                            t = ""
                        else:
                            t, _ = fn["variables_p"][fn_v][n]
                        predictions["classes"][c_i]["funcs"][fn_i]["variables"][fn_v] = t

                    # The return type for class-level functions
                    if predictions["classes"][c_i]["funcs"][fn_i]["ret_exprs"] != []:
                        if len(predictions["classes"][c_i]["funcs"][fn_i]["ret_type_p"]) <= n:
                            t = ""
                        else:
                            t, _ = predictions["classes"][c_i]["funcs"][fn_i]["ret_type_p"][n]
                        predictions["classes"][c_i]["funcs"][fn_i]["ret_type"] = t
        yield predictions


class _Type4Py(PerFileInference):
    @property
    def method(self) -> str:
        return f"type4pyN{self.topn}"

    def __init__(
        self,
        cache: typing.Optional[pathlib.Path],
        model_path: pathlib.Path,
        topn: int,
    ):
        super().__init__(cache)

        self.topn = topn
        self.pretrained = PTType4Py(model_path, topn=topn)
        self.paths2datapoints = None

    def infer(
        self,
        mutable: pathlib.Path,
        readonly: pathlib.Path,
        subset: typing.Optional[set[pathlib.Path]] = None,
    ) -> None:
        self.paths2datapoints = self._create_or_load_datapoints(mutable, subset)
        super().infer(mutable=mutable, readonly=readonly, subset=subset)

    def _infer_file(
        self,
        root: pathlib.Path,
        relative: pathlib.Path,
    ) -> pt.DataFrame[InferredSchema]:
        assert self.paths2datapoints is not None
        if (dps := self.paths2datapoints.get(relative)) is None or not any(
            dp_hint
            for dp_hint in (
                dps.vars_type_hints,
                dps.param_type_hints,
                dps.rets_type_hints,
            )
        ):
            return InferredSchema.example(size=0)

        dps.ext_type_hints = get_type_preds_single_file(
            dps.ext_type_hints,
            dps.all_type_slots,
            (dps.vars_type_hints, dps.param_type_hints, dps.rets_type_hints),
            self.pretrained,
            filter_pred_types=False,
        )

        with (root / relative).open() as f:
            original = libcst.parse_module(f.read())

        rm = metadata.FullRepoManager(repo_root_dir=str(root), paths=[str(relative)], providers={})

        agnostic_preds = []
        for topn, predictions in enumerate(_batchify(dps.ext_type_hints, topn=self.topn), start=1):
            annotated = rm.get_metadata_wrapper_for_path(str(relative)).visit(
                TypeApplier(predictions, apply_nlp=False)
            )

            with (root / relative).open("w") as f:
                f.write(annotated.code)

            agnostic_preds.append(
                build_type_collection(root=root, allow_stubs=False, subset={relative}).df.assign(
                    method=self.method, topn=topn
                )
            )

            with (root / relative).open("w") as f:
                f.write(original.code)

        return pd.concat(agnostic_preds, ignore_index=True).pipe(pt.DataFrame[InferredSchema])

    def _create_or_load_datapoints(
        self, project: pathlib.Path, subset: typing.Optional[set[pathlib.Path]]
    ) -> dict[pathlib.Path, FileDatapoints]:
        datapoints = self._load_cache()

        if subset is None:
            proj_files = set(
                map(
                    lambda r: pathlib.Path(r).relative_to(project),
                    codemod.gather_files([str(project)]),
                )
            )
        else:
            proj_files = subset
        missing_files = sorted(proj_files - datapoints.keys())
        for file in (pbar := tqdm.tqdm(missing_files)):
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

                datapoints[file] = dp = FileDatapoints(
                    ext_type_hints=type_hints,
                    all_type_slots=all_type_slots,
                    vars_type_hints=vars_type_hints,
                    param_type_hints=params_type_hints,
                    rets_type_hints=rets_type_hints,
                )
                self.register_cache(file, dp)

            except (
                UnicodeDecodeError,
                FileNotFoundError,
                IsADirectoryError,
                SyntaxError,
            ) as e:
                print(f"Skipping {file} - {e}")

        return datapoints


class Type4PyN1(_Type4Py):
    def __init__(self, cache: typing.Optional[pathlib.Path], model_path: pathlib.Path):
        super().__init__(cache, model_path, topn=1)


class Type4PyN3(_Type4Py):
    def __init__(self, cache: typing.Optional[pathlib.Path], model_path: pathlib.Path):
        super().__init__(cache, model_path, topn=3)


class Type4PyN5(_Type4Py):
    def __init__(self, cache: typing.Optional[pathlib.Path], model_path: pathlib.Path):
        super().__init__(cache, model_path, topn=5)


class Type4PyN10(_Type4Py):
    def __init__(self, cache: typing.Optional[pathlib.Path], model_path: pathlib.Path):
        super().__init__(cache, model_path, topn=10)
