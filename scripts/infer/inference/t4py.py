import dataclasses
import functools
import itertools
import json
import pathlib
import pickle

import numpy as np
import onnxruntime
import pandera.typing as pt
import torch
import tqdm
from annoy import AnnoyIndex
from gensim.models import Word2Vec
from libsa4py.cst_extractor import Extractor
from type4py.deploy.infer import (
    get_dps_single_file,
    get_type_preds_single_file,
)

from scripts.common.schemas import InferredSchema
from ._base import ProjectWideInference
from ..annotators.type4py import Type4PyProjectApplier


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


def _batchify(
    predictions: dict[pathlib.Path, dict], path: pathlib.Path, topn: int
) -> tuple[pathlib.Path, list[dict]]:
    def read_or_null(l: list, n: int) -> str:
        if n < len(l):
            return l[n][0]
        return ""

    def variables_read(d: dict, n: int) -> dict:
        return {v: read_or_null(d["variables_p"][v], n) for v in d.get("variables", [])}

    def funcs_read(d: dict, n: int) -> list:
        return [
            {
                "name": fn["name"],
                "q_name": fn["q_name"],
                "params": {p: read_or_null(fn["params_p"][p], n) for p in fn["params"]},
                "ret_type": (read_or_null(fn["ret_type_p"], n) if "ret_type_p" in fn else ""),
                "variables": variables_read(fn, n),
            }
            for fn in d.get("funcs", [])
        ]

    def classes_read(d: dict, n: int) -> list:
        return [
            {
                "name": clazz["name"],
                "q_name": clazz["q_name"],
                "funcs": funcs_read(clazz, n),
                "variables": variables_read(clazz, n),
            }
            for clazz in d.get("classes", [])
        ]

    predictions = predictions[path]
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

    return path, batches


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
        # Datapoint collection
        self.logger.info("Extracting datapoints...")
        with self.cpu_executor() as executor:
            tasks = executor.map(_file2datapoint, itertools.repeat(mutable), subset)
            paths2datapoints = dict(
                tqdm.tqdm(tasks, total=len(subset), desc="Datapoint Extraction")
            )
        self.logger.debug(paths2datapoints)

        # Type prediction
        self.logger.info("Executing model...")
        with self.model_executor() as executor:
            tasks = executor.map(
                self._infer_from_datapoints,
                itertools.repeat(paths2datapoints),
                subset,
            )
            paths2predictions = dict(tqdm.tqdm(tasks, total=len(subset), desc="Type Prediction"))
        self.logger.debug(paths2predictions)

        # Batchification
        self.logger.info("Converting predictions into Top-N batches")
        with self.cpu_executor() as executor:
            tasks = executor.map(
                _batchify, itertools.repeat(paths2predictions), subset, itertools.repeat(self.topn)
            )
            paths2batches = dict(tqdm.tqdm(tasks, total=len(subset), desc="Top-N Batching"))
        self.logger.debug(paths2batches)

        return Type4PyProjectApplier.collect_topn(
            project=mutable,
            subset=subset,
            predictions=paths2batches,
            topn=self.topn,
            tool=self,
        )

    def _infer_from_datapoints(
        self,
        datapoints: dict[pathlib.Path, FileDatapoints],
        file: pathlib.Path,
    ) -> tuple[pathlib.Path, dict]:
        datapoints = datapoints[file]

        # Filter out files for which no predictions were made
        has_type_slots = any(
            dp_hint
            for dp_hint in (
                datapoints.vars_type_hints,
                datapoints.param_type_hints,
                datapoints.rets_type_hints,
            )
        )
        if not has_type_slots:
            return file, dict()

        return file, get_type_preds_single_file(
            datapoints.ext_type_hints,
            datapoints.all_type_slots,
            (datapoints.vars_type_hints, datapoints.param_type_hints, datapoints.rets_type_hints),
            self.pretrained,
            filter_pred_types=False,
        )


def _file2datapoint(
    project: pathlib.Path,
    file: pathlib.Path,
) -> tuple[pathlib.Path, FileDatapoints]:
    filepath = project / file
    type_hints = Extractor.extract(filepath.read_text(), include_seq2seq=False).to_dict()

    (
        all_type_slots,
        vars_type_hints,
        params_type_hints,
        rets_type_hints,
    ) = get_dps_single_file(type_hints)

    return file, FileDatapoints(
        ext_type_hints=type_hints,
        all_type_slots=all_type_slots,
        vars_type_hints=vars_type_hints,
        param_type_hints=params_type_hints,
        rets_type_hints=rets_type_hints,
    )


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
