import dataclasses
import itertools
import json
import pathlib
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

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
    type_annotate_file,
)

from scripts.common.schemas import InferredSchema, TypeCollectionCategory
from ._base import ParallelisableInference
from ._utils import wrapped_partial

from scripts.infer.preprocessers import t4py
from ..annotators.t4py import Type4PyProjectApplier

from libcst import codemod
from libsa4py import cst_transformers


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
                "CUDAExecutionProvider"
                if torch.cuda.is_available()
                else "CPUExecutionProvider"
            ],
        )
        self.type4py_model_params = json.load(
            (self.model_path / "model_params.json").open()
        )
        self.type4py_model_params["k"] = topn

        self.w2v_model = Word2Vec.load(
            fname=str(self.model_path / "w2v_token_model.bin")
        )
        self.type_clusters_idx = AnnoyIndex(
            self.type4py_model_params["output_size"], "euclidean"
        )
        self.type_clusters_idx.load(
            str(self.model_path / "type4py_complete_type_cluster")
        )
        self.type_clusters_labels = np.load(
            str(self.model_path / f"type4py_complete_true.npy")
        )
        self.label_enc = pickle.load(
            (self.model_path / "label_encoder_all.pkl").open("rb")
        )

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
                "ret_type": (
                    read_or_null(fn["ret_type_p"], n) if "ret_type_p" in fn else ""
                ),
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


class _Type4Py(ParallelisableInference):
    def __init__(
        self,
        model_path: pathlib.Path,
        topn: int,
        cpu_executor: ProcessPoolExecutor | None = None,
        model_executor: ThreadPoolExecutor | None = None,
    ) -> None:
        super().__init__(cpu_executor=cpu_executor, model_executor=model_executor)

        self.topn = topn
        self.pretrained = PTType4Py(model_path, topn=topn)

    def method(self) -> str:
        return f"type4pyN{self.topn}"

    def preprocessor(self, task: TypeCollectionCategory) -> codemod.Codemod:
        return t4py.Type4PyPreprocessor(context=codemod.CodemodContext(), task=task)

    def _infer_project(
        self, mutable: pathlib.Path, subset: set[pathlib.Path]
    ) -> pt.DataFrame[InferredSchema]:
        # self.logger.info("Extracting datapoints...")
        # paths2datapoints = self.extract_datapoints(mutable, subset)
        # self.logger.debug(paths2datapoints)

        self.logger.info("Extracting Datapoints and Executing model...")
        paths2predictions = {
            path: self.type_annotate_file(
                self.pretrained,
                source_file_path=mutable / path,
                filter_pred_types=True,
            )
            for path in subset
        }
        # paths2predictions = self.make_predictions(paths2datapoints, subset)
        # self.logger.debug(paths2predictions)

        self.logger.info("Converting predictions into Top-N batches")
        paths2batches = self.make_topn_batches(paths2predictions, subset)
        self.logger.debug(paths2batches)

        return Type4PyProjectApplier.collect_topn(
            project=mutable,
            subset=subset,
            predictions=paths2batches,
            topn=self.topn,
            tool=self,
        )

    def type_annotate_file(self, pre_trained_m: PTType4Py, source_file_path: pathlib.Path,
                       filter_pred_types:bool=True) -> dict:
        src_f_read = source_file_path.read_text()
        ext_type_hints = Extractor.extract(src_f_read, include_seq2seq=False).to_dict()
        self.logger.info(f"Extracted JSON-representation of {source_file_path}")

        all_type_slots, vars_type_hints, params_type_hints, rets_type_hints = get_dps_single_file(ext_type_hints)
        self.logger.info("Extracted type hints from JSON")

        if not all_type_slots:
            self.logger.warn(f"No type slots detected in {source_file_path}; no predictions can be made")
            return dict()

        ext_type_hints = get_type_preds_single_file(ext_type_hints, all_type_slots,
                                                    (vars_type_hints, params_type_hints, rets_type_hints),
                                                    pre_trained_m, filter_pred_types)
        self.logger.info("Predicted type annotations for the given file")


        return ext_type_hints

    def extract_datapoints(
        self, mutable: pathlib.Path, subset: set[pathlib.Path]
    ) -> dict[pathlib.Path, FileDatapoints]:
        tasks = self.cpu_executor.map(
            _file2datapoint, itertools.repeat(mutable), subset
        )
        paths2datapoints = dict(
            tqdm.tqdm(tasks, total=len(subset), desc="Datapoint Extraction")
        )

        return paths2datapoints

    def make_predictions(
        self,
        paths2datapoints: dict[pathlib.Path, FileDatapoints],
        subset: set[pathlib.Path],
    ) -> dict[pathlib.Path, dict]:
        # Type prediction
        tasks = self.model_executor.map(
            self._infer_from_datapoints,
            itertools.repeat(paths2datapoints),
            subset,
        )
        paths2predictions = dict(
            tqdm.tqdm(tasks, total=len(subset), desc="Type Prediction")
        )
        return paths2predictions

    def make_topn_batches(
        self, paths2predictions: dict[pathlib.Path, dict], subset: set[pathlib.Path]
    ) -> dict[pathlib.Path, list[dict]]:  # Batchification
        tasks = self.cpu_executor.map(
            _batchify,
            itertools.repeat(paths2predictions),
            subset,
            itertools.repeat(self.topn),
        )
        paths2batches = dict(tqdm.tqdm(tasks, total=len(subset), desc="Top-N Batching"))
        return paths2batches

    def _infer_from_datapoints(
        self,
        datapoints: dict[pathlib.Path, FileDatapoints],
        file: pathlib.Path,
    ) -> tuple[pathlib.Path, dict]:
        datapoint = datapoints[file]

        # Filter out files for which no predictions were made
        has_type_slots = any(
            dp_hint
            for dp_hint in (
                datapoint.vars_type_hints,
                datapoint.param_type_hints,
                datapoint.rets_type_hints,
            )
        )
        if not has_type_slots:
            return file, dict()

        return file, get_type_preds_single_file(
            datapoint.ext_type_hints,
            datapoint.all_type_slots,
            (
                datapoint.vars_type_hints,
                datapoint.param_type_hints,
                datapoint.rets_type_hints,
            ),
            self.pretrained,
            filter_pred_types=True,
        )


def _file2datapoint(
    project: pathlib.Path,
    file: pathlib.Path,
) -> tuple[pathlib.Path, FileDatapoints]:
    filepath = project / file

    source_code = filepath.read_text()
    ext_type_hints = Extractor.extract(source_code, include_seq2seq=False).to_dict()

    (
        all_type_slots,
        vars_type_hints,
        params_type_hints,
        rets_type_hints,
    ) = get_dps_single_file(ext_type_hints)

    return file, FileDatapoints(
        ext_type_hints=ext_type_hints,
        all_type_slots=all_type_slots,
        vars_type_hints=vars_type_hints,
        param_type_hints=params_type_hints,
        rets_type_hints=rets_type_hints,
    )


class Type4PyTopN(_Type4Py):
    def __init__(
        self,
        topn: int,
        cpu_executor: ProcessPoolExecutor | None = None,
        model_executor: ThreadPoolExecutor | None = None,
    ):
        super().__init__(
            model_path=pathlib.Path("models/type4py"),
            topn=topn,
            cpu_executor=cpu_executor,
            model_executor=model_executor,
        )


Type4PyTop1 = wrapped_partial(Type4PyTopN, topn=1)
Type4PyTop3 = wrapped_partial(Type4PyTopN, topn=3)
Type4PyTop5 = wrapped_partial(Type4PyTopN, topn=5)
Type4PyTop10 = wrapped_partial(Type4PyTopN, topn=10)
