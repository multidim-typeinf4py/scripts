import dataclasses
import json
import pathlib
import pickle
import typing

import numpy as np
import pandera.typing as pt
import torch
import tqdm
from annoy import AnnoyIndex
from gensim.models import Word2Vec
from libcst import codemod
from libsa4py.cst_extractor import Extractor
from type4py.__main__ import data_loading_comb as data_loading_funcs
from type4py.data_loaders import (
    load_test_data_per_model,
)
from type4py.deploy.infer import get_dps_single_file
from type4py.predict import compute_type_embed_batch

from common.schemas import InferredSchema
from ._base import PerFileInference


@dataclasses.dataclass
class FileDatapoints:
    all_type_slots: list
    vars_type_hints: list
    param_type_hints: list
    rets_type_hints: list


class PTType4Py:
    def __init__(self, pre_trained_model_path: pathlib.Path, topn: int):
        self.model_path = pre_trained_model_path

        self.type4py_model = torch.load(
            self.model_path / f"type4py_complete_model.pt",
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.type4py_model_params = json.load(
            (self.model_path / "model_params.json").open()
        )

        self.w2v_model = Word2Vec.load(
            fname=str(self.model_path / "w2v_token_model.bin")
        )
        self.type_clusters_idx = AnnoyIndex(
            self.type4py_model_params["output_size_prod"], "euclidean"
        )
        self.type_clusters_labels = np.load(
            str(self.model_path / f"type4py_complete_true.npy")
        )
        self.label_enc = pickle.load(
            (self.model_path / "label_encoder_all.pkl").open("rb")
        )

        self.topn = topn
        self.vths = None

    def load_pretrained_model(self):
        ...


class _Type4Py(PerFileInference):
    @property
    def method(self) -> str:
        return f"type4pyN{self.topn}"

    def __init__(
        self,
        mutable: pathlib.Path,
        readonly: pathlib.Path,
        cache: typing.Optional[pathlib.Path],
        model_path: pathlib.Path,
        topn: int,
    ):
        super().__init__(mutable, readonly, cache)
        self.topn = topn
        self.pretrained = PTType4Py(model_path, topn=topn)
        self.paths2datapoints = None

    def infer(self) -> None:
        self.paths2datapoints = self._create_or_load_datapoints(self.mutable)
        super().infer()

    def _infer_file(self, relative: pathlib.Path) -> pt.DataFrame[InferredSchema]:
        assert self.paths2datapoints is not None

    def _create_or_load_datapoints(
        self, project: pathlib.Path
    ) -> dict[pathlib.Path, FileDatapoints]:
        datapoints = self._load_cache()

        proj_files = set(
            map(
                lambda r: pathlib.Path(r).relative_to(project),
                codemod.gather_files([str(self.mutable / project)]),
            )
        )
        missing_files = sorted(proj_files - datapoints.keys())
        for file in (pbar := tqdm.tqdm(missing_files)):
            pbar.set_description(desc=f"Computing datapoints for {file}")
            filepath = project / file

            try:
                with filepath.open() as f:
                    src_f_read = f.read()
                type_hints = Extractor.extract(
                    src_f_read, include_seq2seq=False
                ).to_dict()

                (
                    all_type_slots,
                    vars_type_hints,
                    params_type_hints,
                    rets_type_hints,
                ) = get_dps_single_file(type_hints)

                datapoints[file] = dp = FileDatapoints(
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
    def __init__(
        self,
        model_path: pathlib.Path,
        mutable: pathlib.Path,
        readonly: pathlib.Path,
        cache: typing.Optional[pathlib.Path],
    ):
        super().__init__(mutable, readonly, cache, model_path, topn=1)


class Type4PyN3(_Type4Py):
    def __init__(
        self,
        model_path: pathlib.Path,
        mutable: pathlib.Path,
        readonly: pathlib.Path,
        cache: typing.Optional[pathlib.Path],
    ):
        super().__init__(mutable, readonly, cache, model_path, topn=3)


class Type4PyN5(_Type4Py):
    def __init__(
        self,
        model_path: pathlib.Path,
        mutable: pathlib.Path,
        readonly: pathlib.Path,
        cache: typing.Optional[pathlib.Path],
    ):
        super().__init__(mutable, readonly, cache, model_path, topn=5)


class Type4PyN10(_Type4Py):
    def __init__(
        self,
        model_path: pathlib.Path,
        mutable: pathlib.Path,
        readonly: pathlib.Path,
        cache: typing.Optional[pathlib.Path],
    ):
        super().__init__(mutable, readonly, cache, model_path, topn=10)
