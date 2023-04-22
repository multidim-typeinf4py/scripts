import json
import pathlib
import pickle
from typing import Optional

from annoy import AnnoyIndex
from type4py.predict import compute_type_embed_batch, predict_type_embed_task

import utils
from common.schemas import InferredSchema
from symbols.collector import TypeCollectorVisitor
from ._base import DatasetWideInference, DatasetFolderStructure

import libcst as cst
from libcst import codemod
import libcst.metadata as metadata

import torch
from torch.utils.data import DataLoader

from libsa4py.cst_transformers import TypeApplier
from libsa4py.cst_pipeline import Pipeline

import pandas as pd, numpy as np
import pandera.typing as pt
import pydantic

from type4py import predict as t4pypred
from type4py.data_loaders import (
    TripletDataset,
    load_training_data_per_model,
    load_test_data_per_model,
)
from type4py.__main__ import data_loading_comb as data_loading_funcs


class Type4Py(DatasetWideInference):
    method = "type4py"

    def __init__(
        self, model_path: pathlib.Path, dataset: pathlib.Path, topn: int
    ) -> None:
        super().__init__(dataset)
        self.topn = topn
        self.model_params = json.load((model_path / "model_params.json").open())

        self.model = torch.load(
            model_path / f"type4py_complete_model.pt",
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ).model
        self.label_encoder = pickle.load(
            (model_path / "label_encoder_all.pkl").open("rb")
        )

        self.type_clusters_idx = AnnoyIndex(self.model.output_size, "euclidean")
        self.type_clusters_idx.load(
            str(model_path / "type4py_complete_type_cluster"), prefault=False
        )
        self.type_cluster_labels = np.load(
            str(model_path / "type4py_complete_true.npy")
        )

    def _infer_dataset(
        self, structure: DatasetFolderStructure
    ) -> pt.DataFrame[InferredSchema]:
        self.logger.info("Loading test set")
        test_data_loader, test_task_indices = load_test_data_per_model(
            data_loading_funcs, str(self.dataset), self.model_params["batches_test"]
        )

        self.logger.info("Mapping test samples to type clusters")
        test_type_embed, embed_test_labels = compute_type_embed_batch(
            self.model, test_data_loader
        )

        self.logger.info("Performing KNN search")

        # train_valid_labels = self.label_encoder.inverse_transform(embed_labels)
        # embed_test_labels = self.label_encoder.inverse_transform(embed_test_labels)
        prediction_indices, distances = self.type_clusters_idx.get_nns_by_vector(
            test_type_embed, n=self.topn, include_distances=True
        )
        prediction_types = embed_test_labels[prediction_indices]

        print(list(zip(prediction_types, distances, strict=True)))
        return None
