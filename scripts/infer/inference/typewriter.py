from __future__ import annotations

import dataclasses
import functools
import itertools
import logging
import os
import pathlib
import pickle
import re
import tempfile
from ast import literal_eval
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from os.path import splitext, basename, join
from typing import List, no_type_check, Optional

import libcst
import libcst as cst
import tqdm
from libcst import metadata, codemod
import numpy as np
import pandas as pd
import pandera.typing as pt
import torch
from gensim.models import Word2Vec
from torch.utils.data import DataLoader, TensorDataset
from typewriter.dltpy.input_preparation.generate_df import encode_aval_types_TW
from typewriter.dltpy.input_preparation.generate_df import format_df
from typewriter.dltpy.preprocessing.extractor import Function, ParseError
from typewriter.dltpy.preprocessing.pipeline import extractor, read_file, preprocessor
from typewriter.extraction import (
    process_datapoints_TW,
    IdentifierSequence,
    TokenSequence,
    CommentSequence,
    gen_aval_types_datapoints,
)
from typewriter.model import load_data_tensors_TW, make_batch_prediction_TW
from typewriter.prepocessing import (
    filter_functions,
    gen_argument_df_TW,
)

from ..annotators.typewriter import Parameter, Return, TWProjectApplier
from scripts.common.schemas import InferredSchema
from scripts.symbols.collector import build_type_collection
from ._base import ProjectWideInference

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pd.set_option("display.max_columns", 20)


# Credits go to the original Author: Amir M. Mir (TU Delft)
@no_type_check
def process_py_src_file(src_file_path):
    """
    It extracts and process functions from a given Python source file

    :param src_file_path:
    :return:
    """
    try:
        functions, _ = extractor.extract(read_file(src_file_path))
        preprocessed_funcs = [preprocessor.preprocess(f) for f in functions]
        return preprocessed_funcs

    except (ParseError, UnicodeDecodeError):
        print(f"Could not parse file {src_file_path}")
        # sys.exit(1)


@no_type_check
def write_ext_funcs(ext_funcs: List[Function], src_file: str, output_dir: str):
    """
    Writes the extracted functions to a pandas Dataframe
    :param ext_funcs:
    :return:
    """

    funcs = []
    columns = None

    for f in ext_funcs:
        if columns is None:
            columns = ["file", "has_type"] + list(f.tuple_keys())

        funcs.append((src_file, f.has_types()) + f.as_tuple())

    if len(funcs) == 0:
        print("WARNING: no functions are extracted...")

    if columns is None:
        columns = [
            "file",
            "has_type",
            "name",
            "docstring",
            "func_descr",
            "arg_names",
            "arg_types",
            "arg_descrs",
            "return_type",
            "return_expr",
            "return_descr",
        ]

    funcs_df = pd.DataFrame(funcs, columns=columns)
    funcs_df["arg_names_len"] = funcs_df["arg_names"].apply(len)
    funcs_df["arg_types_len"] = funcs_df["arg_types"].apply(len)
    funcs_df.to_csv(
        join(output_dir, "ext_funcs_" + splitext(basename(src_file))[0] + ".csv"),
        index=False,
    )


@no_type_check
def filter_ret_funcs(ext_funcs_df: pd.DataFrame):
    """
    Filters out functions based on empty return expressions and return types of Any and None
    :param funcs_df:
    :return:
    """

    print(f"Functions before dropping nan, None, Any return type {len(ext_funcs_df)}")
    to_drop = np.invert(
        (ext_funcs_df["return_type"] == "nan")
        | (ext_funcs_df["return_type"] == "None")
        | (ext_funcs_df["return_type"] == "Any")
    )
    ext_funcs_df = ext_funcs_df[to_drop]
    print(f"Functions after dropping nan return type {len(ext_funcs_df)}")

    print(f"Functions before dropping on empty return expression {len(ext_funcs_df)}")
    ext_funcs_df = ext_funcs_df[
        ext_funcs_df["return_expr"].apply(lambda x: len(literal_eval(x))) > 0
    ]
    print(f"Functions after dropping on empty return expression {len(ext_funcs_df)}")

    return ext_funcs_df


@no_type_check
def load_param_data(vector_dir: str):
    """
    Loads the sequences of parameters from the disk
    :param vector_dir:
    :return:
    """
    return (
        load_data_tensors_TW(join(vector_dir, "identifiers_params_datapoints_x.npy")),
        load_data_tensors_TW(join(vector_dir, "tokens_params_datapoints_x.npy")),
        load_data_tensors_TW(join(vector_dir, "comments_params_datapoints_x.npy")),
        load_data_tensors_TW(join(vector_dir, "params__aval_types_dp.npy")),
    )


@no_type_check
def load_ret_data(vector_dir: str):
    """
    Loads the sequences of return types from the disk
    :param vector_dir:
    :return:
    """
    return (
        load_data_tensors_TW(join(vector_dir, "identifiers_ret_datapoints_x.npy")),
        load_data_tensors_TW(join(vector_dir, "tokens_ret_datapoints_x.npy")),
        load_data_tensors_TW(join(vector_dir, "comments_ret_datapoints_x.npy")),
        load_data_tensors_TW(join(vector_dir, "ret__aval_types_dp.npy")),
    )


@no_type_check
def evaluate_TW(model: torch.nn.Module, data_loader: DataLoader, top_n=1):
    predicted_labels = torch.tensor([], dtype=torch.long).to(device)

    for i, (batch_id, batch_tok, batch_cm, batch_type) in enumerate(data_loader):
        _, batch_labels = make_batch_prediction_TW(
            model.to(device),
            batch_id.to(device),
            batch_tok.to(device),
            batch_cm.to(device),
            batch_type.to(device),
            top_n=top_n,
        )

        predicted_labels = torch.cat((predicted_labels, batch_labels), 0)

    predicted_labels = predicted_labels.data.cpu().numpy()

    return predicted_labels


class _TypeWriter(ProjectWideInference):
    def method(self) -> str:
        return f"typewriterN{self.topn}"

    def __init__(
        self,
        model_path: pathlib.Path,
        topn: int,
        cpu_executor: ProcessPoolExecutor | None = None,
        model_executor: ThreadPoolExecutor | None = None,
    ):
        super().__init__(cpu_executor=cpu_executor, model_executor=model_executor)
        self.topn = topn
        self.model_path = model_path

        self.w2v_token_model = Word2Vec.load(
            str(self.model_path / "w2v_token_model.bin")
        )
        self.w2v_comments_model = Word2Vec.load(
            str(self.model_path / "w2v_comments_model.bin")
        )

        self.tw_model = torch.load(
            self.model_path / "tw_pretrained_model_combined.pt",
            map_location=device,
        )
        self.label_encoder = pickle.load(
            open(join(self.model_path, "label_encoder.pkl"), "rb")
        )

        if not torch.cuda.is_available():
            self.tw_model = self.tw_model.module

    def _infer_project(
        self, mutable: pathlib.Path, subset: set[pathlib.Path]
    ) -> pt.DataFrame[InferredSchema]:
        with tempfile.TemporaryDirectory() as td:
            self.logger.info("Extracting predictables...")
            paths2predictables = self.extract_predictables(td, mutable, subset)
            self.logger.debug(paths2predictables)

            self.logger.info("Transforming sequences and executing model...")
            paths2predictions = self.make_predictions(td, paths2predictables, subset)
            self.logger.debug(paths2predictions)

        return TWProjectApplier.collect_topn(
            project=mutable,
            predictions=paths2predictions,
            subset=subset,
            topn=self.topn,
            tool=self,
        )

    def extract_predictables(
        self, td: str, project: pathlib.Path, subset: set[pathlib.Path]
    ) -> dict[pathlib.Path, tuple[pd.DataFrame, pd.DataFrame] | None]:
        top999_types = pd.read_csv(join(self.model_path, "top_999_types.csv"))
        with self.cpu_executor() as executor:
            tasks = executor.map(
                _file2predictables,
                itertools.repeat(td),
                itertools.repeat(project),
                subset,
                itertools.repeat(top999_types),
            )
            paths2datapoints = dict(
                tqdm.tqdm(tasks, total=len(subset), desc="Predictable Extraction")
            )

            # Remove all entries that

            return paths2datapoints

    def make_predictions(
        self,
        td: str,
        paths2predictables: dict[
            pathlib.Path, tuple[pd.DataFrame, pd.DataFrame] | None
        ],
        subset: set[pathlib.Path],
    ) -> dict[pathlib.Path, tuple[list[list[Parameter]], list[list[Return]]]]:
        with self.model_executor() as executor:
            tasks = executor.map(
                self._predictables2predictions,
                itertools.repeat(td),
                itertools.repeat(paths2predictables),
                subset,
            )
            paths2datapoints = dict(
                tqdm.tqdm(tasks, total=len(subset), desc="Datapoint Extraction")
            )

            return paths2datapoints

    def _predictables2predictions(
        self,
        temp_dir: str,
        paths2predictables: dict[
            pathlib.Path, tuple[pd.DataFrame, pd.DataFrame] | None
        ],
        file: pathlib.Path,
    ) -> tuple[pathlib.Path, tuple[list[list[Parameter]], list[list[Return]]]]:
        if (predictables := paths2predictables.get(file, None)) is None:
            return file, ([], [])

        ext_funcs_df_params, ext_funcs_df_ret = predictables
        temp_dir = os.path.join(temp_dir, str(file.with_suffix("")))

        # Arguments transformers
        id_trans_func_param = lambda row: IdentifierSequence(
            self.w2v_token_model, row.arg_name, row.other_args, row.func_name
        )
        token_trans_func_param = lambda row: TokenSequence(
            self.w2v_token_model, 7, 3, row.arg_occur, None
        )
        cm_trans_func_param = lambda row: CommentSequence(
            self.w2v_comments_model, row.func_descr, row.arg_comment, None
        )

        # Returns transformers
        id_trans_func_ret = lambda row: IdentifierSequence(
            self.w2v_token_model, None, row.arg_names_str, row.name
        )
        token_trans_func_ret = lambda row: TokenSequence(
            self.w2v_token_model, 7, 3, None, row.return_expr_str
        )
        cm_trans_func_ret = lambda row: CommentSequence(
            self.w2v_comments_model, row.func_descr, None, row.return_descr
        )

        dp_ids_params = process_datapoints_TW(
            os.path.join(temp_dir, "ext_funcs_params.csv"),
            temp_dir,
            "identifiers_",
            "params",
            id_trans_func_param,
        )

        dp_ids_ret = process_datapoints_TW(
            os.path.join(temp_dir, "ext_funcs_ret.csv"),
            temp_dir,
            "identifiers_",
            "ret",
            id_trans_func_ret,
        )

        # print("Generating tokens sequences")
        dp_tokens_params = process_datapoints_TW(
            os.path.join(temp_dir, "ext_funcs_params.csv"),
            temp_dir,
            "tokens_",
            "params",
            token_trans_func_param,
        )
        dp_tokens_ret = process_datapoints_TW(
            os.path.join(temp_dir, "ext_funcs_ret.csv"),
            temp_dir,
            "tokens_",
            "ret",
            token_trans_func_ret,
        )

        # print("Generating comments sequences")
        dp_cms_params = process_datapoints_TW(
            join(temp_dir, "ext_funcs_params.csv"),
            temp_dir,
            "comments_",
            "params",
            cm_trans_func_param,
        )
        dp_cms_ret = process_datapoints_TW(
            join(temp_dir, "ext_funcs_ret.csv"),
            temp_dir,
            "comments_",
            "ret",
            cm_trans_func_ret,
        )

        # print("Generating sequences for available types hints")
        dp_params_aval_types, dp_ret__aval_types = gen_aval_types_datapoints(
            join(temp_dir, "ext_funcs_params.csv"),
            join(temp_dir, "ext_funcs_ret.csv"),
            "",
            temp_dir,
        )

        self.logger.debug(
            "--------------------Argument Types Prediction--------------------"
        )
        id_params, tok_params, com_params, aval_params = load_param_data(temp_dir)
        params_data_loader = DataLoader(
            TensorDataset(id_params, tok_params, com_params, aval_params)
        )

        params_pred = [
            p for p in evaluate_TW(self.tw_model, params_data_loader, self.topn)
        ]

        # (function, parameter, [type]s)
        param_inf: list[tuple[str, str, list[str]]] = []
        for i, p in enumerate(params_pred):
            fname = ext_funcs_df_params["func_name"].iloc[i]
            param = ext_funcs_df_params["arg_name"].iloc[i]
            predictables = list(self.label_encoder.inverse_transform(p))

            p = " ".join(
                ["%d. %s" % (j, t) for j, t in enumerate(predictables, start=1)]
            )
            self.logger.debug(f"{file} -> {fname}: {param} -> {p}")

            param_inf.append((fname, param, predictables))

        self.logger.debug(
            "--------------------Return Types Prediction--------------------"
        )
        id_ret, tok_ret, com_ret, aval_ret = load_ret_data(temp_dir)
        ret_data_loader = DataLoader(TensorDataset(id_ret, tok_ret, com_ret, aval_ret))

        ret_pred = [p for p in evaluate_TW(self.tw_model, ret_data_loader, self.topn)]

        ret_inf: list[tuple[str, list[str]]] = []
        for i, p in enumerate(ret_pred):
            fname = ext_funcs_df_ret["name"].iloc[i]
            predictables = list(self.label_encoder.inverse_transform(p))

            p = " ".join(
                ["%d. %s" % (j, t) for j, t in enumerate(predictables, start=1)]
            )
            self.logger.debug(f"{file} -> {fname} -> {p}")

            ret_inf.append((fname, predictables))

        arg_batches: list[list[Parameter]] = []
        ret_batches: list[list[Return]] = []

        for n in range(self.topn):
            arg_batch: list[Parameter] = []
            ret_batch: list[Return] = []

            for fname, argname, ppreds in param_inf:
                arg_batch.append(Parameter(fname=fname, pname=argname, ty=ppreds[n]))

            for fname, rp in ret_inf:
                ret_batch.append(Return(fname=fname, ty=rp[n]))

            arg_batches.append(arg_batch)
            ret_batches.append(ret_batch)

        return file, (arg_batches, ret_batches)


def _file2predictables(
    temp_dir: str,
    project: pathlib.Path,
    file: pathlib.Path,
    top999_types: pd.DataFrame,
) -> tuple[pathlib.Path, tuple[pd.DataFrame, pd.DataFrame] | None]:
    filename = str(project / file)
    temp_dir = os.path.join(temp_dir, str(file.with_suffix("")))

    os.makedirs(name=temp_dir, exist_ok=True)

    ext_funcs = process_py_src_file(filename)
    if not ext_funcs:
        return file, None

    write_ext_funcs(ext_funcs, filename, temp_dir)
    ext_funcs_df = pd.read_csv(
        os.path.join(temp_dir, f"ext_funcs_{file.with_suffix('').name}.csv")
    )
    ext_funcs_df = filter_functions(ext_funcs_df)
    ext_funcs_df_params = gen_argument_df_TW(ext_funcs_df)

    ext_funcs_df_params = ext_funcs_df_params[
        (ext_funcs_df_params["arg_name"] != "self")
        & (
            (ext_funcs_df_params["arg_type"] != "Any")
            & (ext_funcs_df_params["arg_type"] != "None")
        )
    ]

    ext_funcs_df_ret = filter_ret_funcs(ext_funcs_df)
    ext_funcs_df_ret = format_df(ext_funcs_df_ret)

    ext_funcs_df_ret["arg_names_str"] = ext_funcs_df_ret["arg_names"].apply(
        lambda l: " ".join([v for v in l if v != "self"])
    )
    ext_funcs_df_ret["return_expr_str"] = ext_funcs_df_ret["return_expr"].apply(
        lambda l: " ".join([re.sub(r"self\.?", "", v) for v in l])
    )
    ext_funcs_df_ret = ext_funcs_df_ret.drop(
        columns=[
            "has_type",
            "arg_names",
            "arg_types",
            "arg_descrs",
            "return_expr",
        ]
    )

    df_avl_types = top999_types
    ext_funcs_df_params, ext_funcs_df_ret = encode_aval_types_TW(
        ext_funcs_df_params, ext_funcs_df_ret, df_avl_types
    )

    ext_funcs_df_params.to_csv(
        os.path.join(temp_dir, "ext_funcs_params.csv"), index=False
    )
    ext_funcs_df_ret.to_csv(os.path.join(temp_dir, "ext_funcs_ret.csv"), index=False)

    return file, (ext_funcs_df_params, ext_funcs_df_ret)


class TypeWriterTopN(_TypeWriter):
    def __init__(
        self,
        topn: int,
        cpu_executor: ProcessPoolExecutor | None = None,
        model_executor: ThreadPoolExecutor | None = None,
    ):
        super().__init__(
            model_path=pathlib.Path("models/typewriter"),
            topn=topn,
            cpu_executor=cpu_executor,
            model_executor=model_executor,
        )


TypeWriterTop1 = functools.partial(TypeWriterTopN, topn=1)
TypeWriterTop3 = functools.partial(TypeWriterTopN, topn=3)
TypeWriterTop5 = functools.partial(TypeWriterTopN, topn=5)
TypeWriterTop10 = functools.partial(TypeWriterTopN, topn=10)
