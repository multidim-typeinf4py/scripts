import dataclasses
import logging
import os
import pathlib
import pickle
import re
import tempfile
from ast import literal_eval
from os.path import splitext, basename, join
from typing import List, no_type_check, Optional

import libcst
import libcst as cst
import libcst.metadata as metadata
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

from common import visitors
from common.annotations import (
    MultiVarAnnotations,
    FunctionAnnotation,
    FunctionKey,
)
from common.schemas import InferredSchema
from common.storage import TypeCollection
from ._base import PerFileInference

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


class TypeWriter(PerFileInference):
    method = "typewriter"

    _MODEL_DIR = pathlib.Path(
        "/home/benji/Documents/Uni/heidelberg/05/masterarbeit/impls/scripts/models/typewriter"
    )

    def __init__(self, cache: Optional[pathlib.Path], topn: int):
        super().__init__(cache)
        self.topn = topn

        self.w2v_token_model = Word2Vec.load(
            str(TypeWriter._MODEL_DIR / "w2v_token_model.bin")
        )
        self.w2v_comments_model = Word2Vec.load(
            str(TypeWriter._MODEL_DIR / "w2v_comments_model.bin")
        )

        self.tw_model = torch.load(
            TypeWriter._MODEL_DIR / "tw_pretrained_model_combined.pt",
            map_location=device,
        )
        self.label_encoder = pickle.load(
            open(join(TypeWriter._MODEL_DIR, "label_encoder.pkl"), "rb")
        )

        if not torch.cuda.is_available():
            self.tw_model = self.tw_model.module

    def _infer_file(
            self, root: pathlib.Path, relative: pathlib.Path
    ) -> pt.DataFrame[InferredSchema]:
        filename = str(root / relative)

        with tempfile.TemporaryDirectory() as TEMP_DIR:
            ext_funcs = process_py_src_file(filename)
            if not ext_funcs:
                self.logger.warning(
                    f"Did not find any functions in {relative}, therefore no types to infer"
                )
                return InferredSchema.example(size=0)

            # self.logger.debug(f"Number of the extracted functions: {len(ext_funcs)}")

            write_ext_funcs(ext_funcs, filename, TEMP_DIR)

            ext_funcs_df = pd.read_csv(
                os.path.join(TEMP_DIR, f"ext_funcs_{relative.with_suffix('').name}.csv")
            )
            ext_funcs_df = filter_functions(ext_funcs_df)
            ext_funcs_df_params = gen_argument_df_TW(ext_funcs_df)

            # self.logger.debug(
            #    f"Number of extracted arguments: {ext_funcs_df_params['arg_name'].count()}"
            # )
            ext_funcs_df_params = ext_funcs_df_params[
                (ext_funcs_df_params["arg_name"] != "self")
                & (
                        (ext_funcs_df_params["arg_type"] != "Any")
                        & (ext_funcs_df_params["arg_type"] != "None")
                )
                ]

            # self.logger.debug(
            #    f"Number of Arguments after ignoring self and types with Any and None: {ext_funcs_df_params.shape[0]}"
            # )

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

            df_avl_types = pd.read_csv(join(TypeWriter._MODEL_DIR, "top_999_types.csv"))
            ext_funcs_df_params, ext_funcs_df_ret = encode_aval_types_TW(
                ext_funcs_df_params, ext_funcs_df_ret, df_avl_types
            )

            ext_funcs_df_params.to_csv(
                os.path.join(TEMP_DIR, "ext_funcs_params.csv"), index=False
            )
            ext_funcs_df_ret.to_csv(
                os.path.join(TEMP_DIR, "ext_funcs_ret.csv"), index=False
            )

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

            # print("Generating identifiers sequences")
            dp_ids_params = process_datapoints_TW(
                os.path.join(TEMP_DIR, "ext_funcs_params.csv"),
                TEMP_DIR,
                "identifiers_",
                "params",
                id_trans_func_param,
            )
            if dp_ids_params is False:
                return InferredSchema.to_schema().example(size=0)

            dp_ids_ret = process_datapoints_TW(
                os.path.join(TEMP_DIR, "ext_funcs_ret.csv"),
                TEMP_DIR,
                "identifiers_",
                "ret",
                id_trans_func_ret,
            )

            # print("Generating tokens sequences")
            dp_tokens_params = process_datapoints_TW(
                os.path.join(TEMP_DIR, "ext_funcs_params.csv"),
                TEMP_DIR,
                "tokens_",
                "params",
                token_trans_func_param,
            )
            dp_tokens_ret = process_datapoints_TW(
                os.path.join(TEMP_DIR, "ext_funcs_ret.csv"),
                TEMP_DIR,
                "tokens_",
                "ret",
                token_trans_func_ret,
            )

            # print("Generating comments sequences")
            dp_cms_params = process_datapoints_TW(
                join(TEMP_DIR, "ext_funcs_params.csv"),
                TEMP_DIR,
                "comments_",
                "params",
                cm_trans_func_param,
            )
            dp_cms_ret = process_datapoints_TW(
                join(TEMP_DIR, "ext_funcs_ret.csv"),
                TEMP_DIR,
                "comments_",
                "ret",
                cm_trans_func_ret,
            )

            # print("Generating sequences for available types hints")
            dp_params_aval_types, dp_ret__aval_types = gen_aval_types_datapoints(
                join(TEMP_DIR, "ext_funcs_params.csv"),
                join(TEMP_DIR, "ext_funcs_ret.csv"),
                "",
                TEMP_DIR,
            )

            # self.logger.info("--------------------Argument Types Prediction--------------------")
            id_params, tok_params, com_params, aval_params = load_param_data(TEMP_DIR)
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
                predictions = list(self.label_encoder.inverse_transform(p))

                p = " ".join(
                    ["%d. %s" % (j, t) for j, t in enumerate(predictions, start=1)]
                )
                self.logger.debug(f"{fname}: {param} -> {p}")

                param_inf.append((fname, param, predictions))

            # self.logger.info("--------------------Return Types Prediction--------------------")
            id_ret, tok_ret, com_ret, aval_ret = load_ret_data(TEMP_DIR)
            ret_data_loader = DataLoader(
                TensorDataset(id_ret, tok_ret, com_ret, aval_ret)
            )

            ret_pred = [
                p for p in evaluate_TW(self.tw_model, ret_data_loader, self.topn)
            ]

            ret_inf: list[tuple[str, list[str]]] = []
            for i, p in enumerate(ret_pred):
                fname = ext_funcs_df_ret["name"].iloc[i]
                predictions = list(self.label_encoder.inverse_transform(p))

                p = " ".join(
                    ["%d. %s" % (j, t) for j, t in enumerate(predictions, start=1)]
                )
                self.logger.debug(f"{fname} -> {p}")

                ret_inf.append((fname, predictions))

            collections: list[pd.DataFrame] = list()
            module = cst.parse_module((root / relative).open().read())

            arg_batches: list[list[Parameter]] = []
            ret_batches: list[list[Return]] = []

            for n in range(self.topn):
                arg_batch: list[Parameter] = []
                ret_batch: list[Return] = []

                for fname, argname, ppreds in param_inf:
                    arg_batch.append(
                        Parameter(fname=fname, pname=argname, ty=ppreds[n])
                    )

                for fname, rp in ret_inf:
                    ret_batch.append(Return(fname=fname, ty=rp[n]))

                arg_batches.append(arg_batch)
                ret_batches.append(ret_batch)

            for n, (arg_batch, ret_batch) in enumerate(
                    zip(arg_batches, ret_batches), start=1
            ):
                visitor = Typewriter2Annotations(arg_batch, ret_batch, self.logger)
                metadata.MetadataWrapper(module, unsafe_skip_copy=True).visit(visitor)

                collection = TypeCollection.from_annotations(
                    file=relative, annotations=visitor.annotations, strict=True
                )
                collections.append(collection.df.assign(topn=n))

            return (
                pd.concat(collections, ignore_index=True)
                .assign(method=self.method)
                .pipe(pt.DataFrame[InferredSchema])
            )


@dataclasses.dataclass
class Parameter:
    fname: str
    pname: str
    ty: str


@dataclasses.dataclass
class Return:
    fname: str
    ty: str


class Typewriter2Annotations(
    visitors.HintableParameterVisitor, visitors.HintableReturnVisitor
):
    METADATA_DEPENDENCIES = (metadata.QualifiedNameProvider,)

    def __init__(
            self, parameters: list[Parameter], returns: list[Return], logger: logging.Logger
    ) -> None:
        super().__init__()
        self.parameters = parameters
        self.param_cursor = 0

        self.returns = returns
        self.ret_cursor = 0

        self.inferred_func_params: list[cst.Param] = []
        self.inferred_func_return: Optional[cst.Annotation] = None

        self.current_fkey: Optional[FunctionKey] = None
        self.annotations = MultiVarAnnotations.empty()

        self.logger = logger

    def unannotated_function(self, function: libcst.FunctionDef) -> None:
        return self.function(function)

    def annotated_function(
            self, function: libcst.FunctionDef, annotation: libcst.Annotation
    ) -> None:
        return self.function(function)

    def function(self, f: libcst.FunctionDef) -> None:
        qname = next(
            iter(self.get_metadata(metadata.QualifiedNameProvider, f))
        ).name.replace(".<locals>.", ".")
        self.current_fkey = FunctionKey.make(name=qname, params=f.params)

        name = preprocessor.process_identifier(f.name.value)

        rc = self.ret_cursor
        try:
            while (r := self.returns[rc]).fname != name:
                rc += 1
        except IndexError:
            self.logger.warning(
                f"Cannot find prediction for function {f.name.value}, assuming no prediction made"
            )
            self.inferred_func_return = self._read_tw_pred(None)
            return

        if rc - self.ret_cursor > 1:
            self.logger.warning(
                f"Had to skip {rc - self.ret_cursor} ret entries to find {f.name.value}"
            )
        self.ret_cursor = rc

        self.inferred_func_return = self._read_tw_pred(r.ty)
        self.ret_cursor += 1

    def unannotated_param(self, param: libcst.Param) -> None:
        return self.param(param)

    def annotated_param(
            self, param: libcst.Param, annotation: libcst.Annotation
    ) -> None:
        return self.param(param)

    def param(self, param: libcst.Param) -> None:
        if param.name.value == "self":
            # TypeWriter simply ignores self, with no further context checking
            return

        name = preprocessor.process_identifier(param.name.value)

        pc = self.param_cursor
        try:
            while (p := self.parameters[pc]).pname != name:
                pc += 1
        except IndexError:
            self.logger.warning(
                f"Cannot find prediction for parameter {param.name.value}, assuming no prediction made"
            )
            self.inferred_func_params.append(
                cst.Param(name=param.name, annotation=self._read_tw_pred(None))
            )
            return

        if pc - self.param_cursor > 1:
            self.logger.warning(
                f"Had to skip {pc - self.param_cursor} ret entries to find {param.name.value}"
            )
        self.param_cursor = pc

        self.inferred_func_params.append(
            cst.Param(name=param.name, annotation=self._read_tw_pred(p.ty))
        )
        self.param_cursor += 1

    def visit_FunctionDef_body(self, node: libcst.FunctionDef) -> None:
        assert self.current_fkey is not None

        self.annotations.functions[self.current_fkey] = FunctionAnnotation(
            parameters=cst.Parameters(list(self.inferred_func_params)),
            returns=self.inferred_func_return,
        )

        self.current_fkey = None
        self.inferred_func_params.clear()
        self.inferred_func_return = None

    def _read_tw_pred(self, annotation: Optional[str]) -> Optional[libcst.Annotation]:
        if annotation is None or annotation == "other":
            return None

        import builtins, typing, pathlib

        nongeneric, _, _ = annotation.partition("[")
        for mod in (builtins, typing, pathlib):
            if nongeneric in dir(mod):
                return libcst.Annotation(
                    annotation=libcst.parse_expression(f"{mod.__name__}.{annotation}")
                )
        else:
            return libcst.Annotation(annotation=libcst.parse_expression(annotation))
