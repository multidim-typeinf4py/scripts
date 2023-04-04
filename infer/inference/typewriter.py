from ast import literal_eval
import itertools
import operator
import pathlib
import pickle
import re
from typing import List, no_type_check
from os.path import splitext, basename, join
import libcst as cst
import libcst.metadata as metadata

from common.storage import TypeCollection

from ._base import PerFileInference
from common.schemas import InferredSchema

from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import pandera.typing as pt

from typewriter.dltpy.preprocessing.pipeline import extractor, read_file, preprocessor
from typewriter.dltpy.input_preparation.generate_df import format_df
from typewriter.dltpy.preprocessing.extractor import Function, ParseError
from typewriter.extraction import (
    process_datapoints_TW,
    IdentifierSequence,
    TokenSequence,
    CommentSequence,
    gen_aval_types_datapoints,
)
from typewriter.model import load_data_tensors_TW, make_batch_prediction_TW
from typewriter.prepocessing import filter_functions, gen_argument_df_TW, encode_aval_types_TW

from common.annotations import (
    MultiVarAnnotations,
    FunctionAnnotation,
    FunctionKey,
)

import torch
from torch.utils.data import DataLoader, TensorDataset


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
        join(output_dir, "ext_funcs_" + splitext(basename(src_file))[0] + ".csv"), index=False
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

    _MODEL_DIR = pathlib.Path("models", "typewriter")

    def __init__(self, project: pathlib.Path) -> None:
        super().__init__(project)
        self.topn = 3

    def _infer_file(self, relative: pathlib.Path) -> pt.DataFrame[InferredSchema]:
        import tempfile

        TD = tempfile.TemporaryDirectory()
        TEMP_DIR = TD.name

        ext_funcs = process_py_src_file(str(self.project / relative))
        # print("Number of the extracted functions: ", len(ext_funcs))
        # print(
        #    "Writing the extracted functions to the disk: ",
        #    join(
        #        TEMP_DIR,
        #        "ext_funcs_" + splitext(basename(str(self.project / relative)))[0] + ".csv",
        #    ),
        # )
        write_ext_funcs(ext_funcs, str(self.project / relative), TEMP_DIR)

        ext_funcs_df = pd.read_csv(
            join(
                TEMP_DIR,
                "ext_funcs_" + splitext(basename(str(self.project / relative)))[0] + ".csv",
            )
        )
        # print("Filtering out trivial functions like __str__ if exists")
        ext_funcs_df = filter_functions(ext_funcs_df)

        ext_funcs_df_params = gen_argument_df_TW(ext_funcs_df)

        # print("Number of extracted arguments: ", ext_funcs_df_params["arg_name"].count())

        ext_funcs_df_params = ext_funcs_df_params[
            (ext_funcs_df_params["arg_name"] != "self")
            & (
                (ext_funcs_df_params["arg_type"] != "Any")
                & (ext_funcs_df_params["arg_type"] != "None")
            )
        ]

        # print(
        #    "Number of Arguments after ignoring self and types with Any and None: ",
        #    ext_funcs_df_params.shape[0],
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
            columns=["has_type", "arg_names", "arg_types", "arg_descrs", "return_expr"]
        )

        # print("Encodes available types hints...")
        df_avl_types = pd.read_csv(join(TypeWriter._MODEL_DIR, "top_999_types.csv"))
        ext_funcs_df_params, ext_funcs_df_ret = encode_aval_types_TW(
            ext_funcs_df_params, ext_funcs_df_ret, df_avl_types
        )

        ext_funcs_df_params.to_csv(join(TEMP_DIR, "ext_funcs_params.csv"), index=False)
        ext_funcs_df_ret.to_csv(join(TEMP_DIR, "ext_funcs_ret.csv"), index=False)

        path = str(TypeWriter._MODEL_DIR)

        if not hasattr(self, "w2v_token_model"):
            print("Loading pre-trained Word2Vec models")
            self.w2v_token_model = Word2Vec.load(join(path, "w2v_token_model.bin"))
            self.w2v_comments_model = Word2Vec.load(join(path, "w2v_comments_model.bin"))

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
            join(TEMP_DIR, "ext_funcs_params.csv"),
            TEMP_DIR,
            "identifiers_",
            "params",
            id_trans_func_param,
        )
        if dp_ids_params is False:
            return InferredSchema.to_schema().example(size=0)
        dp_ids_ret = process_datapoints_TW(
            join(TEMP_DIR, "ext_funcs_ret.csv"), TEMP_DIR, "identifiers_", "ret", id_trans_func_ret
        )

        # print("Generating tokens sequences")
        dp_tokens_params = process_datapoints_TW(
            join(TEMP_DIR, "ext_funcs_params.csv"),
            TEMP_DIR,
            "tokens_",
            "params",
            token_trans_func_param,
        )
        dp_tokens_ret = process_datapoints_TW(
            join(TEMP_DIR, "ext_funcs_ret.csv"), TEMP_DIR, "tokens_", "ret", token_trans_func_ret
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
            join(TEMP_DIR, "ext_funcs_ret.csv"), TEMP_DIR, "comments_", "ret", cm_trans_func_ret
        )

        # print("Generating sequences for available types hints")
        dp_params_aval_types, dp_ret__aval_types = gen_aval_types_datapoints(
            join(TEMP_DIR, "ext_funcs_params.csv"),
            join(TEMP_DIR, "ext_funcs_ret.csv"),
            "",
            TEMP_DIR,
        )

        if not hasattr(self, "tw_model"):
            print("Loading the pre-trained neural model of TypeWriter from the disk...")
            self.tw_model = torch.load(
                TypeWriter._MODEL_DIR / "tw_pretrained_model_combined.pt", map_location=device
            )
            self.label_encoder = pickle.load(
                open(join(TypeWriter._MODEL_DIR, "label_encoder.pkl"), "rb")
            )

            if not torch.cuda.is_available():
                self.tw_model = self.tw_model.module

        # print("--------------------Argument Types Prediction--------------------")
        id_params, tok_params, com_params, aval_params = load_param_data(TEMP_DIR)
        params_data_loader = DataLoader(
            TensorDataset(id_params, tok_params, com_params, aval_params)
        )

        params_pred = [p for p in evaluate_TW(self.tw_model, params_data_loader, self.topn)]

        # (function, parameter, [type]s)
        param_inf: list[tuple[str, str, list[str]]] = []
        for i, p in enumerate(params_pred):
            fname = ext_funcs_df_params["func_name"].iloc[i]
            param = ext_funcs_df_params["arg_name"].iloc[i]
            predictions = list(self.label_encoder.inverse_transform(p))

            # p = " ".join(["%d. %s" % (j, t) for j, t in enumerate(predictions, start=1)])
            # print(f"{fname}: {param} -> {p}")

            param_inf.append((fname, param, predictions))

        grouped_param_inf: list[tuple[str, list[tuple[str, str, list[str]]]]] = list(
            (k, list(g)) for k, g in itertools.groupby(param_inf, key=operator.itemgetter(0))
        )

        # print("--------------------Return Types Prediction--------------------")
        id_ret, tok_ret, com_ret, aval_ret = load_ret_data(TEMP_DIR)
        ret_data_loader = DataLoader(TensorDataset(id_ret, tok_ret, com_ret, aval_ret))

        ret_pred = [p for p in evaluate_TW(self.tw_model, ret_data_loader, self.topn)]

        ret_inf: list[tuple[str, list[str]]] = []
        for i, p in enumerate(ret_pred):
            fname = ext_funcs_df_ret["name"].iloc[i]
            predictions = list(self.label_encoder.inverse_transform(p))

            ret_inf.append((fname, predictions))

        if len(grouped_param_inf) != len(ret_inf):
            print(
                "WARNING: TYPEWRITER HAS DIFFERING COUNTS OF SIGNATURES AND RETURNS; COVERAGE & ACCURACY MAY DETERIORATE"
            )
        TD.cleanup()

        collections: list[pd.DataFrame] = list()
        module = cst.parse_module((self.project / relative).open().read())

        # param_by_top_n = [
        #     (fname, param, hint)
        #     for fname, infs in grouped_param_inf
        #     for _, param, hints in infs
        #     for hint in hints
        # ]
        #
        # rets_by_top_n = [
        #     (fname, hint)
        #     for fname, infs in ret_inf
        #     for hint in infs
        # ]

        # Top N predictions requires multiple passes due to libcst not supporting
        # multiple hints for a single slot (dictionary based symbol lookup)

        arg_batches: list[list[tuple[str, str, str]]] = []
        ret_batches: list[list[tuple[str, str]]] = []

        for n in range(self.topn):
            arg_batch: list[tuple[str, str, str]] = []
            ret_batch: list[tuple[str, str]] = []

            for (_, arg_predictions), (_, ret_predictions) in zip(
                grouped_param_inf, ret_inf, strict=True
            ):

                for fname, argname, argpreds in arg_predictions:
                    arg_batch.append((fname, argname, argpreds[n]))
                ret_batch.append((fname, ret_predictions[n]))

            arg_batches.append(arg_batch)
            ret_batches.append(ret_batch)

        for n, (arg_batch, ret_batch) in enumerate(zip(arg_batches, ret_batches)):
            visitor = Typewriter2Annotations(arg_batch, ret_batch)
            metadata.MetadataWrapper(module).visit(visitor)

            annotations = visitor.annotations
            collection = TypeCollection.from_annotations(
                file=relative, annotations=annotations, strict=True
            )
            collections.append(collection.df.assign(method=self.method, topn=n))

        return pd.concat(collections, ignore_index=True).pipe(pt.DataFrame[InferredSchema])


class Typewriter2Annotations(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (
        metadata.ScopeProvider,
        metadata.QualifiedNameProvider,
    )

    def __init__(
        self,
        parameters: list[tuple[str, str, str]],
        returns: list[tuple[str, str]],
    ) -> None:
        self.parameters: list[tuple[str, list[tuple[str, str]]]] = [
            (k, list(g)) for k, g in itertools.groupby(parameters, key=operator.itemgetter(0))
        ]
        self.returns = returns
        self.annotations = MultiVarAnnotations.empty()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        is_fn = None
        scope = self.get_metadata(metadata.ScopeProvider, node)
        if isinstance(scope, metadata.ClassScope):
            is_fn = False

        elif isinstance(scope, metadata.GlobalScope):
            is_fn = True

        elif isinstance(scope, metadata.FunctionScope):
            # TypeWriter does not make guesses for inner functions
            return None

        else:
            print(f"WARNING: unexpected {scope=} for {cst.Module([]).code_for_node(node)}!")
            return None

        parameters = self._load_parameters(node, function=is_fn)
        ret_found, ret = self._load_return(node)

        if (parameters is None) ^ (ret_found is False):
            raise RuntimeWarning(
                f"Clash in parameter and return predictions: {self.get_metadata(metadata.QualifiedNameProvider, node)}, {self.parameters=}, {self.returns=}"
            )

        elif parameters is None and ret_found is False:
            return None

        fname = next(iter(self.get_metadata(metadata.QualifiedNameProvider, node)))
        fkey = FunctionKey.make(fname.name, node.params)

        anno_params = []
        for pname, phint in parameters or []:
            anno_params.append(
                cst.Param(
                    name=cst.Name(pname),
                    annotation=cst.Annotation(cst.parse_expression(phint) if phint else None),
                )
            )

        self.annotations.functions[fkey] = FunctionAnnotation(
            parameters=cst.Parameters(params=anno_params),
            returns=cst.Annotation(
                cst.parse_expression(ret) if ret else None,
            ),
        )

    def _load_parameters(
        self, node: cst.FunctionDef, function: bool
    ) -> list[tuple[str, str | None]] | None:
        if not self.parameters:
            return None

        symbol, hints = self.parameters[0]

        hints = [hint[1:] for hint in hints]
        if symbol != preprocessor.process_identifier(node.name.value):
            return None

        # Fix overly greediness of itertools.groupby when identically named functions and methods follow each other
        if function or not len(node.params.params):
            self.parameters[0] = (
                self.parameters[0][0],
                self.parameters[0][1][len(node.params.params) :],
            )

        else:
            hints = [(node.params.params[0].name.value, None)] + hints
            self.parameters[0] = (
                self.parameters[0][0],
                self.parameters[0][1][len(node.params.params) - 1 :],
            )

        if not self.parameters[0][1]:
            self.parameters = self.parameters[1:]

        return [(arg, _handle_missing_coverage(anno)) for arg, anno in hints]

    def _load_return(self, node: cst.FunctionDef) -> tuple[bool, str | None]:
        if not self.returns:
            return False, None

        symbol, hint = self.returns[0]
        if symbol != preprocessor.process_identifier(node.name.value):
            return False, None

        self.returns = self.returns[1:]
        return True, _handle_missing_coverage(hint)


def _handle_missing_coverage(annotation: str | None) -> str | None:
    if annotation is None or annotation == "other":
        return None
    return annotation
