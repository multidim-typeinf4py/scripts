import json
import pathlib
import pprint
import typing
from infer.inference._base import ProjectWideInference

import pandas as pd
import pandera.typing as pt
from common.schemas import InferredSchema

from dpu_utils.utils import RichPath

from data_preparation.scripts.graph_generator import extract_graphs
from typilus.model import model_restore_helper
from typilus.utils.predict import ignore_annotation
from type_check import annotater

import libcst
from libcst import codemod

from symbols.collector import build_type_collection
import utils


class TypilusAnnotator(annotater.Annotater):
    def __init__(self, ppath: str, typing_rules: pathlib.Path) -> None:
        super().__init__(
            tc=None,
            ppath=ppath,
            granularity="var",
            typing_rules=typing_rules,
        )


class TypilusAnnotationApplier(codemod.ContextAwareTransformer):
    def __init__(
        self,
        context: codemod.CodemodContext,
        predictions: RichPath,
        topn: int,
        typing_rules: pathlib.Path,
    ) -> None:
        super().__init__(context)
        self.predictions = predictions
        self.topn = topn

        self.annotator = TypilusAnnotator(
            ppath=predictions.path,
            typing_rules=typing_rules,
        )

    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        if (
            new_fpath := self.annotator.annotate(
                fpath=self.context.filename,
                pred_idx=-1,
                type_idx=self.topn,
            )
        ) and new_fpath != self.context.filename:
            return libcst.parse_module(pathlib.Path(new_fpath).read_text())

        return tree


class TypilusPrediction(typing.TypedDict):
    annotation_type: str
    location: tuple[int, int]
    name: str
    node_id: int
    original_annotation: str
    predicted_annotation_logprob_dist: list[tuple[str, float]]


class Typilus(ProjectWideInference):
    def __init__(self, model_folder: pathlib.Path, topn: int) -> None:
        super().__init__()

        self.topn = topn

        self.model_folder = model_folder
        self.model_path = self.model_folder / "typilus.pkl.gz"

        self.model = model_restore_helper.restore(
            RichPath.create(str(self.model_path)),
            is_train=False,
            hyper_overrides={
                "run_id": "indexing",
                "dropout_keep_rate": 1.0,
            },
        )

        self.typing_rules = self.model_folder / "typingRules.json"

    def method(self) -> str:
        return f"typilisN{self.topn}"

    def _infer_project(
        self, mutable: pathlib.Path, subset: set[pathlib.Path]
    ) -> pt.DataFrame[InferredSchema]:
        # Transform repo into test dataset
        test_dataset_path = self.repo_to_dataset(mutable)

        # Predict over transformed dataset
        pred_path = self.predict(
            dataset=test_dataset_path,
            predictions=mutable / ".predictions.json.gz",
        )

        # Apply annotations
        return self.annotate_and_collect(mutable, subset, pred_path)

    def repo_to_dataset(self, repo: pathlib.Path) -> RichPath:
        test_dataset = repo / "inference-dataset"

        with (duplicates := repo / "dummies.json.gz").open("w") as f:
            json.dump({}, f)

        extract_graphs.main(
            {
                "SOURCE_FOLDER": str(repo),
                "DUPLICATES_JSON": str(duplicates),
                "SAVE_FOLDER": str(test_dataset),
                "TYPING_RULES": str(self.typing_rules),
            }
        )
        return RichPath.create(path=str(test_dataset))

    def predict(self, dataset: RichPath, predictions: pathlib.Path) -> RichPath:
        ps = []

        chunks = dataset.get_filtered_files_in_dir("*.jsonl.gz")
        for annotation in self.model.annotate(chunks):
            if ignore_annotation(annotation.original_annotation):
                continue

            ordered_preds = sorted(
                annotation.predicted_annotation_logprob_dist,
                key=lambda x: annotation.predicted_annotation_logprob_dist[x],
                reverse=True,
            )[: self.topn]

            annotation_dict = annotation._asdict()
            logprobs = annotation_dict["predicted_annotation_logprob_dist"]
            filtered_logprobs = []
            for annot in ordered_preds:
                logprob = float(logprobs[annot])
                if annot == "%UNK%" or annot == "%UNKNOWN%":
                    annot = "typing.Any"
                filtered_logprobs.append((annot, logprob))

            filtered_logprobs.extend([("typing.Any", -1000)] * (self.topn - len(filtered_logprobs)))
            annotation_dict["predicted_annotation_logprob_dist"] = filtered_logprobs

            ps.append(annotation_dict)

        output = RichPath.create(str(predictions))
        output.save_as_compressed_file(ps)
        return output

    def annotate_and_collect(
        self,
        repo: pathlib.Path,
        subset: set[pathlib.Path],
        predictions: RichPath,
    ) -> pt.DataFrame[InferredSchema]:
        collections = [InferredSchema.example(size=0)]

        for topn in range(1, self.topn + 1):
            with utils.scratchpad(repo) as sc:
                anno_res = codemod.parallel_exec_transform_with_prettyprint(
                    transform=TypilusAnnotationApplier(
                        context=codemod.CodemodContext(),
                        predictions=predictions,
                        typing_rules=self.typing_rules,
                        topn=topn - 1,
                    ),
                    jobs=utils.worker_count(),
                    repo_root=str(sc),
                    files=[str(sc / p) for p in subset],
                )

                self.logger.info(
                    utils.format_parallel_exec_result(
                        f"Annotated with Typilus @ topn={topn}", result=anno_res
                    )
                )

                c = build_type_collection(root=sc, allow_stubs=False, subset=subset).df.assign(
                    topn=topn
                )
                collections.append(c)

        return (
            pd.concat(collections, ignore_index=True)
            .assign(method=self.method())
            .pipe(pt.DataFrame[InferredSchema])
        )


class _TypilusTopN(Typilus):
    def __init__(self, topn: int) -> None:
        super().__init__(model_folder=pathlib.Path.cwd() / "models" / "typilus", topn=topn)


class TypilusTop1(_TypilusTopN):
    def __init__(self) -> None:
        super().__init__(topn=1)


class TypilusTop3(_TypilusTopN):
    def __init__(self) -> None:
        super().__init__(topn=3)


class TypilusTop5(_TypilusTopN):
    def __init__(self) -> None:
        super().__init__(topn=5)


class TypilusTop10(_TypilusTopN):
    def __init__(self) -> None:
        super().__init__(topn=10)
