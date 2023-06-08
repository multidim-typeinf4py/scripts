import functools
import json
import pathlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import pandera.typing as pt
from data_preparation.scripts.graph_generator import extract_graphs
from dpu_utils.utils import RichPath
from typilus.model import model_restore_helper
from typilus.utils.predict import ignore_annotation

from scripts.common.schemas import InferredSchema
from scripts.infer.annotators.typilus import TypilusProjectApplier
from scripts.infer.inference._base import ProjectWideInference


class Typilus(ProjectWideInference):
    def __init__(
        self,
        model_folder: pathlib.Path,
        topn: int,
        cpu_executor: ProcessPoolExecutor | None = None,
        model_executor: ThreadPoolExecutor | None = None,
    ) -> None:
        super().__init__(cpu_executor=cpu_executor, model_executor=model_executor)

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
        return f"typilusN{self.topn}"

    def _infer_project(
        self, mutable: pathlib.Path, subset: set[pathlib.Path]
    ) -> pt.DataFrame[InferredSchema]:
        # Transform repo into test dataset
        test_dataset_path = self.repo_to_dataset(mutable)

        # Predict over transformed dataset
        pred_path = self.predict(
            dataset=test_dataset_path,
            predictions_out=mutable / "typilus-predictions.json.gz",
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

    def predict(self, dataset: RichPath, predictions_out: pathlib.Path) -> RichPath:
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

            filtered_logprobs.extend(
                [("typing.Any", -1000)] * (self.topn - len(filtered_logprobs))
            )
            annotation_dict["predicted_annotation_logprob_dist"] = filtered_logprobs

            ps.append(annotation_dict)

        output = RichPath.create(str(predictions_out))
        output.save_as_compressed_file(ps)
        return output

    def annotate_and_collect(
        self,
        repo: pathlib.Path,
        subset: set[pathlib.Path],
        predictions: RichPath,
    ) -> pt.DataFrame[InferredSchema]:
        return TypilusProjectApplier.collect_topn(
            project=repo,
            subset=subset,
            predictions=predictions,
            topn=self.topn,
            tool=self,
            typing_rules=self.typing_rules,
        )


class TypilusTopN(Typilus):
    def __init__(
        self,
        topn: int,
        cpu_executor: ProcessPoolExecutor | None = None,
        model_executor: ThreadPoolExecutor | None = None,
    ) -> None:
        super().__init__(
            model_folder=pathlib.Path("models/typilus"),
            topn=topn,
            cpu_executor=cpu_executor,
            model_executor=model_executor,
        )


TypilusTop1 = functools.partial(TypilusTopN, topn=1)
TypilusTop3 = functools.partial(TypilusTopN, topn=3)
TypilusTop5 = functools.partial(TypilusTopN, topn=5)
TypilusTop10 = functools.partial(TypilusTopN, topn=10)
