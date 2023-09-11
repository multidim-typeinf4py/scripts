import abc
import collections
import math
import pathlib
import typing

from tqdm import tqdm

import pandas as pd

import pandera as pa
import pandera.typing as pt

from typet5.static_analysis import ProjectPath
from typet5.type_env import annot_path

from scripts.common.output import InferenceArtifactIO
from scripts.common.schemas import (
    InferredSchema,
    TypeCollectionCategory,
    RepositoryInferredSchema,
)
from scripts.infer.structure import DatasetFolderStructure


class InferredWithProbasSchema(RepositoryInferredSchema):
    probability: pt.Series[float] = pa.Field(nullable=True, coerce=False)


class ProbabilityJoiner(abc.ABC):
    def find_probability(
        self,
        probs: dict[typing.Any, float],
        file: str,
        category: TypeCollectionCategory,
        qname: str,
        qname_ssa: str,
    ) -> float:
        ...


class Type4PyJoiner(ProbabilityJoiner):
    def find_probability(
        self,
        probs: dict[str, dict[str, float]],
        file: str,
        category: TypeCollectionCategory,
        qname: str,
        qname_ssa: str,
    ) -> float:
        if (qname2probs := probs.get(file)) is None:
            # print(f"WARNING: MISSING FILE: {file}")
            return pd.NA
        elif (prob := qname2probs.get(qname)) is None:
            return pd.NA
        return prob


class TypilusJoiner(ProbabilityJoiner):
    def find_probability(
        self,
        probs: dict[str, dict[str, float]],
        file: str,
        category: TypeCollectionCategory,
        qname: str,
        qname_ssa: str,
    ) -> float:
        if (qname2probs := probs.get(file)) is None:
            # print(f"WARNING: MISSING FILE: {file}")
            return pd.NA
        elif (prob := qname2probs.get(qname_ssa)) is not None:
            return prob
        return qname2probs.get(qname, pd.NA)


class TypeT5Joiner(ProbabilityJoiner):
    def find_probability(
        self,
        probs: dict[ProjectPath, float],
        file: str,
        category: TypeCollectionCategory,
        qname: str,
        qname_ssa: str,
    ) -> float:
        if category is TypeCollectionCategory.CALLABLE_PARAMETER:
            key = ".".join(qname.split(".")[:-1])
        else:
            key = qname

        path = ProjectPath.from_annot_path(
            rel_path=pathlib.Path(file), p=annot_path(*key.split("."))
        )
        if path not in probs:
            print(f"WARNING: Could not find {path}")

        prob = probs.get(path, pd.NA)
        # print(file, qname, "->", prob)
        return prob


class ProbabilityLoader(abc.ABC):
    def __init__(
        self,
        artifact_root: pathlib.Path,
        dataset: DatasetFolderStructure,
        task: TypeCollectionCategory | str,
    ) -> None:
        self.artifact_root = artifact_root
        self.dataset = dataset
        self.task = task

    @abc.abstractmethod
    def load_probabilities(self, repository: pathlib.Path) -> dict[typing.Any, float]:
        ...


class Type4PyProbabilityLoader(ProbabilityLoader):
    def load_probabilities(
        self, repository: pathlib.Path
    ) -> dict[tuple[str, str], float]:
        artifact = InferenceArtifactIO(
            artifact_root=self.artifact_root,
            dataset=self.dataset,
            repository=repository,
            tool_name="type4pyN1",
            task=self.task,
        )

        predictions: dict[pathlib.Path, dict]
        (predictions,) = artifact.read()

        probas = dict[tuple[str, str], float]()
        for path, ps in predictions.items():
            path = str(path)

            for clazz in ps.get("classes", []):
                cqname = clazz["q_name"]
                for method in clazz.get("funcs", []):
                    mqname = method["q_name"]

                    if "ret_type_p" in method and len(method["ret_type_p"]) >= 1:
                        ret_pred, ret_prob = method["ret_type_p"][0]
                        probas[(path, mqname)] = ret_prob

                    for param, pred in method.get("params_p", {}).items():
                        if len(pred) < 1:
                            continue
                        param_pred, param_prob = pred[0]
                        probas[(path, f"{mqname}.{param}")] = param_prob

                    for variable, pred in method.get("variables_p", {}).items():
                        if len(pred) < 1:
                            continue
                        var_pred, var_prob = pred[0]
                        probas[(path, f"{mqname}.{variable}")] = var_prob

                for variable, pred in clazz.get("variables_p", {}).items():
                    if len(pred) < 1:
                        continue
                    var_pred, var_prob = pred[0]
                    probas[(path, f"{cqname}.{variable}")] = var_prob

            for func in ps.get("funcs", []):
                fqname = func["q_name"]

                if "ret_type_p" in func and len(func["ret_type_p"]) >= 1:
                    ret_pred, ret_prob = func["ret_type_p"][0]
                    probas[(path, fqname)] = ret_prob

                for param, pred in func.get("params_p", {}).items():
                    if len(pred) < 1:
                        continue
                    param_pred, param_prob = pred[0]
                    probas[(path, f"{fqname}.{param}")] = param_prob

                for variable, pred in func.get("variables_p", {}).items():
                    if len(pred) < 1:
                        continue
                    var_pred, var_prob = pred[0]
                    probas[(path, f"{fqname}.{variable}")] = var_prob

            for variable, pred in ps.get("variables_p", {}).items():
                if len(pred) < 1:
                    continue
                var_pred, var_prob = pred[0]
                probas[(path, variable)] = var_prob

        transformed = collections.defaultdict[str, dict[str, float]](dict)

        for (path, qname), prob in probas.items():
            transformed[path].update({qname.replace(".<locals>.", "."): prob})

        return transformed


class TypilusProbabilityLoader(ProbabilityLoader):
    def load_probabilities(
        self, repository: pathlib.Path
    ) -> dict[str, dict[str, float]]:
        import codecs, gzip, io, json

        artifact = InferenceArtifactIO(
            artifact_root=self.artifact_root,
            dataset=self.dataset,
            repository=repository,
            tool_name="typilusN1",
            task=self.task,
        )
        (predictions,) = artifact.read()


        mapped = collections.defaultdict[str, dict[str, float]](dict)
        for file, preds in predictions.items():
            qname_counter = collections.Counter()

            for pred in preds:
                qname = pred["qname"]
               # print(pred["annotation_type"], qname)

                if pred["annotation_type"] == "variable":
                    qname_counter.update((qname))
                    qname_ssa = f"{qname}λ{qname_counter.get(qname, 1)}"
                else:
                    qname_ssa = qname
                
                _, type_prob = pred["predicted_annotation_logprob_dist"][0]
                mapped[file].update({qname_ssa: math.exp(type_prob)})

        return mapped


class TypeT5ProbabilityLoader(ProbabilityLoader):
    def load_probabilities(self, repository: pathlib.Path) -> dict[ProjectPath, float]:
        artifact = InferenceArtifactIO(
            artifact_root=self.artifact_root,
            dataset=self.dataset,
            repository=repository,
            tool_name="TypeT5TopN1",
            task=self.task,
        )

        probabilities: dict[ProjectPath, float]
        _, probabilities = artifact.read()

        return {key: math.exp(logit) for key, logit in probabilities.items()}


class ArtifactToProbabilityPipeline:
    def __init__(self, loader: ProbabilityLoader, joiner: ProbabilityJoiner) -> None:
        self.loader = loader
        self.joiner = joiner

    def pipe(
        self,
        inferred: pt.DataFrame[RepositoryInferredSchema],
        dataset: DatasetFolderStructure,
    ) -> pt.DataFrame[InferredWithProbasSchema]:
        test_cases = {str(dataset.author_repo(p)): p for p in dataset.test_set()}

        assert not inferred.empty
        inferred_with_probas = (
            inferred.groupby(
                by=RepositoryInferredSchema.repository,
                dropna=False,
                as_index=False,
            )
            .apply(
                lambda x: x.assign(
                    probability=self._join_probabilities(x.name, x, test_cases)
                )
            )
            .droplevel(level=0)
        )
        # print(
        #    inferred_with_probas[
        #        ["repository", "file", "qname", "qname_ssa", "anno", "probability"]
        #    ]
        # )
        return inferred_with_probas

    def _join_probabilities(
        self,
        repository: str,
        inference: pt.DataFrame[InferredSchema],
        test_cases: dict[str, pathlib.Path],
    ) -> list[float]:
        probas = list[float]()

        # print(repository, inference, sep="\n")
        probabilities = self.loader.load_probabilities(test_cases[repository])

        for idx, file, category, qname, qname_ssa, anno in inference[
            [
                InferredSchema.file,
                InferredSchema.category,
                InferredSchema.qname,
                InferredSchema.qname_ssa,
                InferredSchema.anno,
            ]
        ].itertuples(index=True):
            if pd.notna(anno):
                prob = self.joiner.find_probability(
                    probabilities, file, category, qname, qname_ssa
                )
            else:
                prob = pd.NA

            probas.append(prob)
        return probas


def load_inferred_with_probablities(
    artifact_root: pathlib.Path,
    dataset: DatasetFolderStructure,
    tool_name: str,
    task: TypeCollectionCategory | str,
    inferred: pt.DataFrame[RepositoryInferredSchema],
) -> pt.DataFrame[InferredWithProbasSchema]:
    if tool_name == "typet5":
        loader, joiner = (
            TypeT5ProbabilityLoader(artifact_root, dataset, task),
            TypeT5Joiner(),
        )

    elif tool_name == "type4py":
        loader, joiner = (
            Type4PyProbabilityLoader(artifact_root, dataset, task),
            Type4PyJoiner(),
        )

    elif tool_name == "typilus":
        loader, joiner = (
            TypilusProbabilityLoader(artifact_root, dataset, task),
            TypilusJoiner(),
        )

    else:
        assert f"Unknown tool: {tool_name}"

    pipeline = ArtifactToProbabilityPipeline(loader, joiner)
    with_probabilities = pipeline.pipe(inferred, dataset)

    print(
        "Valid:",
        (
            (with_probabilities.probability.notna() & with_probabilities.anno.notna())
            | (with_probabilities.probability.isna() & with_probabilities.anno.isna())
        ).sum(),
        "/",
        len(with_probabilities.anno),
    )

    print(
        "Annotated & Has Probability:",
        (
            with_probabilities.probability.notna() & with_probabilities.anno.notna()
        ).sum(),
        "/",
        with_probabilities.anno.notna().sum(),
    )

    suspect = with_probabilities[
        with_probabilities.probability.notna() & with_probabilities.anno.isna()
    ]
    print(
        len(suspect),
        "entries with a probability but no annotation",
    )

    if not suspect.empty:
        print(
            suspect[["repository", "file", "category", "qname", "anno", "probability"]]
        )

    suspect = with_probabilities[
        with_probabilities.probability.isna() & with_probabilities.anno.notna()
    ]
    print(len(suspect), "entries with an annotation but no probability")

    if not suspect.empty:
        print(
            suspect[["repository", "file", "category", "qname", "anno", "probability"]]
        )
        print(suspect[["repository", "file"]].drop_duplicates())

    return with_probabilities
