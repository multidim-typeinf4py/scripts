from scripts.common.schemas import TypeCollectionCategory
import seaborn as sns

import numpy as np
import pandas as pd
import sklearn.metrics as skm, sklearn.preprocessing as skp

import matplotlib.pyplot as plt

from typet5.model import ModelWrapper

UBIQUITOUS_TYPES = ["str", "int", "List", "bool", "float"]
# print(f"{UBIQUITOUS_TYPES=}", f"{COMMON_TYPES=}", sep="\n")


def ubiq_mask(df: pd.DataFrame) -> pd.DataFrame:
    return df["trait_gt_form"].isin(UBIQUITOUS_TYPES)


def ubiquitous_types(df: pd.DataFrame) -> pd.DataFrame:
    return df[ubiq_mask(df)]


def common_mask(df: pd.DataFrame) -> pd.DataFrame:
    most_common_types = df["trait_gt_form"].value_counts(ascending=False)
    top100 = df["trait_gt_form"].isin(most_common_types.head(100).index)
    # Exclude ubiq types from common types

    return top100 & ~ubiq_mask(df)


def common_types(df: pd.DataFrame) -> pd.DataFrame:
    return df[common_mask(df)]


def rare_mask(df: pd.DataFrame) -> pd.DataFrame:
    return ~ubiq_mask(df) & ~common_mask(df)


def rare_types(df: pd.DataFrame) -> pd.DataFrame:
    return df[rare_mask(df)]


def co_occurrences(
    df: pd.DataFrame,
    truth: str,
    pred: str,
    threshold: float,
    figsize: tuple[int, int],
    ax: plt.Axes,
    unsupported=list[TypeCollectionCategory](),
):
    df = df[~df.category.isin(unsupported)]
    ct = pd.crosstab(
        index=df[pred].rename("predictions"),
        columns=df[truth].rename("ground truth"),
        normalize="columns",
    ).sort_index(axis=0, ascending=True)
    below_threshold = ct[(ct >= threshold).any(axis=1)]

    plt.figure(figsize=figsize)
    sns.heatmap(below_threshold.T, annot=True, ax=ax)

    plt.setp(ax.yaxis.get_majorticklabels(), rotation=90)
    # plt.tight_layout()


def performance(evaluatable: pd.DataFrame, total: bool = False) -> pd.DataFrame:
    # Ensure masks do not overlap
    ut, ct, rt = (
        ubiq_mask(evaluatable),
        common_mask(evaluatable),
        rare_mask(evaluatable),
    )
    ts = ut + ct + rt
    print(ts[ts != 1])
    assert (ts == 1).all()

    # Separate DataFrame into subclasses of regarded type classes

    evaluatable["class"] = np.select(
        [
            ut,
            ct,
            rt,
            evaluatable["trait_gt_form"].isna(),
        ],
        choicelist=["ubiq", "common", "rare", "missing"],
        default="unknown",
    )
    assert "unknown" not in evaluatable["class"]

    evaluatable["match"] = evaluatable.gt_anno == evaluatable.anno

    groups = []
    for clazz, group in evaluatable.groupby(by="class"):
        ...
        # df = pd.DataFrame.from_dict(
        #     {
        #         "observations": group.match.count,
        #         "predictions": group.anno.count,
        #         "unassigned": group.anno.isna().sum(),
        #         "matches": group.match.sum(),
        #     }
        # )

        # sklearn perf metrics
        # df = df.assign(
        #     accuracy=skm.accuracy_score(y_true=group.gt_anno, y_pred=group.anno)
        # )

        # le = skp.LabelEncoder()
        # y = le.fit_transform(group.anno.to_numpy())

        # y = skp.label_binarize(
        #    y=group.anno, classes=list(range(1, group.anno.nunique() + 1))
        # )
        # y_score = group.probability
        # n_classes = y.shape[1]

        # precision, recall, average_precision = dict(), dict(), dict()
        # for i in range(n_classes):
        #    precision[i], recall[i], _ = skm.precision_recall_curve(
        #        y_true=y_test[:, i], probas_pred=y_score[:, i]
        #    )
        #    average_precision[i] = skm.average_precision_score(
        #        y_true=y_test[:, i], y_score=y_score[:, i]
        #    )

        # precision["micro"], recall["micro"], _ = skm.precision_recall_curve(
        #     y_true=y_test.ravel(), probas_pred=y_score.ravel()
        # )
        # average_precision["micro"] = skm.average_precision_score(
        #     y_true=y_test, y_score=y_score, average="micro"
        # )
        # df = df.assign(
        #     precision=precision["micro"],
        #     recall=recall["micro"],
        # )

        # groups.append(df)

    perf = evaluatable.groupby(by="class").aggregate(
        observations=pd.NamedAgg(column="match", aggfunc="count"),
        predictions=pd.NamedAgg(column="anno", aggfunc="count"),
        unassigned=pd.NamedAgg(column="anno", aggfunc=lambda x: x.isna().sum()),
        matches=pd.NamedAgg(column="match", aggfunc="sum"),
        # sklearn perf metrics
        accuracy=pd.NamedAgg(
            column="match",
            aggfunc=lambda x: skm.accuracy_score(y_true=np.ones(x.size), y_pred=x),
        ),
    )

    if total:
        weighted_total = pd.DataFrame(
            {
                "observations": [perf.observations.sum()],
                "predictions": [perf.predictions.sum()],
                "unassigned": [perf.unassigned.sum()],
                "matches": [perf.matches.sum()],
                "accuracy": [
                    perf.apply(
                        lambda row: row.accuracy
                        * (row.observations / perf.observations.sum()),
                        axis=1,
                    ).sum()
                ],
            },
            index=["total"],
        )
        perf = pd.concat([perf, weighted_total])

    return perf.astype(
        {
            "predictions": int,
            "observations": int,
            "unassigned": int,
            "matches": int,
            "accuracy": float,
        }
    )
