from scripts.common.schemas import TypeCollectionCategory
import seaborn as sns

import numpy as np
import pandas as pd
import sklearn.metrics as skm, sklearn.preprocessing as skp

import matplotlib.pyplot as plt

from typet5.model import ModelWrapper

# UBIQUITOUS_TYPES = ["str", "int", "List", "bool", "Dict"]
UBIQUITOUS_TYPES = ["str", "int", "bool", "float", "Dict"]

# print(f"{UBIQUITOUS_TYPES=}", f"{COMMON_TYPES=}", sep="\n")


def ubiq_mask(df: pd.DataFrame) -> pd.DataFrame:
    return df["trait_gt_form"].isin(UBIQUITOUS_TYPES)


def ubiquitous_types(df: pd.DataFrame) -> pd.DataFrame:
    return df[ubiq_mask(df)]


def common_mask(df: pd.DataFrame) -> pd.DataFrame:
    type_frequency = df["trait_gt_form"].value_counts(ascending=False)
    occurring_over_100 = type_frequency[type_frequency >= 100]
    common = df["trait_gt_form"].isin(occurring_over_100.index)
    # Exclude ubiq types from common types

    return common & ~ubiq_mask(df)


def common_types(df: pd.DataFrame) -> pd.DataFrame:
    return df[common_mask(df)]


def rare_mask(df: pd.DataFrame) -> pd.DataFrame:
    return ~ubiq_mask(df) & ~common_mask(df) & df["trait_gt_form"].notna()


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

    predictions = df[pred].rename("predictions")
    groundtruth = df[truth].rename("ground truth")

    pred_other_mask = ~predictions.isin(groundtruth)
    predictions[pred_other_mask] = "§OTHER"

    ct = pd.crosstab(
        index=predictions,
        columns=groundtruth,
        normalize="columns",
    ).sort_index(axis=0, ascending=True)
    below_threshold = ct[(ct >= threshold).any(axis=1)]

    plt.figure(figsize=figsize)
    sns.heatmap(below_threshold.T, annot=True, ax=ax)

    # plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    # plt.setp(ax.yaxis.get_majorticklabels(), rotation=90)
    # plt.tight_layout()


def performance(
    evaluatable: pd.DataFrame,
    *,
    ubiq_types: pd.Series,
    comm_types: pd.Series,
    rare_types: pd.Series,
    total: bool = False,
) -> pd.DataFrame:
    assert ubiq_types.notna().all()
    assert comm_types.notna().all()
    assert rare_types.notna().all()

    assert not (i := set(ubiq_types) & set(comm_types)), f"{i}"
    assert not (i := set(comm_types) & set(rare_types)), f"{i}"
    assert not (i := set(ubiq_types) & set(rare_types)), f"{i}"

    # Ensure masks do not overlap
    umask, cmask, rmask = (
        evaluatable["trait_gt_form"].isin(ubiq_types),
        evaluatable["trait_gt_form"].isin(comm_types),
        evaluatable["trait_gt_form"].isin(rare_types),
    )

    # Separate DataFrame into subclasses of regarded type classes
    complete_mask = umask + cmask + rmask
    assert (complete_mask == 1).all(), evaluatable[complete_mask != 1]
    evaluatable["class"] = np.select(
        [
            umask,
            cmask,
            rmask,
        ],
        choicelist=["ubiquitous", "common", "rare"],
        default="unknown",
    )
    assert (
        "unknown" not in evaluatable["class"]
    ), f"{evaluatable[evaluatable['class'] == 'unknown']}"

    evaluatable["match"] = evaluatable.gt_anno == evaluatable.anno

    groups = []
    for clazz, group in evaluatable.groupby(by="class"):
        observations = group.match.count()
        predictions = group.anno.count()

        unassigned = group.anno.isna().sum()
        matches = group.match.sum()

        accuracy = (group.gt_anno == group.anno).sum() / len(group)

        with_pred_mask = group.anno.notna()
        nohole_accuracy = skm.accuracy_score(
            y_true=group.gt_anno[with_pred_mask],
            y_pred=group.anno[with_pred_mask],
        )

        groups.append(
            pd.DataFrame(
                {
                    "observations": [observations],
                    "predictions": [predictions],
                    "unassigned": [unassigned],
                    "matches": [matches],
                    "stracc": [accuracy],
                    "relacc": [nohole_accuracy],
                },
                index=[clazz],
            )
        )

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

    """     perf = evaluatable.groupby(by="class").aggregate(
        observations=pd.NamedAgg(column="match", aggfunc="count"),
        predictions=pd.NamedAgg(column="anno", aggfunc="count"),
        unassigned=pd.NamedAgg(column="anno", aggfunc=lambda x: x.isna().sum()),
        matches=pd.NamedAgg(column="match", aggfunc="sum"),
        # sklearn perf metrics
        accuracy=pd.NamedAgg(
            column="match",
            aggfunc=lambda x: skm.accuracy_score(y_true=np.ones(x.size), y_pred=x),
        ),
    ) """
    perf = pd.concat(groups).reindex(["ubiquitous", "common", "rare"])

    if total:
        weighted_total = pd.DataFrame(
            {
                "observations": [perf.observations.sum()],
                "predictions": [perf.predictions.sum()],
                "unassigned": [perf.unassigned.sum()],
                "matches": [perf.matches.sum()],
                "stracc": [
                    perf.apply(
                        lambda row: row.stracc
                        * (row.observations / perf.observations.sum()),
                        axis=1,
                    ).sum()
                ],
                "relacc": [
                    perf.apply(
                        lambda row: row.relacc
                        * (row.predictions / perf.predictions.sum()),
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
            "stracc": float,
            "relacc": float,
        }
    )


CATEGORY_KEYS = {
    TypeCollectionCategory.CALLABLE_PARAMETER: "PARAMETER",
    TypeCollectionCategory.CALLABLE_RETURN: "RETURN",
    TypeCollectionCategory.VARIABLE: "VARIABLE",
}


def by_category_performance(
    evaluatable: pd.DataFrame,
    *,
    ubiq_types: pd.Series,
    comm_types: pd.Series,
    rare_types: pd.Series,
    total: bool = False,
):
    category_perf = []
    for category, group in evaluatable.groupby(by="category", sort=False):
        perf = performance(
            group,
            ubiq_types=ubiq_types,
            comm_types=comm_types,
            rare_types=rare_types,
            total=total,
        )
        category_perf.append(pd.concat([perf], keys=[CATEGORY_KEYS[category]], axis=0))

    all_perf = performance(
        evaluatable,
        ubiq_types=ubiq_types,
        comm_types=comm_types,
        rare_types=rare_types,
        total=total,
    )
    category_perf.append(pd.concat([all_perf], keys=["ALL"], axis=0))
    return pd.concat(category_perf)
