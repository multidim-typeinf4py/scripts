import seaborn as sns

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

UBIQUITOUS_TYPES = ["str", "int", "List", "bool", "float"]


def ubiquitous(df: pd.DataFrame) -> pd.DataFrame:
    ubiq_mask = df["gt_anno"].isin(UBIQUITOUS_TYPES)
    print(ubiq_mask.sum(), "/", len(ubiq_mask), "ground truth labels are ubiquitous")

    return df[ubiq_mask]


def co_occurrences(df: pd.DataFrame, truth: str, pred: str, threshold: float, figsize: tuple[int, int], ax: plt.Axes):
    ct = pd.crosstab(index=df[pred].rename("predictions"), columns=df[truth].rename("ground truth"),
                     normalize="columns").sort_index(axis=0, ascending=True)
    below_threshold = ct[(ct >= threshold).any(axis=1)]

    plt.figure(figsize=figsize)
    sns.heatmap(below_threshold.T, annot=True, ax=ax)

    # plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    # plt.tight_layout()