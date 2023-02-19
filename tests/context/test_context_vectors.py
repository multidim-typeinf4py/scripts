import pathlib

import pandas as pd
import pandera.typing as pt

from common.schemas import ContextSymbolSchema
from context.features import RelevantFeatures
from context.visitors import generate_context_vectors_for_file

import pytest


@pytest.fixture
def context_dataset() -> pt.DataFrame[ContextSymbolSchema]:
    repo = pathlib.Path.cwd() / "tests" / "context"
    cvs = generate_context_vectors_for_file(
        features=RelevantFeatures(
            loop=True, reassigned=True, nested=True, user_defined=True, branching=True
        ),
        repo=repo,
        path=repo / "resource.py",
    )

    return cvs


@pytest.mark.parametrize(
    argnames=("qnames", "feature"),
    argvalues=(
        # 1. loopage i.e. the annotatable is in some kind of loop
        (["looping.x", "looping.a", "branching.a"], ContextSymbolSchema.loop),
        # 2. nestage i.e. the annotatble is in some nested scope (class in class, function in function)
        (["f.g", "f.g.a"], ContextSymbolSchema.nested),
        # 3. user-deffed i.e. the attached annotation is not a builtin type
        (["userdeffed.udc"], ContextSymbolSchema.user_defined),
        # 4. reassigned i.e. the annotatable's symbol occurs multiple times in the same scope
        (
            ["looping.x", "looping.a", "local_reassign.c", "parammed.p", "a"],
            ContextSymbolSchema.reassigned,
        ),
        (["branching.b", "branching.a"], ContextSymbolSchema.branching),
    ),
    ids=str,
)
def test_feature(
    context_dataset: pt.DataFrame[ContextSymbolSchema], qnames: list[str], feature: str
):
    assert pd.Series(qnames).isin(context_dataset[ContextSymbolSchema.qname]).all()

    mask = context_dataset[ContextSymbolSchema.qname].isin(qnames)
    positive_df, negative_df = context_dataset[mask], context_dataset[~mask]

    pos_failing = positive_df[positive_df[feature] != 1]
    assert (
        pos_failing.empty
    ), f"{pos_failing[ContextSymbolSchema.qname].unique()} are not marked as '{feature}'"

    neg_failing = negative_df[negative_df[feature] != 0]
    assert (
        neg_failing.empty
    ), f"{neg_failing[ContextSymbolSchema.qname].unique()} shouldn't be marked as '{feature}'"


@pytest.fixture
def tuple_dataset() -> pt.DataFrame[ContextSymbolSchema]:
    repo = pathlib.Path.cwd() / "tests" / "context"
    cvs = generate_context_vectors_for_file(
        features=RelevantFeatures(
            loop=True, reassigned=True, nested=True, user_defined=True, branching=True
        ),
        repo=repo,
        path=repo / "tuples.py",
    )

    return cvs


def test_tuple_handling(tuple_dataset: pt.DataFrame[ContextSymbolSchema]):
    print(tuple_dataset)
    assert False
