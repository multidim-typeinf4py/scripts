import pathlib

import pandas as pd
import pandera.typing as pt

from common.schemas import ContextSymbolSchema
from context.features import RelevantFeatures
from context.visitors import generate_context_vectors_for_file

import pytest


@pytest.fixture(scope="class")
def context_dataset() -> pt.DataFrame[ContextSymbolSchema]:
    repo = pathlib.Path.cwd() / "tests" / "context"
    cvs = generate_context_vectors_for_file(
        features=RelevantFeatures(
            loop=True, reassigned=True, nested=True, user_defined=True, branching=True
        ),
        repo=repo,
        path=repo / "resource.py",
    )

    print(cvs)
    return cvs


class TestFeatures:
    def test_loop(self, context_dataset: pt.DataFrame[ContextSymbolSchema]):
        self.exact_check(
            context_dataset, ["looping.x", "looping.a", "looping._", "branching.a", "branching.e"], ContextSymbolSchema.loop
        )

    def test_reassigned(self, context_dataset: pt.DataFrame[ContextSymbolSchema]):
        self.one_positive_check(
            context_dataset,
            ["looping.x", "looping.a", "local_reassign.c", "parammed.p", "a"],
            ContextSymbolSchema.reassigned,
        )
        self.one_negative_check(
            context_dataset,
            ["looping.x", "looping.a", "local_reassign.c", "parammed.p", "a"],
            ContextSymbolSchema.reassigned,
        )

    def test_nested(self, context_dataset: pt.DataFrame[ContextSymbolSchema]):
        self.exact_check(context_dataset, ["f.g", "f.g.a"], ContextSymbolSchema.nested)

    def test_userdeffed(self, context_dataset: pt.DataFrame[ContextSymbolSchema]):
        self.exact_check(context_dataset, ["userdeffed.udc"], ContextSymbolSchema.user_defined)

    def test_branching(self, context_dataset: pt.DataFrame[ContextSymbolSchema]):
        self.all_positive_check(context_dataset, ["branching.b"], ContextSymbolSchema.branching)
        self.all_negative_check(
            context_dataset,
            [
                "looping",
                "looping.x",
                "looping.a",
                "userdeffed",
                "userdeffed.abc",
                "userdeffed.efg",
                "userdeffed.udc",
                "local_reassign",
                "local_reassign.c",
                "f",
                "f.a",
                "f.g",
                "f.g.a",
                "g",
                "parammed",
                "parammed.p",
                "branching",
                "branching.x",
            ],
            ContextSymbolSchema.branching
        )
        self.one_positive_check(context_dataset, ["branching.a"], ContextSymbolSchema.branching)
        self.one_negative_check(context_dataset, ["branching.a"], ContextSymbolSchema.branching)

    def one_positive_check(
        self, context_dataset: pt.DataFrame[ContextSymbolSchema], qnames: list[str], feature: str
    ):
        """Check for at least one occurrence of a qname with a 1 set in the corresponding feature"""
        qnames_ser = pd.Series(qnames)
        present = qnames_ser.isin(context_dataset[ContextSymbolSchema.qname])
        assert present.all(), f"{qnames_ser[~present].unique()} missing from dataset!"

        for qname in qnames:
            select = context_dataset[ContextSymbolSchema.qname] == qname
            assert context_dataset.loc[
                select, feature
            ].any(), f"{qname} is not marked as '{feature}'"

    def one_negative_check(
        self, context_dataset: pt.DataFrame[ContextSymbolSchema], qnames: list[str], feature: str
    ):
        """Check for at least one occurrence of a qname with a 0 set in the corresponding feature"""
        qnames_ser = pd.Series(qnames)
        present = qnames_ser.isin(context_dataset[ContextSymbolSchema.qname])
        assert present.all(), f"{qnames_ser[~present].unique()} missing from dataset!"

        for qname in qnames:
            select = context_dataset[ContextSymbolSchema.qname] == qname
            assert not context_dataset.loc[
                select, feature
            ].all(), f"{qname} is should not be marked as '{feature}'"

    def all_positive_check(
        self, context_dataset: pt.DataFrame[ContextSymbolSchema], qnames: list[str], feature: str
    ):
        qnames_ser = pd.Series(qnames)
        present = qnames_ser.isin(context_dataset[ContextSymbolSchema.qname])

        assert present.all(), f"{qnames_ser[~present].unique()} missing from dataset!"

        mask = context_dataset[ContextSymbolSchema.qname].isin(qnames)
        positive_df = context_dataset[mask]

        pos_failing = positive_df[positive_df[feature] != 1]
        if not pos_failing.empty:
            pytest.fail(
                f"{pos_failing[ContextSymbolSchema.qname].unique()} are not marked as '{feature}'\n{pos_failing}"
            )

    def all_negative_check(
        self, context_dataset: pt.DataFrame[ContextSymbolSchema], qnames: list[str], feature: str
    ):
        qnames_ser = pd.Series(qnames)
        present = qnames_ser.isin(context_dataset[ContextSymbolSchema.qname])

        assert present.all(), f"{qnames_ser[~present].unique()} missing from dataset!"

        mask = context_dataset[ContextSymbolSchema.qname].isin(qnames)
        negative_df = context_dataset[mask]

        neg_failing = negative_df[negative_df[feature] != 0]
        assert (
            neg_failing.empty
        ), f"{neg_failing[ContextSymbolSchema.qname].unique()} are not marked as '{feature}'"

    def exact_check(
        self, context_dataset: pt.DataFrame[ContextSymbolSchema], qnames: list[str], feature: str
    ):
        qnames_ser = pd.Series(qnames)
        present = qnames_ser.isin(context_dataset[ContextSymbolSchema.qname])

        assert present.all(), f"{qnames_ser[~present].unique()} missing from dataset!"

        mask = context_dataset[ContextSymbolSchema.qname].isin(qnames)
        positive_df, negative_df = context_dataset[mask], context_dataset[~mask]

        errors = []

        pos_failing = positive_df[positive_df[feature] != 1]
        if not pos_failing.empty:
            errors.append(
                f"{pos_failing[ContextSymbolSchema.qname].unique()} are not marked as '{feature}'\n{pos_failing}"
            )

        neg_failing = negative_df[negative_df[feature] != 0]
        if not neg_failing.empty:
            errors.append(
                f"{neg_failing[ContextSymbolSchema.qname].unique()} should not be marked as '{feature}'\n{neg_failing}"
            )

        assert not errors, ",".join(errors)


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


# def test_tuple_handling(tuple_dataset: pt.DataFrame[ContextSymbolSchema]):
# print(tuple_dataset)
# assert False
