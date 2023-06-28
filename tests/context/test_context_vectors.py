import pathlib

import pandas as pd
import pandera.typing as pt
import pytest

from scripts.common.schemas import ContextCategory, ContextSymbolSchema
from scripts.context import RelevantFeatures
from scripts.context.visitors import generate_context_vectors


@pytest.fixture(scope="class")
def context_dataset() -> pt.DataFrame[ContextSymbolSchema]:
    repo = pathlib.Path.cwd() / "tests" / "resources" / "context"
    filepath = pathlib.Path("x.py")

    cvs = generate_context_vectors(
        features=RelevantFeatures.default(),
        project=repo,
        subset={filepath},
    )

    print(cvs)
    return cvs


class TestFeatures:
    def test_loop(self, context_dataset: pt.DataFrame[ContextSymbolSchema]):
        self.exact_check(
            context_dataset,
            ["looping.x", "looping.a", "looping._", "branching.a", "branching.e", "categories.xs"],
            ContextSymbolSchema.loop,
        )

    def test_reassigned(self, context_dataset: pt.DataFrame[ContextSymbolSchema]):
        self.one_positive_check(
            context_dataset,
            ["looping.x", "looping.a", "local_reassign.c", "parammed.p", "a"],
            ContextSymbolSchema.reassigned,
        )
        self.one_negative_check(
            context_dataset,
            ["looping.x", "looping.a", "local_reassign.c", "parammed.p"],
            ContextSymbolSchema.reassigned,
        )

    def test_nested(self, context_dataset: pt.DataFrame[ContextSymbolSchema]):
        self.exact_check(context_dataset, ["f.g", "f.g.a"], ContextSymbolSchema.nested)

    def test_userdeffed(self, context_dataset: pt.DataFrame[ContextSymbolSchema]):
        self.exact_check(
            context_dataset,
            ["userdeffed.abc", "userdeffed.efg", "parammed.p", "categories.b"],
            ContextSymbolSchema.builtin_source,
        )

    def test_branching(self, context_dataset: pt.DataFrame[ContextSymbolSchema]):
        self.all_positive_check(context_dataset, ["branching.b"], ContextSymbolSchema.flow_control)
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
            ContextSymbolSchema.flow_control,
        )
        self.one_positive_check(context_dataset, ["branching.a"], ContextSymbolSchema.flow_control)
        self.one_negative_check(context_dataset, ["branching.a"], ContextSymbolSchema.flow_control)

    def test_categories(self, context_dataset: pt.DataFrame[ContextSymbolSchema]):
        qnames = context_dataset[ContextSymbolSchema.qname]

        categories = context_dataset[qnames == "categories"]
        assert len(categories) == 1
        assert (categories[ContextSymbolSchema.context_category] == ContextCategory.CALLABLE_RETURN).all()

        categoriesx = context_dataset[qnames == "categories.x"]
        assert len(categoriesx) == 1
        assert (categoriesx[ContextSymbolSchema.context_category] == ContextCategory.CALLABLE_PARAMETER).all()

        categoriesa = context_dataset[qnames == "categories.a"]
        assert len(categoriesa) == 1
        assert (categoriesa[ContextSymbolSchema.context_category] == ContextCategory.SINGLE_TARGET_ASSIGN).all()

        categoriesb = context_dataset[qnames == "categories.b"]
        assert len(categoriesb) == 1
        assert (categoriesb[ContextSymbolSchema.context_category] == ContextCategory.ANN_ASSIGN).all()

        categoriesc = context_dataset[qnames == "categories.c"]
        assert len(categoriesc) == 1
        assert (categoriesc[ContextSymbolSchema.context_category] == ContextCategory.AUG_ASSIGN).all()

        categoriesd = context_dataset[qnames == "categories.d"]
        assert len(categoriesd) == 1
        assert (categoriesd[ContextSymbolSchema.context_category] == ContextCategory.MULTI_TARGET_ASSIGN).all()

        categoriese = context_dataset[qnames == "categories.C.e"]
        assert len(categoriese) == 1
        assert (categoriese[ContextSymbolSchema.context_category] == ContextCategory.INSTANCE_ATTRIBUTE).all()

        categoriesxs = context_dataset[qnames == "categories.xs"]
        assert len(categoriesxs) == 1
        assert (categoriesxs[ContextSymbolSchema.context_category] == ContextCategory.FOR_TARGET).all()

        categoriesf = context_dataset[qnames == "categories.f"]
        assert len(categoriesf) == 1
        assert (categoriesf[ContextSymbolSchema.context_category] == ContextCategory.WITH_TARGET).all()

    def one_positive_check(
        self,
        context_dataset: pt.DataFrame[ContextSymbolSchema],
        qnames: list[str],
        feature: str,
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
        self,
        context_dataset: pt.DataFrame[ContextSymbolSchema],
        qnames: list[str],
        feature: str,
    ):
        """Check for at least one occurrence of a qname with a 0 set in the corresponding feature"""
        qnames_ser = pd.Series(qnames)
        present = qnames_ser.isin(context_dataset[ContextSymbolSchema.qname])
        assert present.all(), f"{qnames_ser[~present].unique()} missing from dataset!"

        for qname in qnames:
            select = context_dataset[ContextSymbolSchema.qname] == qname
            assert not context_dataset.loc[
                select, feature
            ].all(), f"{qname} should not be marked as '{feature}'"

    def all_positive_check(
        self,
        context_dataset: pt.DataFrame[ContextSymbolSchema],
        qnames: list[str],
        feature: str,
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
        self,
        context_dataset: pt.DataFrame[ContextSymbolSchema],
        qnames: list[str],
        feature: str,
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
        self,
        context_dataset: pt.DataFrame[ContextSymbolSchema],
        qnames: list[str],
        feature: str,
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


# def test_tuple_handling(tuple_dataset: pt.DataFrame[ContextSymbolSchema]):
# print(tuple_dataset)
# assert False
