import pathlib

import libcst
import pandas as pd
import pandera.typing as pt

from common.schemas import ContextSymbolSchema, ContextSymbolSchemaColumns
from context.features import RelevantFeatures
from context.visitors import generate_context_vectors_for_file

import pytest


@pytest.fixture
def context_dataset() -> pt.DataFrame[ContextSymbolSchema]:
    repo = pathlib.Path.cwd() / "tests" / "context"
    return generate_context_vectors_for_file(
        features=RelevantFeatures(loop=True, reassigned=True, nested=True, user_defined=True),
        repo=repo,
        path=repo / "resource.py",
    )


# 1. loopage i.e. the annotatable is in some kind of loop
def test_loopage(context_dataset: pt.DataFrame[ContextSymbolSchema]):
    x = "looping.x"
    a = "looping.a"

    expected_in_loop = (x, a)

    for v in expected_in_loop:
        select = context_dataset[context_dataset["qname"] == v]
        assert select[ContextSymbolSchema.loop].all(), f"{v} is not in a loop!; {select}"

    remainder = context_dataset[~context_dataset["qname"].isin(expected_in_loop)]
    in_loop = remainder[remainder[ContextSymbolSchema.loop] == 1]
    assert in_loop.empty, f"{in_loop} are marked as in a loop!"


# 2. nestage i.e. the annotatble is in some nested scope (class in class, function in function)
def test_nestage(context_dataset: pt.DataFrame[ContextSymbolSchema]):
    fdotg = "f.g"
    fdotgdota = "f.g.a"

    expected_nested = (fdotg, fdotgdota)

    for v in expected_nested:
        select = context_dataset[context_dataset["qname"] == v]
        assert select[ContextSymbolSchema.nested].all(), f"{v} is not marked as nested!; {select}"

    remainder = context_dataset[~context_dataset["qname"].isin(expected_nested)]
    nested = remainder[remainder[ContextSymbolSchema.nested] == 1]
    assert nested.empty, f"{nested} are marked as nested!"


# 3. user-deffed i.e. the attached annotation is not a builtin type
def test_userdeffed(context_dataset: pt.DataFrame[ContextSymbolSchema]):
    udc = "userdeffed.udc"

    expected_userdeffed = (udc,)

    for v in expected_userdeffed:
        select = context_dataset[context_dataset["qname"] == v]
        assert select[
            ContextSymbolSchema.user_defined
        ].all(), f"{v} is not marked as user defined!; {select}"

    remainder = context_dataset[~context_dataset["qname"].isin(expected_userdeffed)]
    nested = remainder[remainder[ContextSymbolSchema.user_defined] == 1]
    assert nested.empty, f"{nested} are marked as user defined!"


# 4. reassigned i.e. the annotatable's symbol occurs multiple times in the same scope
def test_reassigned(context_dataset: pt.DataFrame[ContextSymbolSchema]):
    expected_reassigned = ("looping.x", "looping.a", "local_reassign.c", "a", "g.a")

    for v in expected_reassigned:
        select = context_dataset[context_dataset["qname"] == v]
        assert select[
            ContextSymbolSchema.reassigned
        ].all(), f"{v} is not marked as reassigned!; {select}"

    remainder = context_dataset[~context_dataset["qname"].isin(expected_reassigned)]
    nested = remainder[remainder[ContextSymbolSchema.user_defined] == 1]
    assert nested.empty, f"{nested} are marked as reassigned!"
