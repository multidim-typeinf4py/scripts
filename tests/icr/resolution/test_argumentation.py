from common.schemas import TypeCollectionCategory, InferredSchema
from infer.resolution._base import Metadata
from infer.resolution.voting import (
    SubtypeVoting,
    Majority,
    OF,
    SF,
    CF,
    build_discussion_from_predictions,
    compute_collective_decision,
    compute_collective_labelling,
    draw_profile,
    const,
)

import pandera.typing as pt
import pytest


class IntSubclass(int):
    ...


class StrSubclass(str):
    ...


intsc_qualname = IntSubclass.__module__ + "." + IntSubclass.__qualname__
strsc_qualname = StrSubclass.__module__ + "." + StrSubclass.__qualname__


@pytest.mark.parametrize(
    argnames=["stat_preds", "dyn_preds", "prob_preds", "correct"],
    argvalues=[
        ## More than one approach voted for the same type; capture it
        (["int"], ["bytearray"], ["str", "int", "float"], "int"),
        ## All approaches predict differently, but subtyping was found
        (["bytes"], ["int"], ["str", "float", intsc_qualname], "int"),
        ## Same derived type AND base type with emphasis on base type
        (["str"], ["str"], ["bool", "float", strsc_qualname], "str"),
        ## Same derived type AND base type with emphasis on derived
        (["str"], [strsc_qualname], ["bool", "float", strsc_qualname], "str"),
    ],
    ids=[
        "simple majority vote",
        "simple subtype majority",
        "multiple base type + single derived type",
        "single base type + multiple derived type",
    ],
)
def test_discussion_building(
    stat_preds: list[str], dyn_preds: list[str], prob_preds: list[str], correct: str
):

    # static inference infers type int
    static = pt.DataFrame[InferredSchema](
        {
            "method": ["static"] * len(stat_preds),
            "file": ["x.py"] * len(stat_preds),
            "category": [TypeCollectionCategory.VARIABLE] * len(stat_preds),
            "qname": ["x"] * len(stat_preds),
            "qname_ssa": ["xλ1"] * len(stat_preds),
            "anno": stat_preds,
            "topn": list(range(len(stat_preds)))
        }
    )

    # dynamic inference infers bytestring
    dynamic = pt.DataFrame[InferredSchema](
        {
            "method": ["dynamic"] * len(dyn_preds),
            "file": ["x.py"] * len(dyn_preds),
            "category": [TypeCollectionCategory.VARIABLE] * len(dyn_preds),
            "qname": ["x"] * len(dyn_preds),
            "qname_ssa": ["xλ1"] * len(dyn_preds),
            "anno": dyn_preds,
            "topn": list(range(len(dyn_preds)))
        }
    )

    # probabilistic inference infers top-3, including subtype of int in place 2
    prob = pt.DataFrame[InferredSchema](
        {
            "method": ["prob"] * len(prob_preds),
            "file": ["x.py"] * len(prob_preds),
            "category": [TypeCollectionCategory.VARIABLE] * len(prob_preds),
            "qname": ["x"] * len(prob_preds),
            "qname_ssa": ["xλ1"] * len(prob_preds),
            "anno": prob_preds,
            "topn": list(range(len(prob_preds)))
        }
    )

    prediction = SubtypeVoting(project=None, reference=None).forward(
        static=static,
        dynamic=dynamic,
        probabilistic=prob,
        metadata=Metadata(file="x.py", category=TypeCollectionCategory.VARIABLE, qname="x", qname_ssa="xλ1"),
    )

    assert prediction is not None
    assert len(prediction) == 1
    assert prediction["anno"].iloc[0] == correct
