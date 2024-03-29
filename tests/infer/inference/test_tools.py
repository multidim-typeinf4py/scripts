import pandas as pd
import pytest

from src.common.schemas import InferredSchema, TypeCollectionSchema

from src.infer.inference import PyreInfer, PyreQuery, MyPy
from src.infer.inference.hitypewriter import HiTypewriterTop3
from src.infer.inference.t4py import Type4PyTop3
from src.infer.inference.typewriter import TypeWriterTop3
from src.infer.inference.hit4py import HiType4Py3
from src.infer.inference.hitypilus import HiTypilusTop3
from src.infer.inference.tt5 import TypeT5Top3
from src.infer.inference.typilus import TypilusTop3
from src.infer.inference import Inference

from ._utils import Project, ProjectSubset, example_project, example_project_subset

tools = [
    (PyreInfer(), 1),
    (PyreQuery(), 1),
    (MyPy(), 1),
    (Type4PyTop3(), 3),
    (TypeWriterTop3(), 3),
    (TypeT5Top3(), 3),
    (TypilusTop3(), 3),
    (HiTypilusTop3(), 3),
    (HiTypewriterTop3(), 3),
    (HiType4Py3(), 3),
]


@pytest.mark.parametrize(
    argnames=["tool", "topn"],
    argvalues=tools,
    ids=list(map(lambda t: type(t[0]).__name__, tools)),
)
def test_full_inference(tool: Inference, topn: int, example_project: Project) -> None:
    inferred = tool.infer(mutable=example_project.mutable, readonly=example_project.readonly)

    print(inferred)
    assert not inferred.empty

    assert topn in inferred[InferredSchema.topn]
    assert (inferred[InferredSchema.topn] <= topn).all()

    # Show differences between top-2
    if topn > 1:
        top1, top2 = (
            inferred[inferred[InferredSchema.topn] == 1].drop(columns=InferredSchema.topn),
            inferred[inferred[InferredSchema.topn] == 2].drop(columns=InferredSchema.topn),
        )

        diff = pd.merge(top1, top2, how='outer', suffixes=['_1', '_2'], indicator=True)
        print(diff)


@pytest.mark.parametrize(
    argnames=["tool", "topn"],
    argvalues=tools,
    ids=list(map(lambda t: type(t[0]).__name__, tools)),
)
def test_subset_inference(
    tool: Inference, topn: int, example_project_subset: ProjectSubset
) -> None:
    inferred = tool.infer(
        mutable=example_project_subset.mutable,
        readonly=example_project_subset.readonly,
        subset=example_project_subset.subset,
    )
    print(inferred)
    assert not inferred.empty

    assert (
        inferred[TypeCollectionSchema.file].isin(set(map(str, example_project_subset.subset))).all()
    )

    assert topn in inferred[InferredSchema.topn]
    assert (inferred[InferredSchema.topn] <= topn).all()
