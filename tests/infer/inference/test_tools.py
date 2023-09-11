import pandas as pd
import pytest

from scripts.common.schemas import InferredSchema, TypeCollectionSchema, TypeCollectionCategory

from scripts.infer.inference import PyreInfer, PyreQuery, MyPy, _utils
#from scripts.infer.inference.hitypewriter import HiTypeWriterTop3
from scripts.infer.inference.t4py import Type4PyTop3
#from scripts.infer.inference.typewriter import TypeWriterTop3
from scripts.infer.inference.hit4py import HiType4PyTop3
from scripts.infer.inference.hitypilus import HiTypilusTop3
#from scripts.infer.inference.tt5 import TypeT5Top3
from scripts.infer.inference.typilus import TypilusTop1
from scripts.infer.inference import Inference

from ._utils import (
    Project,
    ProjectSubset,
    example_project,
    example_project_subset,
    preprocess_project,
)

ce, me = _utils.cpu_executor(), _utils.model_executor()

tools = [
    #(PyreInfer(), 1),
    #(PyreQuery(), 1),
    #(MyPy(), 1),
    #(Type4PyTop3(cpu_executor=ce, model_executor=me), 3),
    #(TypeWriterTop3(cpu_executor=ce, model_executor=me), 3),
    #(TypeT5Top3(), 3),
    (TypilusTop1(), 1),
    #(HiTypilusTop3(), 3),
    #(HiTypeWriterTop3(cpu_executor=ce, model_executor=me), 3),
    #(HiType4PyTop3(cpu_executor=ce, model_executor=me), 3),
]


@pytest.mark.parametrize(
    argnames=["tool", "topn"],
    argvalues=tools,
    ids=list(map(lambda t: type(t[0]).__name__, tools)),
)
def test_full_inference(tool: Inference, topn: int, example_project: Project) -> None:
    preprocess_project(
        dataset=example_project.mutable,
        preprocessor=tool.preprocessor(TypeCollectionCategory.VARIABLE),
    )
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

        diff = pd.merge(top1, top2, how="outer", suffixes=["_1", "_2"], indicator=True)
        print(diff)


@pytest.mark.parametrize(
    argnames=["tool", "topn"],
    argvalues=tools,
    ids=list(map(lambda t: type(t[0]).__name__, tools)),
)
def test_subset_inference(
    tool: Inference, topn: int, example_project_subset: ProjectSubset
) -> None:
    preprocess_project(
        dataset=example_project_subset.mutable,
        preprocessor=tool.preprocessor(TypeCollectionCategory.VARIABLE),
    )
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
