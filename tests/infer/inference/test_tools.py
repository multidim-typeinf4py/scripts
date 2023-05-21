import operator
import pytest

from common.schemas import InferredSchema, TypeCollectionSchema

from infer.inference import PyreInfer, PyreQuery, MyPy
from infer.inference import Type4PyTop10, TypeWriterTop10
from infer.inference.hity import HiTyperType4PyTop10
from infer.inference import Inference

from ._utils import Project, example_project, ProjectSubset, example_project_subset


tools = [
    (PyreInfer(), 1),
    (PyreQuery(), 1),
    (MyPy(), 1),
    (Type4PyTop10(), 10),
    (HiTyperType4PyTop10(), 10),
    (TypeWriterTop10(), 10),
]


@pytest.mark.parametrize(
    argnames=["tool", "topn"],
    argvalues=tools,
)
def test_full_inference(tool: Inference, topn: int, example_project: Project) -> None:
    inferred = tool.infer(mutable=example_project.mutable, readonly=example_project.readonly)

    print(inferred)
    assert not inferred.empty

    assert topn in inferred[InferredSchema.topn]
    assert (inferred[InferredSchema.topn] <= topn).all()


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
