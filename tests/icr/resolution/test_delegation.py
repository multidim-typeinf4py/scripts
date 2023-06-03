import pathlib
from scripts.icr import Delegation, DelegationOrder

from scripts.common import SymbolSchema, InferredSchema, TypeCollectionCategory
from scripts.symbols.collector import build_type_collection

from pandas._libs import missing
import pandera.typing as pt



# from tests..helpers import dfassertions

import pytest

pytest.skip("Fix imports", allow_module_level=True)

agent1 = pt.DataFrame[InferredSchema](
    {
        "file": ["x.py"] * 3,
        "method": ["static"] * 3,
        "category": [TypeCollectionCategory.CALLABLE_PARAMETER] * 3,
        "qname": [f"function.{name}" for name in "abc"],
        "qname_ssa": [f"function.{name}" for name in "abc"],
        "anno": [missing.NA, "int", missing.NA],
        "topn": [0] * 3
    }
)

agent2 = pt.DataFrame[InferredSchema](
    {
        "file": ["x.py"] * 4,
        "method": ["prob"] * 4,
        "category": [TypeCollectionCategory.CALLABLE_PARAMETER] * 4,
        "qname": [f"function.{name}" for name in "abcc"],
        "qname_ssa": [f"function.{name}" for name in "abcc"],
        "anno": [missing.NA, "bool", "str", "bytes"],
        "topn": [0] * 3 + [1]
    }
)

## Test data summary:
## a is entirely unknown, c is unknown to agent1, b is known to both
## c has a top 2 prediction for agent2; the less likely one should never be picked


@pytest.fixture()
def proj1() -> tuple[pathlib.Path, pt.DataFrame[SymbolSchema]]:
    path = pathlib.Path.cwd() / "tests" / "resources" / "proj1"
    reference = build_type_collection(path).df.drop(columns="anno").pipe(pt.DataFrame[SymbolSchema])

    return (path, reference)


def test_static_over_prob(proj1: tuple[pathlib.Path, pt.DataFrame[SymbolSchema]]):
    path, reference = proj1
    d = Delegation(
        project=path,
        reference=reference,
        order=(DelegationOrder.STATIC, DelegationOrder.PROBABILISTIC),
    )
    inferred = d.forward(static=agent1, dynamic=None, probabilistic=agent2)

    assert dfassertions.has_parameter(inferred, f_qname="function", arg_name="a", anno=missing.NA)
    assert dfassertions.has_parameter(inferred, f_qname="function", arg_name="b", anno="int")
    assert dfassertions.has_parameter(inferred, f_qname="function", arg_name="c", anno="str")


def test_prob_over_static(proj1: tuple[pathlib.Path, pt.DataFrame[SymbolSchema]]):
    path, reference = proj1
    d = Delegation(
        project=path,
        reference=reference,
        order=(DelegationOrder.PROBABILISTIC, DelegationOrder.STATIC),
    )
    inferred = d.forward(static=agent1, dynamic=None, probabilistic=agent2)

    assert dfassertions.has_parameter(inferred, f_qname="function", arg_name="a", anno=missing.NA)
    assert dfassertions.has_parameter(inferred, f_qname="function", arg_name="b", anno="bool")
    assert dfassertions.has_parameter(inferred, f_qname="function", arg_name="c", anno="str")
