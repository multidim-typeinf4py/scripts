import pathlib

from common.schemas import InferredSchema
from tests.icr.helpers import dfassertions

from icr.inference import HiTyper, PyreInfer, Type4Py, TypeWriter

import pytest

import pandera.typing as pt


@pytest.fixture(
    scope="class", params=[HiTyper, PyreInfer, Type4Py, TypeWriter], ids=lambda e: e.__qualname__
)
def df(request) -> pt.DataFrame[InferredSchema]:
    inf = request.param(pathlib.Path.cwd() / "tests" / "resources" / "proj1")
    inf.infer()
    df = inf.inferred

    return df


class TestCoverage:
    def test_function(self, df: pt.DataFrame[InferredSchema]):
        # Returns
        assert dfassertions.has_callable(df, f_qname="function")

        # Params
        assert dfassertions.has_parameter(df, f_qname="function", arg_name="a")
        assert dfassertions.has_parameter(df, f_qname="function", arg_name="b")
        assert dfassertions.has_parameter(df, f_qname="function", arg_name="c")

    def test_function_body(self, df: pt.DataFrame[InferredSchema]):
        # Body
        assert dfassertions.has_variable(df, var_qname="function.v")

    def test_function_with_multiline_parameters(self, df: pt.DataFrame[InferredSchema]):
        # Returns
        assert dfassertions.has_callable(df, f_qname="function_with_multiline_parameters")

        # Params
        assert dfassertions.has_parameter(
            df, f_qname="function_with_multiline_parameters", arg_name="a"
        )

        assert dfassertions.has_parameter(
            df, f_qname="function_with_multiline_parameters", arg_name="b"
        )
        assert dfassertions.has_parameter(
            df, f_qname="function_with_multiline_parameters", arg_name="c"
        )

    def test_function_with_multiline_parameters_body(self, df: pt.DataFrame[InferredSchema]): 
        # Body
        assert dfassertions.has_variable(df, var_qname="function_with_multiline_parameters.v")

    def test_Clazz_a(self, df: pt.DataFrame[InferredSchema]):
        # Body
        assert dfassertions.has_variable(df, var_qname="Clazz.a")

    def test_Clazz_init(self, df: pt.DataFrame[InferredSchema]):
        # Returns
        assert dfassertions.has_callable(df, f_qname="Clazz.__init__")

        # Params
        assert dfassertions.has_parameter(df, f_qname="Clazz.__init__", arg_name="a")

    def test_Clazz_init_body(self, df: pt.DataFrame[InferredSchema]): 
        # Body
        assert dfassertions.has_variable(df, var_qname="Clazz.__init__.self.a")

    def test_Clazz_method(self, df: pt.DataFrame[InferredSchema]):
        # Returns
        assert dfassertions.has_callable(df, f_qname="Clazz.method")

        # Params
        assert dfassertions.has_parameter(df, f_qname="Clazz.method", arg_name="a")
        assert dfassertions.has_parameter(df, f_qname="Clazz.method", arg_name="b")
        assert dfassertions.has_parameter(df, f_qname="Clazz.method", arg_name="c")

        # Body

    def test_Clazz_multiline_method(self, df: pt.DataFrame[InferredSchema]):
        # Returns
        assert dfassertions.has_callable(df, f_qname="Clazz.multiline_method")

        # Params
        assert dfassertions.has_parameter(df, f_qname="Clazz.multiline_method", arg_name="a")
        assert dfassertions.has_parameter(df, f_qname="Clazz.multiline_method", arg_name="b")
        assert dfassertions.has_parameter(df, f_qname="Clazz.multiline_method", arg_name="c")

        # Body

    def test_Clazz_function(self, df: pt.DataFrame[InferredSchema]):
        # Returns
        assert dfassertions.has_callable(df, f_qname="Clazz.function")

        # Params
        assert dfassertions.has_parameter(df, f_qname="Clazz.function", arg_name="a")
        assert dfassertions.has_parameter(df, f_qname="Clazz.function", arg_name="b")
        assert dfassertions.has_parameter(df, f_qname="Clazz.function", arg_name="c")

    def test_Clazz_function_body(self, df: pt.DataFrame[InferredSchema]): 
        # Body
        assert dfassertions.has_variable(df, var_qname="Clazz.function.v")

    def test_a(self, df: pt.DataFrame[InferredSchema]):
        assert dfassertions.has_variable(df, var_qname="a")

    def test_outer_nested(self, df: pt.DataFrame[InferredSchema]):
        # Returns
        assert dfassertions.has_callable(df, f_qname="outer.nested")

        # Params
        assert dfassertions.has_parameter(df, f_qname="outer.nested", arg_name="a")

    def test_outer_nested_body(self, df: pt.DataFrame[InferredSchema]): 
        # Body
        assert dfassertions.has_variable(df, var_qname="outer.nested.result")

    def test_outer(self, df: pt.DataFrame[InferredSchema]):
        # Returns
        assert dfassertions.has_callable(df, f_qname="outer")

    def test_Outer_Inner(self, df: pt.DataFrame[InferredSchema]):
        # Returns
        assert dfassertions.has_callable(df, f_qname="Outer.Inner.__init__")

        # Params

    def test_Outer_Inner_body(self, df: pt.DataFrame[InferredSchema]): 
        # Body
        assert dfassertions.has_variable(df, var_qname="Outer.Inner.__init__.self.x")
