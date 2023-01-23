import pathlib

from common.schemas import InferredSchema
from tests.icr.helpers import dfassertions

from icr.inference._base import Inference
from icr.inference import HiTyper, PyreInfer, PyreQuery, Type4Py, TypeWriter

import pytest

import pandera.typing as pt


@pytest.fixture(
    scope="class",
    # params=[HiTyper, PyreInfer, PyreQuery, Type4Py, TypeWriter],
    params=[TypeWriter],
    ids=lambda e: e.__qualname__,
)
def methoddf(request) -> tuple[str, pt.DataFrame[InferredSchema]]:
    inf: Inference = request.param(pathlib.Path.cwd() / "tests" / "resources" / "proj1")
    inf.infer()
    df = inf.inferred

    return type(inf).__qualname__, df


DONT_SUPPORT_VARIABLES = (TypeWriter.__qualname__,)


class TestCoverage:
    def test_function(self, methoddf: tuple[str, pt.DataFrame[InferredSchema]]):
        method, df = methoddf
        # Returns
        assert dfassertions.has_callable(df, f_qname="function")

        # Params
        assert dfassertions.has_parameter(df, f_qname="function", arg_name="a")
        assert dfassertions.has_parameter(df, f_qname="function", arg_name="b")
        assert dfassertions.has_parameter(df, f_qname="function", arg_name="c")

    def test_function_body(self, methoddf: tuple[str, pt.DataFrame[InferredSchema]]):
        method, df = methoddf
        if method in DONT_SUPPORT_VARIABLES:
            pytest.skip(f"{method} does not support vars!")

        # Body
        assert dfassertions.has_variable(df, var_qname="function.v")

    def test_function_with_multiline_parameters(
        self, methoddf: tuple[str, pt.DataFrame[InferredSchema]]
    ):
        method, df = methoddf
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

    def test_function_with_multiline_parameters_body(
        self, methoddf: tuple[str, pt.DataFrame[InferredSchema]]
    ):
        method, df = methoddf
        if method in DONT_SUPPORT_VARIABLES:
            pytest.skip(f"{method} does not support vars!")

        # Body
        assert dfassertions.has_variable(df, var_qname="function_with_multiline_parameters.v")

    def test_Clazz_a(self, methoddf: tuple[str, pt.DataFrame[InferredSchema]]):
        method, df = methoddf
        if method in DONT_SUPPORT_VARIABLES:
            pytest.skip(f"{method} does not support vars!")

        # Body
        assert dfassertions.has_variable(df, var_qname="Clazz.a")

    def test_Clazz_init(self, methoddf: tuple[str, pt.DataFrame[InferredSchema]]):
        method, df = methoddf
        # Returns
        assert dfassertions.has_callable(df, f_qname="Clazz.__init__")

        # Params
        assert dfassertions.has_parameter(df, f_qname="Clazz.__init__", arg_name="a")

    def test_Clazz_init_body(self, methoddf: tuple[str, pt.DataFrame[InferredSchema]]):
        method, df = methoddf
        if method in DONT_SUPPORT_VARIABLES:
            pytest.skip(f"{method} does not support vars!")

        # Body
        assert dfassertions.has_variable(df, var_qname="Clazz.__init__.self.a")

    def test_Clazz_method(self, methoddf: tuple[str, pt.DataFrame[InferredSchema]]):
        method, df = methoddf

        # Returns
        assert dfassertions.has_callable(df, f_qname="Clazz.method")

        # Params
        assert dfassertions.has_parameter(df, f_qname="Clazz.method", arg_name="a")
        assert dfassertions.has_parameter(df, f_qname="Clazz.method", arg_name="b")
        assert dfassertions.has_parameter(df, f_qname="Clazz.method", arg_name="c")

        # Body

    def test_Clazz_multiline_method(self, methoddf: tuple[str, pt.DataFrame[InferredSchema]]):
        method, df = methoddf
        if method in DONT_SUPPORT_VARIABLES:
            pytest.skip(f"{method} does not support vars!")

        # Returns
        assert dfassertions.has_callable(df, f_qname="Clazz.multiline_method")

        # Params
        assert dfassertions.has_parameter(df, f_qname="Clazz.multiline_method", arg_name="a")
        assert dfassertions.has_parameter(df, f_qname="Clazz.multiline_method", arg_name="b")
        assert dfassertions.has_parameter(df, f_qname="Clazz.multiline_method", arg_name="c")

        # Body

    def test_Clazz_function(self, methoddf: tuple[str, pt.DataFrame[InferredSchema]]):
        method, df = methoddf
        # Returns
        assert dfassertions.has_callable(df, f_qname="Clazz.function")

        # Params
        assert dfassertions.has_parameter(df, f_qname="Clazz.function", arg_name="a")
        assert dfassertions.has_parameter(df, f_qname="Clazz.function", arg_name="b")
        assert dfassertions.has_parameter(df, f_qname="Clazz.function", arg_name="c")

    def test_Clazz_function_body(self, methoddf: tuple[str, pt.DataFrame[InferredSchema]]):
        method, df = methoddf
        if method in DONT_SUPPORT_VARIABLES:
            pytest.skip(f"{method} does not support vars!")

        # Body
        assert dfassertions.has_variable(df, var_qname="Clazz.function.v")

    def test_a(self, methoddf: tuple[str, pt.DataFrame[InferredSchema]]):
        method, df = methoddf
        if method in DONT_SUPPORT_VARIABLES:
            pytest.skip(f"{method} does not support vars!")

        assert dfassertions.has_variable(df, var_qname="a")

    def test_outer_nested(self, methoddf: tuple[str, pt.DataFrame[InferredSchema]]):
        method, df = methoddf
        # Returns
        assert dfassertions.has_callable(df, f_qname="outer.nested")

        # Params
        assert dfassertions.has_parameter(df, f_qname="outer.nested", arg_name="a")

    def test_outer_nested_body(self, methoddf: tuple[str, pt.DataFrame[InferredSchema]]):
        method, df = methoddf
        if method in DONT_SUPPORT_VARIABLES:
            pytest.skip(f"{method} does not support vars!")

        # Body
        assert dfassertions.has_variable(df, var_qname="outer.nested.result")

    def test_outer(self, methoddf: tuple[str, pt.DataFrame[InferredSchema]]):
        method, df = methoddf
        # Returns
        assert dfassertions.has_callable(df, f_qname="outer")

    def test_Outer_Inner(self, methoddf: tuple[str, pt.DataFrame[InferredSchema]]):
        method, df = methoddf
        # Returns
        assert dfassertions.has_callable(df, f_qname="Outer.Inner.__init__")

        # Params

    def test_Outer_Inner_body(self, methoddf: tuple[str, pt.DataFrame[InferredSchema]]):
        method, df = methoddf
        if method in DONT_SUPPORT_VARIABLES:
            pytest.skip(f"{method} does not support vars!")

        # Body
        assert dfassertions.has_variable(df, var_qname="Outer.Inner.__init__.self.x")

    def test_unique(self, methoddf: tuple[str, pt.DataFrame[InferredSchema]]):
        method, df = methoddf
        # unique based on slot, e.g. parameter and variable can be the same
        dups = df[
            df.duplicated(
                subset=[
                    InferredSchema.category,
                    InferredSchema.qname_ssa,
                    InferredSchema.anno,
                    InferredSchema.topn,
                ],
                keep=False,
            )
        ]
        assert dups.empty, str(dups)
