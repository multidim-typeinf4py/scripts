from scripts.common import extending

import pytest

class Test_ParametricConversions:
    def test_none_stays_same(self):
        assert extending.make_parametric(None) is None

    def test_nonparametric_stays_same(self):
        assert extending.make_parametric("builtins.int") == "builtins.int"

    def test_parametric_dict_is_rewritten(self):
        assert (
            extending.make_parametric("builtins.dict[builtins.str, builtins.int]")
            == "builtins.dict"
        )

    def test_inner_parametric_disappears(self):
        assert (
            extending.make_parametric(
                "builtins.dict[builtins.dict[builtins.str, builtins.int], builtins.int]"
            )
            == "builtins.dict"
        )


class Test_ComplexityCounter:
    @pytest.mark.parametrize(argnames="code", argvalues=[
        "int", "foo.Bar"
    ])
    def test_simple(self, code: str):
        assert extending.is_simple_or_complex(annotation=code) == "simple"

    @pytest.mark.parametrize(argnames="code", argvalues=[
        "dict[str, foo.Bar]", "tuple[int]"
    ])
    def test_complex(self, code: str):
        assert extending.is_simple_or_complex(code) == "complex"