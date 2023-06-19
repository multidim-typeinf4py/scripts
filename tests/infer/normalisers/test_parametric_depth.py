from libcst import codemod

from scripts.infer.normalisers import parametric_depth

class Test_Unstringify(codemod.CodemodTest):
    TRANSFORM = parametric_depth.ParametricTypeDepthReducer

    def test_transformed_into_any(self):
        self.assertCodemod(
            before="l: List[List[Tuple[int]]]",
            after="l: List[List[typing.Any]]"
        )
