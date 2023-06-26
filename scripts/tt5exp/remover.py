from typet5.experiments import utils
from typet5.experiments.utils import SupportedSyntax
from typet5.static_analysis import AnnotRemover

import libcst
from libcst import codemod

class TT5AllAnnotRemover(codemod.Codemod):
    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        simpler_syntax = utils.remove_newer_syntax(tree, supported=SupportedSyntax(
            pattern_match=False,
            union_types=False,
            basic_types=False,
        ))
        return simpler_syntax.visit(AnnotRemover())