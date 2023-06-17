import libcst
from libcst import codemod, matchers as m


class LiteralToBaseClass(codemod.ContextAwareTransformer):
    @m.call_if_inside(m.Annotation(
        m.Attribute(m.Name("builtins"), m.Name("False") | m.Name("True"))
        | m.Name("False") | m.Name("True")
    ))
    def leave_Annotation(
        self, original_node: libcst.Annotation, updated_node: libcst.Annotation
    ) -> libcst.Annotation:
        return libcst.Annotation(libcst.Name("bool"))