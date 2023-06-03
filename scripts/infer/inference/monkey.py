from libcst.codemod.visitors import GatherImportsVisitor
from libcst import codemod
import pandera.typing as pt

from ._base import ProjectWideInference
from scripts.common import InferredSchema

class MonkeyType(ProjectWideInference):
    def _infer_project(self) -> pt.DataFrame[InferredSchema]:
        requirements
        import_gatherer = GatherImportsVisitor()
        codemod.exec_transform_with_prettyprint(import_gatherer, )
