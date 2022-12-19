import sys

import pandera.typing as pt

import libcst.codemod._cli as cstcli
from mypy import stubgen

from . import _adaptors
from ._base import ProjectWideInference, scratchpad
from common.schemas import TypeCollectionSchema


class MyPy(ProjectWideInference):
    method = "mypy"

    _OUTPUT_DIR = ".mypy-stubs"

    def _infer_project(self) -> pt.DataFrame[TypeCollectionSchema]:
        with scratchpad(self.project) as sp:
            stubgen.generate_stubs(
                options=stubgen.Options(
                    ignore_errors=False,
                    no_import=False,
                    parse_only=False,
                    include_private=True,
                    export_less=True,
                    pyversion=sys.version_info[:2],
                    output_dir=str(sp / MyPy._OUTPUT_DIR),
                    files=cstcli.gather_files([str(sp)], include_stubs=False),
                    doc_dir="",
                    search_path=[],
                    interpreter=sys.executable,
                    modules=[],
                    packages=[],
                    verbose=False,
                    quiet=True,
                )
            )

            return _adaptors.stubs2df(sp / MyPy._OUTPUT_DIR)
