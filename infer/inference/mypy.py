import pathlib
import sys
from typing import Optional

import pandera.typing as pt

from libcst import codemod
from mypy import stubgen

from . import _adaptors
from ._base import ProjectWideInference
from utils import scratchpad
from common.schemas import InferredSchema


class MyPy(ProjectWideInference):
    method = "mypy"

    _OUTPUT_DIR = ".mypy-stubs"

    def _infer_project(
        self, root: pathlib.Path, subset: Optional[set[pathlib.Path]] = None
    ) -> pt.DataFrame[InferredSchema]:
        stubgen.generate_stubs(
            options=stubgen.Options(
                ignore_errors=False,
                no_import=False,
                parse_only=False,
                include_private=True,
                export_less=True,
                pyversion=sys.version_info[:2],
                output_dir=str(root / MyPy._OUTPUT_DIR),
                files=codemod.gather_files([str(root)], include_stubs=False),
                doc_dir="",
                search_path=[],
                interpreter=sys.executable,
                modules=[],
                packages=[],
                verbose=False,
                quiet=True,
            )
        )

        hintdf = _adaptors.stubs2df(root / MyPy._OUTPUT_DIR)
        if hintdf is not None:
            if subset is not None:
                retainable = list(map(str, subset))
                hintdf = hintdf[~hintdf[InferredSchema.file].isin(retainable)]

        return hintdf
