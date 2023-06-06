import pathlib
import sys
from typing import Optional

import pandera.typing as pt

from libcst import codemod
from mypy import stubgen

from . import _adaptors
from ._base import ProjectWideInference
from src.common.schemas import InferredSchema


class MyPy(ProjectWideInference):
    def method(self) -> str:
        return "mypy"

    _OUTPUT_DIR = ".mypy-stubs"

    def _infer_project(
        self, mutable: pathlib.Path, subset: Optional[set[pathlib.Path]]
    ) -> pt.DataFrame[InferredSchema]:
        options = stubgen.Options(
            ignore_errors=True,
            no_import=False,
            parse_only=False,
            include_private=True,
            export_less=True,
            pyversion=sys.version_info[:2],
            output_dir=str(mutable / MyPy._OUTPUT_DIR),
            files=codemod.gather_files([str(mutable)], include_stubs=False),
            doc_dir="",
            search_path=[],
            interpreter=sys.executable,
            modules=[],
            packages=[],
            verbose=False,
            quiet=True,
        )

        try:
            stubgen.generate_stubs(options=options)

        except SystemExit as e:
            self.logger.error(f"Falling back to --parse_only=True: {e}")
            options.parse_only = True

            try:
                stubgen.generate_stubs(options=options)
            except SystemExit as e:
                self.logger.error(f"Fallback failed too; {e}; giving up...")

        except Exception as e:
            self.logger.error(f"Stub Generation failed: {e}; giving up...")

        finally:
            if (mutable / MyPy._OUTPUT_DIR).is_dir():
                return (
                    _adaptors.stubs2df(
                        project_folder=mutable,
                        stubs_folder=mutable / MyPy._OUTPUT_DIR,
                        subset=subset,
                    )
                    .assign(method=self.method(), topn=1)
                    .pipe(pt.DataFrame[InferredSchema])
                )
            return InferredSchema.example(size=0)
