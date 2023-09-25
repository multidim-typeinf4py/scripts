import collections
import io
import itertools
import pathlib
import subprocess
import time

import os, tarfile
import pandera.typing as pt
from libcst import helpers, codemod

from scripts.common.schemas import InferredSchema, TypeCollectionCategory
from scripts.infer.preprocessers import monkey
from scripts.infer.structure import DatasetFolderStructure
from scripts.symbols.collector import build_type_collection
from ._base import ProjectWideInference

from ._adaptors import stubs2df
from scripts import utils

from typet5.static_analysis import PythonProject


class MonkeyType(ProjectWideInference):
    def method(self) -> str:
        return "monkeytype"

    def _infer_project(
        self,
        mutable: pathlib.Path,
        subset: set[pathlib.Path],
    ) -> pt.DataFrame[InferredSchema]:
        flapy_results = os.environ.get("FLAPY_RESULTS")
        assert (
            flapy_results is not None
        ), f"Inference with Monkeytype requires the FLAPY_RESULTS environment variable"

        dataset = DatasetFolderStructure(pathlib.Path(os.environ["DATASET_ROOT"]))
        repository = pathlib.Path(os.environ["REPOSITORY"])
        artifact_name = str(dataset.author_repo(repository))

        """monkeytype_archive = (
            pathlib.Path(flapy_results) / artifact_name / "monkeytype.sqlite3"
        )
        if not monkeytype_archive.is_file():
            self.logger.error(f"{monkeytype_archive} was not found")
            return InferredSchema.example(size=0)

        module_list = set(
            subprocess.check_output(
                args=["monkeytype", "list-modules"],
                env=dict(os.environ, MT_DB_PATH=str(monkeytype_archive.resolve())),
            )
            .decode(encoding="utf-8", errors="strict")
            .splitlines()
        )
        self.logger.info(module_list)

        if not module_list or module_list == {""}:
            self.logger.error("No modules were traced!")
            return InferredSchema.example(size=0)

        working_dirs = collections.defaultdict[pathlib.Path, list[str]](list)
        # Convert subset paths to working dirs for working around MonkeyType's odd submodule paths
        for subdir in filter(
            lambda p: p.is_dir(),
            itertools.chain((mutable,), mutable.rglob("*")),
        ):
            if not module_list:
                break
            for candidate in subset:
                try:
                    relpath = (mutable / candidate).relative_to(subdir)
                    modname = PythonProject.rel_path_to_module_name(relpath)

                except ValueError:
                    continue

                if modname in module_list:
                    working_dirs[subdir].append(modname)
                    module_list.remove(modname)

        if module_list:
            self.logger.error(f"Could not find modules for inference: {module_list}")
        self.logger.info(working_dirs)

        for working_dir, modules in working_dirs.items():
            self.logger.info(f"{working_dir} -> {modules}")
            subprocess.Popen(
                f"xargs -d , -n1 monkeytype -v apply --ignore-existing-annotations",
                shell=True,
                env=dict(os.environ, MT_DB_PATH=str(monkeytype_archive.resolve())),
                cwd=working_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
            ).communicate(input=bytes(",".join(modules), "utf-8"))

        self.logger.info(f"Finished applying {monkeytype_archive}")"""

        monkeytype_archive = (
            pathlib.Path(flapy_results) / artifact_name / "annotated.tar.gz"
        )
        if not monkeytype_archive.exists():
            self.logger.error(f"{monkeytype_archive} was not found")
            return InferredSchema.example(size=0)

        with tarfile.open(monkeytype_archive) as t:
            t.extractall(path=mutable)

        stub_location = mutable / "stubs"
        for stub in stub_location.rglob("*.pyi"):
            src = (mutable / stub.relative_to(stub_location)).with_suffix(".py")

            if src.exists():
                self.logger.info(f"{stub} -> {src}")
                subprocess.Popen(
                    ["merge-pyi", "-i", str(src), str(stub)],
                    cwd=mutable,
                ).wait()
            else:
                self.logger.error(f"{src} does not exist; deduced from {stub}")

        return (
            build_type_collection(
                root=mutable,
                allow_stubs=False,
                subset=subset,
            )
            .df.assign(topn=1, method=self.method())
            .pipe(pt.DataFrame[InferredSchema])
        )

    def preprocessor(self, task: TypeCollectionCategory) -> codemod.Codemod:
        return monkey.MonkeyPreprocessor(context=codemod.CodemodContext(), task="all")
