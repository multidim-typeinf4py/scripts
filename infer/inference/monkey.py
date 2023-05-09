import os
import pathlib
import shutil
import tempfile
from typing import Optional

import docker
import pandera.typing as pt
from docker.errors import ContainerError
from libcst import helpers, codemod

from common.schemas import InferredSchema
from symbols.collector import build_type_collection
from ._base import ProjectWideInference


class MonkeyType(ProjectWideInference):
    method = "monkeytype"

    def _infer_project(
        self,
        mutable: pathlib.Path,
        readonly: pathlib.Path,
        subset: Optional[set[pathlib.Path]] = None,
    ) -> pt.DataFrame[InferredSchema]:
        client = docker.from_env()
        repo = str(mutable.resolve())

        if subset is None:
            s = codemod.gather_files([str(mutable)])
        else:
            s = subset

        modules = ",".join(
            {
                helpers.calculate_module_and_package(
                    repo_root=repo, filename=str(filename)
                ).name
                for filename in s
            }
        )

        with tempfile.TemporaryDirectory() as mtoutput:
            shutil.copytree(
                repo,
                mtoutput,
                ignore_dangling_symlinks=True,
                symlinks=True,
                dirs_exist_ok=True,
            )
            try:
                client.containers.run(
                    image="monkeytype",
                    volumes={
                        repo: {"bind": "/repo", "mode": "ro"},
                        mtoutput: {"bind": "/monkeytype-output", "mode": "rw"},
                    },
                    command=f"bash run.sh /repo /monkeytype-output {modules}",
                    remove=True,
                )
            except ContainerError as e:
                print(f"Error occurred while applying monkeytype, returning:\n{e}")
                return InferredSchema.example(size=0)

            return (
                build_type_collection(root=pathlib.Path(mtoutput), subset=subset)
                .df.assign(method=self.method, topn=1)
                .pipe(pt.DataFrame[InferredSchema])
            )
