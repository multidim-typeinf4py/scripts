import pathlib
from typing import Optional

import docker
from docker.errors import ContainerError
from docker.models.containers import Container

import pandera.typing as pt
from docker.types import Mount

from ._base import ProjectWideInference
from common.schemas import InferredSchema


class MonkeyType(ProjectWideInference):
    method = "monkeytype"

    def _infer_project(
        self,
        mutable: pathlib.Path,
        readonly: pathlib.Path,
        subset: Optional[set[pathlib.Path]] = None,
    ) -> pt.DataFrame[InferredSchema]:
        client = docker.from_env()

        try:
            client.containers.run(
                image="monkeytype",
                volumes={str(mutable.resolve()): {"bind": "/repo", "mode": "rw"}},
                command="bash run.sh",
                remove=True
            )
        except ContainerError as e:
            print(f"Error occurred while applying monkeytype, returning:\n{e}")
            return InferredSchema.example(size=0)

        return None
