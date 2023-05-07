import pathlib
from typing import Optional
import docker

import pandera.typing as pt

from ._base import ProjectWideInference
from common.schemas import InferredSchema


class MonkeyType(ProjectWideInference):
    def infer(
        self,
        mutable: pathlib.Path,
        readonly: pathlib.Path,
        subset: Optional[set[pathlib.Path]] = None,
    ) -> None:
        client = docker.from_env()
        client.containers.run(image="monkeytype")
