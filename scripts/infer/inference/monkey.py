import pathlib
import subprocess
import time

import docker
import pandera.typing as pt
from libcst import helpers, codemod

from scripts.common.schemas import InferredSchema, TypeCollectionCategory
from scripts.infer.preprocessers import monkey
from scripts.symbols.collector import build_type_collection
from ._base import ProjectWideInference

from ._adaptors import stubs2df
from ... import utils


class MonkeyType(ProjectWideInference):
    def method(self) -> str:
        return "monkeytype"

    def _infer_project(
        self,
        mutable: pathlib.Path,
        subset: set[pathlib.Path],
    ) -> pt.DataFrame[InferredSchema]:
        mutable = mutable.resolve()

        client = docker.from_env()
        repo = str(mutable)

        container = None
        try:
            self.logger.info("Creating Docker Image...")
            container = client.containers.create(
                image="monkeytype2", command="sleep infinity"
            )

            self.logger.info(f"Waiting for {container.id} to start...")
            container.start()

            while client.containers.get(container.id).status != "running":
                time.sleep(1)

            self.logger.info("Copying repository into running container...")
            subprocess.run(
                [
                    "docker",
                    "cp",
                    f"{repo}",
                    f"{container.id}:/repo/",
                ],
                check=True,
            )

            self.logger.info("Executing MonkeyType run script")
            exit_code, output = client.containers.get(container.id).exec_run(
                cmd=f"bash run.sh /repo /repo/monkeytype.sqlite3",
                detach=False,
                stream=True,
                environment={"PYTHONUNBUFFERED": "1"},
            )

            for chunk in output:
                message = chunk.decode("utf-8").strip()
                self.logger.info(message)

            if exit_code:
                self.logger.warning(f"From docker container: {output}")

            self.logger.info("Extracting transformed repository")

            subprocess.run(
                [
                    "docker",
                    "cp",
                    f"{container.id}:/repo/",
                    f"{mutable}",
                ],
                check=True,
            )

            with (mutable / "repo" / "monkeytype.sqlite3").open("rb") as f:
                self.register_artifact(f.read())

            df = build_type_collection(root=mutable / "repo", allow_stubs=False, subset=subset).df
            return df.assign(method=self.method(), topn=1).pipe(pt.DataFrame[InferredSchema])

        except Exception as e:
            self.logger.error(f"Could not collect MonkeyType results", exc_info=True)
            raise e

        finally:
            if container is not None:
                container.stop()
                container.remove(v=True)

    def preprocessor(self, task: TypeCollectionCategory) -> codemod.Codemod:
        return monkey.MonkeyPreprocessor(context=codemod.CodemodContext())
