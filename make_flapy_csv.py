import pathlib
from scripts.infer.structure import DatasetFolderStructure

from tqdm import tqdm
import pandas as pd
import sys, subprocess

assert len(sys.argv) > 2, f"{__file__} $DATASET $FLAPY-OUT.CSV"
print(dataset := DatasetFolderStructure(pathlib.Path(sys.argv[1])))

dependency_files = ["*requirements*.txt", "Pipfile", "poetry.lock"]

projects = []

for project in (pbar := tqdm(dataset.test_set())):
    id_ = str(dataset.author_repo(project))
    pbar.set_description(desc=id_)

    pfolder = project.resolve()
    if (pfolder / "poetry.lock").is_file():
        print(pfolder, "has poetry.lock")
        requirements_out = pfolder / "poetry-requirements.txt"

        subprocess.Popen(
            [
                "poetry",
                "export",
                "-f",
                "requirements.txt",
                "--output",
                str(requirements_out),
                "--with",
                "dev",
                "--without-hashes",
            ],
            cwd=pfolder,
        )

    if any(list(pfolder.glob(pattern)) for pattern in dependency_files):
        projects.append((id_, str(pfolder), "", "", "", "", "1"))

pd.DataFrame.from_records(
    projects,
    columns="PROJECT_NAME,PROJECT_URL,PROJECT_HASH,PYPI_TAG,FUNCS_TO_TRACE,TESTS_TO_BE_RUN,NUM_RUNS".split(
        ","
    ),
).to_csv(pathlib.Path(sys.argv[2]), index=False)
