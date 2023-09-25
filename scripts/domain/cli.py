import pprint
import sys
import click
import libcst
from libcst import matchers as m
import pathlib
import pandas as pd
import toml
from tqdm import tqdm

import trove_classifiers

from scripts.infer.structure import DatasetFolderStructure


@click.command(name="domain", help="Extract domain-specific labels")
@click.option(
    "-d",
    "--dataset",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    required=True,
)
@click.option(
    "-o",
    "--outpath",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    required=True,
    help="Folder for type collection dataframe to be written into",
)
def cli_entrypoint(dataset: pathlib.Path, outpath: pathlib.Path) -> None:
    classifiers = dict[str, str](
        map(
            lambda c: (c, c.split(" :: ")[1]),
            filter(lambda c: c.startswith("Topic"), trove_classifiers.classifiers),
        )
    )
    structure = DatasetFolderStructure(dataset_root=dataset)
    print(structure)

    projects2trove = dict[str, set[str]]()
    for repo in (pbar := tqdm(structure.test_set())):
        pbar.set_description(desc=str(structure.author_repo(repo)))
        projects2trove[structure.author_repo(repo)] = extract_trove_classifiers(
            repo, classifiers
        )

    viable = {
        repo: classifiers for repo, classifiers in projects2trove.items() if classifiers
    }

    for repo, classifiers in viable.items():
        print(repo, "->", classifiers)
        assert all(c.startswith("Topic ::") for c in classifiers)

    pd.DataFrame(
        [
            (repository, classifier)
            for repository in viable
            for classifier in viable[repository]
        ],
        columns=["repository", "classifier"],
    ).to_csv(structure.dataset_root / "troved.csv", index=False)


def extract_trove_classifiers(
    project: pathlib.Path, classifiers: dict[str, str]
) -> set[str]:
    selected = set[str]()
    for pyproject in project.rglob("*pyproject*.toml"):
        selected.update(extract_classifier_from_pyproject_toml(pyproject, classifiers))
    for setup in project.rglob("setup.py"):
        selected.update(extract_classifier_from_setuppy(setup, classifiers))

    # if selected:
    #    print(project, "->", selected)
    return selected


def extract_classifier_from_pyproject_toml(
    toml_file: pathlib.Path, classifiers: dict[str, str]
) -> set[str]:
    with toml_file.open() as f:
        pyproject = toml.load(f)

    extracted = classifiers.keys() & set(
        pyproject.get("project", dict()).get("classifiers", [])
    )
    return {e for e in extracted if e in classifiers}


setup_matcher = m.Arg(keyword=m.Name("classifiers"), value=m.List())


def extract_classifier_from_setuppy(
    setuppy_file: pathlib.Path, classifiers: dict[str, str]
) -> set[str]:
    extracted = set[str]()

    setup_module = libcst.parse_module(setuppy_file.read_text())
    classifiers_arg: libcst.Arg
    for classifiers_arg in m.findall(tree=setup_module, matcher=setup_matcher):
        for trove_classifier in classifiers_arg.value.elements:
            c = trove_classifier.value.value[1:-1]
            assert type(c) is str
            if c in classifiers:
                extracted.add(c)

    return extracted


def extract_classifier_from_string(classifier: str) -> str:
    assert len(s := classifier.split(" :: ")) >= 1
    return s[1:]
