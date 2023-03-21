import pathlib
from itertools import chain, combinations
import typing

from infer.inference import Type4Py, HiTyper, PyreInfer, PyreQuery, TypeWriter


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def file_format(tool: str, removing: list[str]) -> str:
    return f"{tool}-{'_'.join(removing)}"

def inference_function(removing: list[str]) -> str:
    return '_'.join(removing) + "_inference"

if __name__ == "__main__":
    supported = (
            PyreInfer.__name__.lower(),
            PyreQuery.__name__.lower(),
            HiTyper.__name__.lower(),
            TypeWriter.__name__.lower(),
            Type4Py.__name__.lower(),
    )


    for tool in supported:
        for removal in powerset(["variable", ""]):
            filename = 