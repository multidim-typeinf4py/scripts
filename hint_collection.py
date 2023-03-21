import pathlib
import sys, os

from symbols.collector import build_type_collection


import libcst


if __name__ == "__main__":
    for author in os.listdir(sys.argv[1]):
        for repo in os.listdir(os.path.join(sys.argv[1], author)):
            path = pathlib.Path(sys.argv[1]) / author / repo

            try:
                _ = build_type_collection(path, allow_stubs=False)
            except libcst.ParserSyntaxError as pse:
                sys.stdout.write(f"{author}/{repo} - {pse.message}")
            except Exception as e:
                sys.stderr.write(f"{e}")