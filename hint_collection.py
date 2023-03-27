import pathlib
import sys, os

from symbols.collector import build_type_collection

import tqdm


if __name__ == "__main__":
    for author in (pbar := tqdm.tqdm(os.listdir(sys.argv[1]), position=0)):
        for repo in os.listdir(os.path.join(sys.argv[1], author)):
            pbar.set_description(desc=f"{author}/{repo}")
            path = pathlib.Path(sys.argv[1]) / author / repo

            try:
                _ = build_type_collection(path, allow_stubs=False)
            # except libcst.ParserSyntaxError as pse:
            #    out.write(f"{author}/{repo} - {pse.message}\n")
            except Exception as e:
                sys.stderr.write(f"{author}/{repo} - {e}\n")
