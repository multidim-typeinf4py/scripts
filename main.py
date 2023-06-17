import logging
import click

# import context
# import logregr
# import harness
#from scripts import dataset, infer, srcdiff, utils

from scripts.dataset.cli import cli_entrypoint as dce
from scripts.infer.cli import click as ice
from scripts import utils

# import hintdiff
# import symbols


if __name__ == "__main__":
    # os.environ["LIBCST_PARSER_TYPE"] = "native"

    fmt="[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
    datefmt="%Y-%m-%d %H:%M:%S"

    # formatter = logging.Formatter(
    #     fmt="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    # )

    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt)
    print(f"{utils.worker_count()=}")
    # logging.info(f"{sys.path=}")

    main = click.Group(
        commands=[
            # context.cli_entrypoint,
            ice,
            # harness.cli_entrypoint,
            dce,
            # symbols.cli_entrypoint,
            # srcdiff.cli_entrypoint,
            # logregr.cli_entrypoint,
        ]
    )
    main()
