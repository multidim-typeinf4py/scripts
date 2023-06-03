import logging
import click

#import context
#import dataset
#import logregr
#import harness
from scripts import infer, srcdiff, utils

# import hintdiff
# import symbols


if __name__ == "__main__":
    # os.environ["LIBCST_PARSER_TYPE"] = "native"

    # fmt="[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
    # datefmt="%Y-%m-%d %H:%M:%S"

    # logging.basicConfig(format=fmt, datefmt=datefmt, level=logging.INFO)
    # logging.info(f"{ utils.worker_count()=}")
    # logging.info(f"{sys.path=}")

    main = click.Group(
        commands=[
            #context.cli_entrypoint,
            infer.cli_entrypoint,
            #harness.cli_entrypoint,
            #dataset.cli_entrypoint,
            # symbols.cli_entrypoint,
            srcdiff.cli_entrypoint,
            #logregr.cli_entrypoint,
        ]
    )
    main()
