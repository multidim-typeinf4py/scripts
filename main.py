import logging
import click

# import logregr
# import harness
from scripts import context, dataset, domain, infer, srcdiff, tt5exp, utils, mt_copy

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
            infer.cli_entrypoint,
            tt5exp.cli_entrypoint,
            dataset.cli_entrypoint,
            context.cli_entrypoint,
            domain.cli_entrypoint,
            # harness.cli_entrypoint,
            # symbols.cli_entrypoint,
            srcdiff.cli_entrypoint,
            mt_copy.cli_entrypoint,
            # logregr.cli_entrypoint,
        ]
    )
    main()
