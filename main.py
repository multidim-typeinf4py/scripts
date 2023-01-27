import click

import context
import logregr
import icr
import srcdiff
import hintdiff
import symbols

import os

if __name__ == "__main__":
    os.environ["LIBCST_PARSER_TYPE"] = "native"

    # logging518.config.fileConfig("pyproject.toml")

    main = click.Group(
        commands=[
            context.entrypoint,
            icr.entrypoint,
            hintdiff.entrypoint,
            symbols.entrypoint,
            srcdiff.entrypoint,
            logregr.entrypoint,
        ]
    )
    main()
