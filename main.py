import click

import context
import logregr
import icr
import srcdiff
import hintdiff
import symbols

import logging518.config

if __name__ == "__main__":
    logging518.config.fileConfig("pyproject.toml")

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
