import click

import context
import logregr
import icr
import srcdiff
import hintdiff
import symbols

if __name__ == "__main__":
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
