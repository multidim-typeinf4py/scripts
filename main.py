import click

import icr
import srcdiff
import hintdiff
import symbols

if __name__ == "__main__":
    main = click.Group(
        commands=[icr.entrypoint, hintdiff.entrypoint, symbols.entrypoint, srcdiff.entrypoint]
    )
    main()
