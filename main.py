import click

import srcdiff
import hintdiff
import symbols

if __name__ == "__main__":
    main = click.Group(commands=[srcdiff.entrypoint, symbols.entrypoint])
    main()
