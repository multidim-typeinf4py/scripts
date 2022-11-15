import click

import diff
import symbols

if __name__ == "__main__":
    main = click.Group(commands=[diff.entrypoint, symbols.entrypoint])
    main()
