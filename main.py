
import click

import context
# import dataset
import logregr
import harness
import infer
import srcdiff
# import hintdiff
import symbols


if __name__ == "__main__":
    # os.environ["LIBCST_PARSER_TYPE"] = "native"

    # FORMAT = "%(asctime)s %(clientip)-15s %(user)-8s %(message)s"
    # logging.basicConfig(format=FORMAT)

    main = click.Group(
        commands=[
            context.cli_entrypoint,
            infer.cli_entrypoint,
            harness.cli_entrypoint,
            # dataset.cli_entrypoint,
            symbols.cli_entrypoint,
            srcdiff.cli_entrypoint,
            logregr.cli_entrypoint,
        ]
    )
    main()
