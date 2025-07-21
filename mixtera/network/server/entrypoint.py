import argparse
from pathlib import Path

from loguru import logger

from mixtera.network.server import MixteraServer


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mixtera Server")
    parser.add_argument(
        "directory",
        type=Path,
        action="store",
        help="Underlying directory where sqlite index will be stored",
    )

    parser.add_argument(
        "--host",
        type=str,
        const="0.0.0.0",
        default="0.0.0.0",
        action="store",
        nargs="?",
        help="To which IP the MixteraServer should bind. Defaults to 0.0.0.0, i.e., listens to all incoming requests.",
    )

    parser.add_argument(
        "--port",
        type=int,
        const=8888,
        default=8888,
        action="store",
        nargs="?",
        help="To which Port the MixteraServer should bind. Defaults to 8888.",
    )
    return parser


def main() -> None:
    parser = setup_argparser()
    args = parser.parse_args()
    directory = args.directory
    host = args.host
    port = args.port

    logger.info(f"Starting server, serving from directory {directory}")

    server = MixteraServer(directory, host, port)
    server.run()

    logger.debug("Server has exited, exiting entrypoint script.")


if __name__ == "__main__":
    main()
