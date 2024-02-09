import argparse
import logging
from typing import Any

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_argparser() -> argparse.ArgumentParser:
    parser_ = argparse.ArgumentParser(description="Mixtera CLI")
    return parser_


def validate_args(args: Any) -> None:
    del args  # currently unused


def main() -> None:
    parser = setup_argparser()
    args = parser.parse_args()

    validate_args(args)

    logger.info("This is the Mixtera CLI.")


if __name__ == "__main__":
    main()
