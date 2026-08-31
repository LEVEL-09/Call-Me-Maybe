from argparse import ArgumentParser, Namespace


DEFAULT_FUNCTIONS_DEFINITION = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        type=str,
        default=DEFAULT_FUNCTIONS_DEFINITION,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args()


def main() -> None:
    pass
