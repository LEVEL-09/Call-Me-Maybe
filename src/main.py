from argparse import ArgumentParser, Namespace

from .llm_response import LLMResponse, ResultDictType
from .parsers import load_function_definitions, load_prompts
from .parsers.error_handler import FileParsingErrorHandler

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
    args = parse_args()
    result: list[ResultDictType] = []

    with FileParsingErrorHandler():
        prompts = load_prompts(args.input)
        function_definitions = load_function_definitions(
            args.functions_definition
        )

        llm_response = LLMResponse(result, function_definitions)
        for item in prompts:
            llm_response.generate_response(item.prompt)

        for i in result:
            print(i)
