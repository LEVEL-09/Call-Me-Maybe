import numpy as np
from typing import Any, TypedDict

from llm_sdk import Small_LLM_Model

from .parsers.functions_definition_parser import FunctionDefinition

SYSTEM_PROMPT = """
You are a function choicer.

Choose the right function name just from Available functions.

User prompt:
{PROMPT}

Available functions:
{FUNCTIONS}

The answer is:
"""


class ResultDictType(TypedDict):
    """Type definition for the result dictionary."""

    prompt: str
    name: str
    parameters: dict[str, Any]


class LLMResponse:
    def __init__(
        self,
        result: list[ResultDictType],
        functions_definition: list[FunctionDefinition],
    ) -> None:
        self.llm = Small_LLM_Model()  # NOTE: inherent?
        self.result = result
        self.functions_definition = functions_definition

    def create_dict(self, prompt: str, name: str) -> ResultDictType:
        return {"prompt": prompt, "name": name, "parameters": None}

    def functions_constrained_decoding(self, logits: list[float], index: int) -> list[float]:
        names_2d = [self.llm.encode(function.name).tolist()[0] for function in self.functions_definition]
        new_logits = np.full_like(logits, -np.inf)

        i = 0
        while i < len(names_2d):
            new_logits[names_2d[i][index]] = logits[names_2d[i][index]]
            i += 1
        return new_logits

    def create_available_function(self) -> str:
        functions_prompt = ""
        for function_definition in self.functions_definition:
            functions_prompt += "".join(str(function_definition))
        return functions_prompt

    def _finish_tokens(self, text: str) -> bool:
        names = [function.name for function in self.functions_definition]
        return text not in names

    def generate_response(self, prompt: str) -> None:
        function_name = ""
        # text = SYSTEM_PROMPT.format(
        #     PROMPT=prompt, FUNCTIONS=self.create_available_function()
        # )
        text = "XX_addnumber"

        input_ids: list[int] = self.llm.encode(text).tolist()[0]
        index = 0
        while self._finish_tokens(function_name):
            logits = self.llm.get_logits_from_input_ids(input_ids)
            logits = self.functions_constrained_decoding(logits, index)
            next_token = np.argmax(logits)
            function_name += self.llm.decode(next_token)
            print(function_name)
            input_ids.append(next_token)
            index += 1

        self.result.append(self.create_dict(prompt, function_name))
