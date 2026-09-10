from typing import Any, TypedDict

import numpy as np

from llm_sdk import Small_LLM_Model  # type: ignore

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


def create_dict(
    prompt: str, name: str, parameters: dict[str, Any],
) -> ResultDictType:
    """Create a result dictionary with prompt, function name, and parameters.

    Args:
        prompt: The user prompt.
        name: The function name.
        parameters: The function parameters.

    Returns:
        ResultDictType containing the prompt, name, and parameters.
    """
    return {
        "prompt": prompt,
        "name": name,
        "parameters": parameters
    }


class LLMResponse:
    """
        Handles LLM response generation with constrained
        decoding for function calling.
    """

    def __init__(
        self,
        result: list[ResultDictType],
        functions_definition: list[FunctionDefinition],
    ) -> None:
        """Initialize with result list and function definitions."""
        self.llm = Small_LLM_Model()
        self.result = result
        self.functions_definition = functions_definition

    def functions_constrained_decoding(
        self, logits: list[float], index: int,
    ) -> list[float]:
        """
            Apply constrained decoding to restrict logits to
            valid function name tokens.
        """
        new_logits = np.full_like(logits, -np.inf)

        i = 0
        while i < len(self.names_2d):
            token_id = self.names_2d[i][index]
            new_logits[token_id] = logits[token_id]
            i += 1

        i = 0
        while i < len(self.names_2d):
            if self.names_2d[i][index] != np.argmax(new_logits):
                del self.names_2d[i]
                continue
            i += 1
        return list(new_logits)

    def parameter_constrained_decoding(
            self, logits: list[float], parameter_type: str
    ) -> list[float]:
        """
            Apply constrained decoding based on parameter type
            (number, integer, boolean, string).
        """
        new_logits = np.full_like(logits, -np.inf)

        if parameter_type == "number":
            allowed = self.llm.encode(".0123456789").tolist()[0]
            allowed.append(1)
        elif parameter_type == "integer":
            allowed = self.llm.encode("0123456789").tolist()[0]
            allowed.append(1)
        elif parameter_type == "boolean":
            allowed = [1866, 3849]
        else:
            return logits

        for i in range(len(allowed)):
            new_logits[allowed[i]] = logits[allowed[i]]

        return list(new_logits)

    def create_available_function(self) -> str:
        """Create a formatted string of all available function definitions.

        Returns:
            Formatted string of function definitions.
        """
        functions_prompt = ""
        for function_definition in self.functions_definition:
            functions_prompt += "".join(str(function_definition))
        return functions_prompt

    def _finish_tokens(self, text: str) -> bool:
        """Check if text is a complete function name from available functions.

        Args:
            text: The text to check.

        Returns:
            True if text is not a complete function name, False otherwise.
        """
        names = [function.name for function in self.functions_definition]
        return text not in names

    def generate_response(self, prompt: str) -> None:
        """Generate LLM response with function name and parameters.

        Args:
            prompt: The user prompt to generate response for.
        """
        function_name = ""
        self.names_2d = [
            self.llm.encode(function.name).tolist()[0]
            for function in self.functions_definition
        ]
        text = SYSTEM_PROMPT.format(
            PROMPT=prompt, FUNCTIONS=self.create_available_function()
        )

        input_ids: list[int] = self.llm.encode(text).tolist()[0]
        index = 0
        while self._finish_tokens(function_name):
            logits = self.llm.get_logits_from_input_ids(input_ids)
            logits = self.functions_constrained_decoding(logits, index)
            next_token = int(np.argmax(logits))
            function_name += self.llm.decode(next_token)
            input_ids.append(next_token)
            index += 1

        parameters: dict[str, Any] = {}
        match_function = next(
            function for function in self.functions_definition
            if function.name == function_name
        )

        text = f"""
        <think>

        </think>
        Extract the parameters required to call the following\
        function from the user request.

        Function:
        {match_function!s}

        User prompt:
        {prompt}

        Answer is:"""
        for k, v in match_function.parameters.items():
            text += f" \"{k}\": \""
            input_ids = self.llm.encode(text).tolist()[0]
            value = ""
            max_token = 0
            while max_token < len(prompt):
                logits = self.llm.get_logits_from_input_ids(input_ids)
                logits = self.parameter_constrained_decoding(logits, v["type"])
                next_token = int(np.argmax(logits))
                if "\"" in self.llm.decode(next_token):
                    break
                value += self.llm.decode(next_token)
                input_ids.append(next_token)
                max_token += 1

            text += value

            if v["type"] == "number":
                parameters[k] = float(value)
            elif v["type"] == "integer":
                parameters[k] = int(value)
            elif v["type"] == "boolean":
                parameters[k] = bool(value)
            else:
                parameters[k] = str(value)

        self.result.append(create_dict(prompt, function_name, parameters))
