_This project has been created as part of the 42 curriculum by mkhoubaz_

# Call-Me-Maybe

## Description

This project is simple introduction to function calling in LLMs and constrained decoding. The goal is to create a json schema that have function name with its parameters from the user prompt.

## Instructions

To install the project dependencies, run the following command:

```bash
    make install
```

Run the project using the following command:

```bash
    make run
```

## Algorithm explanation

Constrained decoding use in two parts, first to generate the functions names by change logits that don't match the names from json file to assure the name is full correct i use prefix tree to remove any token that don't match the last token generated, then to generate function parameters i check type of the parameter thereafter depending on the type i generate the correct token for example if type is number i let the model generate a number token, if type is string i let the model generate a string token, and so on.

## Design decisions

For this project, it's was both OOP and functional because i was need Pydantic to validate the json schema and generate llm result without load the model every time, and i was need functional programming to parse command line arguments, and raise error for invalid json schema with Pydantic, I was created custom context manager to error handling to make code more clean.

## Performance analysis

I generated output for every prompt one at a time so my code depended on number of prompts to generate output, but the best performance was in constrained decoding where i don't loop over every logits but only what it's allowed because i ready new theme from index, and i don't generate prompt key in json output just get it from input, and use prefix tree for functions names make it more efficient because it's remove the token id will not match next token.

## Challenges faced

The hard part was the generate correct answer from small model like qwen3-0.6b, because it's not good with regex and programming. to solve this problem i was `chat template` to stop the model from thinking by set `<think>\n</think>` tags in the prompt, so the model will not generate complex output.

## Testing strategy

I used Pytest to test the and use 42 moulinette to verify the model json output if match the expected output.

## Example usage

```bash
    uv run python -m src --input <input_prompts_json> --functions_definition <functions_definition_json> --output <output_json>
```

Example input:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?"
  },
  {
    "prompt": "What is the sum of 265 and 345?"
  }
]
```

Example functions definition:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {
        "type": "number"
      },
      "b": {
        "type": "number"
      }
    },
    "returns": {
      "type": "number"
    }
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {
      "name": {
        "type": "string"
      }
    },
    "returns": {
      "type": "string"
    }
  }
]
```

Example output:

```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2.0,
            "b": 3.0
        }
    },
    {
        "prompt": "What is the sum of 265 and 345?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 265.0,
            "b": 345.0
        }
    },
]
```
