import json

from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_core import PydanticCustomError


class PromptItem(BaseModel):
    """Represents one function-calling prompt."""

    prompt: str


def load_prompts(file_path: str) -> list[PromptItem]:
    """Load and validate prompts from a JSON file."""

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        adapter = TypeAdapter(list[PromptItem])

        return adapter.validate_python(data)

    except OSError as e:
        raise OSError(f"Error loading file {file_path}:\n\t{e.strerror}")

    except json.JSONDecodeError as error:
        raise json.JSONDecodeError(
            msg=f"Error Invalid JSON: {error}", doc=error.doc, pos=error.pos
        )

    except ValidationError:
        raise PydanticCustomError(
            "validation_error",
            "Invalid data: The data does not follow the expected format."
        )
