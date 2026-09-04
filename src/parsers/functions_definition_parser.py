import json
from typing import TypedDict

from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_core import PydanticCustomError


class DictType(TypedDict):
    """Represents a dictionary type with a type field."""

    type: str


class FunctionDefinition(BaseModel):
    """Represents a function definition with its name, description, parameters,
    and return type."""

    name: str
    description: str
    parameters: dict[str, DictType]
    returns: DictType

    def __str__(self) -> str:
        parameters = {name: type_hint["type"] for name, type_hint
                      in self.parameters.items()}
        return f"""
           {self.name}
                Description: {self.description}
                parameters: {parameters}
                return: {self.returns}
        """


def load_function_definitions(file_path: str) -> list[FunctionDefinition]:
    """Loads function definitions from a JSON file."""

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        adapter = TypeAdapter(list[FunctionDefinition])

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
