import json
from typing import Literal, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    ValidationError,
    field_validator,
)
from pydantic_core import PydanticCustomError


class DictType(TypedDict):
    """Represents a dictionary type with a type field."""

    type: Literal["number", "string", "integer", "boolean"]


class FunctionDefinition(BaseModel):
    """Represents a function definition with its name, description, parameters,
    and return type."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, DictType]
    returns: DictType

    @field_validator("name")
    def validate_function_name(cls, name: str) -> str:
        if not name.isidentifier():
            raise PydanticCustomError(
                "invalid_function_name",
                "Function name must be a valid identifier",
            )
        return name

    def __str__(self) -> str:
        parameters = {
            name: type_hint["type"] for name, type_hint
            in self.parameters.items()
        }
        return f"""
            {self.name}
                Description: {self.description}
                parameters: {parameters}
                return: {self.returns}
        """


def parse_function_definition(
    data: list[tuple[str, None | str]],
) -> dict[str, None | str]:
    keys = [key for key, _ in data]
    duplicates = {key for key in keys if keys.count(key) > 1}
    if duplicates:
        raise ValueError(f"Duplicate keys found: {duplicates}")

    return dict(data)


def load_function_definitions(file_path: str) -> list[FunctionDefinition]:
    """Loads function definitions from a JSON file."""

    try:
        with open(file_path, "r") as f:
            data = json.load(f, object_pairs_hook=parse_function_definition)

        adapter = TypeAdapter(list[FunctionDefinition])

        definitions = adapter.validate_python(data)

        names = [definition.name for definition in definitions]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise PydanticCustomError(
                "duplicate_function_name_error",
                "Duplicate function names found: {duplicates}",
                {"duplicates": list(duplicates)},
            )

        return definitions

    except OSError as e:
        raise OSError(f"Error loading file {file_path}:\n\t{e.strerror}")

    except json.JSONDecodeError as error:
        raise json.JSONDecodeError(
            msg=f"Error Invalid JSON: {error}", doc=error.doc, pos=error.pos
        )

    except ValidationError as e:
        first_error_msg = e.errors()[0]["msg"]
        raise PydanticCustomError(
            "validation_error", f"Invalid data: {first_error_msg}"
        )
