import json
import sys
from types import TracebackType
from typing import Self

from pydantic_core import PydanticCustomError


class FileParsingErrorHandler:
    """Context manager to catch OSError, JSONDecodeError,
    or ValidationError."""

    def __init__(self) -> None:
        self.exception = (
            OSError,
            json.JSONDecodeError,
            PydanticCustomError,
            ValueError,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        if exc_type is None:
            return True

        if issubclass(exc_type, self.exception):
            print(exc_val)
            sys.exit(1)

        return False
