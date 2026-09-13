from ._codec import Metadata, ScalarMetadata

def version() -> tuple[str, str]: ...
def abi_layout() -> dict[str, tuple[int, ...] | tuple[int, tuple[int, ...]]]: ...

class FileHandle:
    def __init__(self, path: str, readonly: bool, memory: bool = False) -> None: ...
    def close(self) -> None: ...
    def read(self, name: str) -> tuple[bytes, Metadata, str, str | None, str | None]: ...
    def write(
        self,
        name: str,
        frequency: int,
        first: int,
        payload: bytes,
        overwrite: bool,
        element: int,
        element_frequency: int,
        length: int,
        marker: str | None,
        object_marker: str | None = None,
    ) -> None: ...
    def read_array(self, name: str) -> tuple[bytes, Metadata, str, str | None, str | None]: ...
    def write_array(
        self,
        name: str,
        object_type: int,
        axis_type: int,
        frequency: int,
        first: int,
        payload: bytes,
        overwrite: bool,
        element: int,
        element_frequency: int,
        length: int,
        marker: str | None,
        object_marker: str | None = None,
    ) -> None: ...
    def read_scalar(self, name: str) -> tuple[bytes, ScalarMetadata, str]: ...
    def write_scalar(
        self, name: str, kind: int, frequency: int, payload: bytes, overwrite: bool = False
    ) -> None: ...
    def delete(self, name: str, recursive: bool = False) -> None: ...
    def truncate(self) -> None: ...
    def catalog_size(self) -> int: ...
