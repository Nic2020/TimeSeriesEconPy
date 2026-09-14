from ._codec import MatrixMetadata, Metadata, ScalarMetadata, TensorMetadata

def version() -> tuple[str, str]: ...
def abi_layout() -> dict[str, tuple[int, ...] | tuple[int, tuple[int, ...]]]: ...

class FileHandle:
    def __init__(self, path: str, readonly: bool, memory: bool = False) -> None: ...
    def close(self) -> None: ...
    def read(
        self, name: str
    ) -> tuple[bytes, Metadata | MatrixMetadata, str, str | None, str | None, str | None]: ...
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
        parent: str = "/",
    ) -> None: ...
    def read_array(
        self, name: str
    ) -> tuple[
        bytes, Metadata | MatrixMetadata | TensorMetadata, str, str | None, str | None, str | None
    ]: ...
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
        parent: str = "/",
    ) -> None: ...
    def write_matrix(
        self,
        name: str,
        object_type: int,
        element: int,
        element_frequency: int,
        axis1_type: int,
        rows: int,
        frequency: int,
        first: int,
        columns: int,
        names: str | None,
        payload: bytes,
        overwrite: bool,
        marker: str | None,
        object_marker: str | None = None,
        parent: str = "/",
    ) -> None: ...
    def write_tensor(
        self,
        name: str,
        object_type: int,
        element: int,
        element_frequency: int,
        shape: tuple[int, ...],
        payload: bytes,
        overwrite: bool,
        marker: str | None,
        object_marker: str | None = None,
        parent: str = "/",
    ) -> None: ...
    def read_scalar(self, name: str) -> tuple[bytes, ScalarMetadata, str]: ...
    def write_scalar(
        self,
        name: str,
        kind: int,
        frequency: int,
        payload: bytes,
        overwrite: bool = False,
        parent: str = "/",
    ) -> None: ...
    def delete(self, name: str, recursive: bool = False) -> None: ...
    def truncate(self) -> None: ...
    def catalog_size(self, path: str = "/") -> int: ...
    def describe_path(self, path: str) -> tuple[int, int, int, int, str, str, int, int]: ...
    def describe_id(self, object_id: int) -> tuple[int, int, int, int, str, str, int, int]: ...
    def exists(self, path: str) -> bool: ...
    def new_catalog(self, name: str, parent: str = "/") -> int: ...
    def list_children(self, object_id: int) -> list[tuple[int, int, int, int, str]]: ...
    def get_attribute(self, object_id: int, name: str) -> str | None: ...
    def attribute_names(self, object_id: int) -> list[str]: ...
    def set_attribute(self, object_id: int, name: str, value: str) -> None: ...
