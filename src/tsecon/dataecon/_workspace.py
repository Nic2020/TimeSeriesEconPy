# SPDX-License-Identifier: MIT
"""Workspace interchange: whole catalog trees to and from the core ``Workspace``.

Julia's ``writedb``/``readdb`` walk a ``Workspace`` recursively, store each
member through the typed writer its type selects (a nested ``Workspace``
becomes a catalog), and on read rebuild a ``Workspace`` from each catalog
through the typed loader its stored class selects. A member that cannot be
written or read is logged and skipped; nothing is rolled back.

The Python form keeps that behavior as the default while making it visible:
every skipped member is a :class:`SkippedMember` in a :class:`WorkspaceReport`
naming its full path and reason, and ``strict=True`` raises instead. The
traversal is iterative (an explicit stack, never Python recursion), so a
catalog tree deeper than the interpreter's recursion limit is fine; a nested
``Workspace`` that is its own ancestor is reported as a cycle before any
catalog is created, while the same object under two names is stored twice.
Dispatch is by Python type only; each member is encoded once by the typed
method that stores it, with that method's own validation and residue rules.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from tsecon.mit import MIT, Duration
from tsecon.mitrange import MITRange
from tsecon.mvtseries import MVTSeries
from tsecon.tseries import TSeries
from tsecon.workspace import Workspace

from ._arrays import StoredArray, StoredText
from ._errors import DataEconError
from ._represented import StoredMVTSeries, StoredSeries
from ._scalars import IntegerComplex, StoredScalar

if TYPE_CHECKING:
    from . import DataEconFile

__all__ = [
    "LoadedWorkspace",
    "SkippedMember",
    "WorkspaceReport",
    "classify",
    "read_tree",
    "reader_for",
    "write_tree",
]


@dataclass(frozen=True, slots=True)
class SkippedMember:
    """One member that a Workspace write or read did not store or load.

    ``path`` is the member's full path in the file. ``category`` is
    ``"unsupported"`` (no dispatch family, a ``TypeError`` from the typed
    method, or a native-only object type), ``"invalid"`` (a ``ValueError``:
    an invalid key, value or payload), ``"exists"`` (a name conflict that was
    not overwritten), ``"native"`` (a ``DataEconError`` from the native store
    or load) or ``"cycle"`` (a nested Workspace that is its own ancestor).
    ``reason`` carries the exception class and message. ``subtree`` is true
    when the entry stands for a whole nested Workspace or catalog whose
    members were never attempted.
    """

    path: str
    category: str
    reason: str
    subtree: bool = False


@dataclass(frozen=True, slots=True)
class WorkspaceReport:
    """What a Workspace write or read did: the catalog, a count and the skips.

    ``count`` is the number of objects stored or loaded, catalogs included;
    ``skipped`` lists the members that were not, in traversal order. ``ok``
    is true when nothing was skipped.
    """

    path: str
    count: int
    skipped: tuple[SkippedMember, ...]

    @property
    def ok(self) -> bool:
        """Whether every member was stored or loaded."""
        return not self.skipped


class LoadedWorkspace(NamedTuple):
    """The result of a Workspace read: the ``Workspace`` and its report."""

    workspace: Workspace
    report: WorkspaceReport


_REPORTED = (TypeError, ValueError, DataEconError)
_SERIES_TYPES = (TSeries, MVTSeries, StoredSeries, StoredMVTSeries)
_ARRAY_TYPES = (np.ndarray, StoredArray, StoredText, list, tuple, range, MITRange)
_SCALAR_TYPES = (
    bool,
    int,
    float,
    complex,
    str,
    MIT,
    Duration,
    np.generic,
    StoredScalar,
    Fraction,
    dt.date,
    IntegerComplex,
)
# Stored (class, type) pairs the typed readers accept; every other pair is a
# native-only encoding that Julia never writes and Python does not load.
_READERS = {
    (1, 1): "read_scalar",
    (1, 2): "read_scalar",
    (1, 3): "read_scalar",
    (1, 4): "read_scalar",
    (1, 5): "read_scalar",
    (1, 6): "read_scalar",
    (2, 12): "read_series",
    (3, 21): "read_series",
    (2, 10): "read_array",
    (2, 11): "read_array",
    (3, 20): "read_array",
    (4, 30): "read_array",
}


def classify(value: object) -> str | None:
    """Name the typed writer for a value (``workspace``/``series``/``array``/``scalar``) or None."""
    if isinstance(value, Workspace):
        return "workspace"
    if isinstance(value, _SERIES_TYPES):
        return "series"
    if isinstance(value, _ARRAY_TYPES):
        return "array"
    if isinstance(value, _SCALAR_TYPES):
        return "scalar"
    return None


def reader_for(object_class: int, object_type: int) -> str | None:
    """Name the typed reader for a stored class/type, or None for a native-only type."""
    return _READERS.get((object_class, object_type))


def _key_problem(name: object) -> str | None:
    if not isinstance(name, str):
        return f"Workspace keys must be strings, got {type(name).__name__}."
    if not name:
        return "Empty member name."
    if "/" in name:
        return "Member names cannot contain '/' (nest Workspaces instead)."
    if "\0" in name:
        return "Member names cannot contain NUL."
    if not name.strip():
        return "Blank member name."
    return None


def _skipped(exc: BaseException, path: str, subtree: bool) -> SkippedMember:
    if isinstance(exc, DataEconError):
        category = "exists" if exc.code == -985 else "native"
    elif isinstance(exc, ValueError):
        category = "invalid"
    else:
        category = "unsupported"
    return SkippedMember(path, category, f"{type(exc).__name__}: {exc}", subtree)


def _raise_for(exc: BaseException, path: str) -> None:
    exc.add_note(f"Workspace member {path!r}")
    raise exc


def _prepare_catalog(db: DataEconFile, path: str, overwrite: bool) -> None:
    # Create the catalog for a nested Workspace member. Without overwrite an
    # existing name of any class is the native DE_EXISTS pre-check (-985, no
    # failed statement). With overwrite an existing catalog or leaf is deleted
    # first (recursively for a catalog: Julia replaces the whole subtree, it
    # does not merge) and the catalog is created new.
    if overwrite:
        existing = db._describe_or_none(path)
        if existing is not None:
            db.delete(path, recursive=existing.kind == "catalog")
    db.new_catalog(path)


def _write_leaf(db: DataEconFile, family: str, path: str, value: Any, overwrite: bool) -> None:
    if overwrite:
        existing = db._describe_or_none(path)
        if existing is not None and existing.kind == "catalog":
            raise DataEconError(
                -985,
                "write_workspace",
                db.path,
                "Object already exists and is a catalog; a value never replaces a catalog "
                "(delete it with recursive=True first).",
                path,
            )
    if family == "series":
        db.write_series(path, value, overwrite=overwrite)
    elif family == "array":
        db.write_array(path, value, overwrite=overwrite)
    else:
        db.write_scalar(path, value, overwrite=overwrite)


def _store(
    db: DataEconFile,
    family: str | None,
    path: str,
    value: Any,
    overwrite: bool,
    report: Callable[[BaseException, str, bool], None],
) -> bool:
    # Store one member through its family's writer; True when it was stored,
    # False when the failure was reported (or raised by ``report`` in strict
    # mode). Only the three documented classes are reported; anything else
    # is a programming error and propagates.
    if family is None:
        report(
            TypeError(f"No DataEcon storage class for a member of type {type(value).__name__}."),
            path,
            False,
        )
        return False
    try:
        if family == "workspace":
            _prepare_catalog(db, path, overwrite)
        else:
            _write_leaf(db, family, path, value, overwrite)
    except _REPORTED as exc:
        report(exc, path, family == "workspace")
        return False
    return True


def write_tree(
    db: DataEconFile,
    workspace: Workspace,
    base: str,
    report_path: str,
    overwrite: bool,
    strict: bool,
) -> WorkspaceReport:
    """Store every member of ``workspace`` below the catalog ``base`` (``""`` for the root).

    The caller holds the file lock and has verified that the owner is open and
    writable and that the destination exists and is a catalog. Members are
    visited depth-first in insertion order (Julia's order, so ids follow it)
    from an explicit stack of iterators; the ids of the Workspaces on the
    current path detect cycles before any catalog is created for them.
    """
    skipped: list[SkippedMember] = []
    count = 0
    on_path = {id(workspace)}
    stack: list[tuple[Iterator[tuple[str, Any]], str, int]] = [
        (iter(workspace.items()), base, id(workspace))
    ]

    def report(exc: BaseException, path: str, subtree: bool) -> None:
        if isinstance(exc, DataEconError) and db.closed:
            # A quarantined owner is never used again; the failure propagates.
            raise exc
        if strict:
            _raise_for(exc, path)
        skipped.append(_skipped(exc, path, subtree))

    while stack:
        items, parent, wid = stack[-1]
        try:
            name, value = next(items)
        except StopIteration:
            stack.pop()
            on_path.discard(wid)
            continue
        path = f"{parent}/{name}"
        problem = _key_problem(name)
        if problem is not None:
            report(ValueError(problem), path, isinstance(value, Workspace))
        elif isinstance(value, Workspace):
            if id(value) in on_path:
                exc = ValueError("This nested Workspace is its own ancestor (a cycle).")
                if strict:
                    _raise_for(exc, path)
                skipped.append(SkippedMember(path, "cycle", str(exc), True))
            elif _store(db, "workspace", path, value, overwrite, report):
                count += 1
                on_path.add(id(value))
                stack.append((iter(value.items()), path, id(value)))
        elif _store(db, classify(value), path, value, overwrite, report):
            count += 1
    return WorkspaceReport(report_path, count, tuple(skipped))


def _listed_path(target: Workspace, parent: str, name: str) -> tuple[str, str | None]:
    # The path a listed row's name reconstructs, and why it cannot be trusted
    # when it cannot: the name comes back as a C string, so a foreign row
    # whose name holds NUL is cut short, and a foreign name may hold "/" or be
    # blank (the native name rule is not enforced on foreign SQLite writers).
    # A cut or slashed name can alias another object's path, and two rows
    # can cut to the same key.
    path = f"{parent}/{name}"
    if not name or "/" in name or not name.strip():
        return (
            path,
            f"Listed name {name!r} is not a valid path component; the row is unreadable by path.",
        )
    if name in target:
        return (
            path,
            f"Listed name {name!r} repeats a member already loaded from this catalog "
            "(an aliased foreign name).",
        )
    return path, None


def _verify_identity(db: DataEconFile, path: str, listed_id: int) -> None:
    # The reconstructed path must resolve to the very row that was listed;
    # otherwise reading it would return another object's bytes under this key.
    found = db._handle.describe_path(path)[0]
    if found != listed_id:
        raise ValueError(
            f"Path {path!r} resolves to object {found}, not the listed object {listed_id}: "
            "the stored name aliases another object (a foreign name with NUL or '/')."
        )


def read_tree(
    db: DataEconFile, start_id: int, base: str, report_path: str, strict: bool
) -> LoadedWorkspace:
    """Load the catalog ``start_id`` (path ``base``, ``""`` for the root) into a ``Workspace``.

    The caller holds the file lock and has verified that the object is a
    catalog. Each catalog is listed once (owned rows in byte order, fully
    consumed and finalized); the traversal is depth-first pre-order from an
    explicit stack of row iterators, so a nested catalog's members are read
    before the catalog's later siblings (Julia's recursion order), and keys
    come back in UTF-8 byte order per level. Before a member is read, its
    reconstructed path must resolve to the listed object id and its name
    must be a valid, not yet used key; a foreign name that is cut at NUL,
    holds ``/`` or repeats a sibling is reported, never read as another
    object. A catalog whose listing fails is reported as a skipped subtree
    and removed from its parent; a member that fails is reported alone.
    """
    root = Workspace()
    skipped: list[SkippedMember] = []
    count = 0

    def report(exc: BaseException, path: str, subtree: bool) -> None:
        if isinstance(exc, DataEconError) and db.closed:
            raise exc
        if strict:
            _raise_for(exc, path)
        skipped.append(_skipped(exc, path, subtree))

    # One frame per catalog being filled: its owned rows, the Workspace that
    # receives them and the catalog's path. The destination's own listing
    # failure is the caller's error, not a skipped member.
    stack: list[tuple[Iterator[tuple[int, int, int, int, str]], Workspace, str]] = [
        (iter(db._children(start_id)), root, base)
    ]
    while stack:
        rows, target, parent = stack[-1]
        try:
            child_id, _pid, obj_class, obj_type, name = next(rows)
        except StopIteration:
            stack.pop()
            continue
        path, problem = _listed_path(target, parent, name)
        if problem is not None:
            report(ValueError(problem), path, obj_class == 0)
            continue
        if obj_class == 0:
            try:
                _verify_identity(db, path, child_id)
                child_rows = db._children(child_id)
            except _REPORTED as exc:
                report(exc, path, True)
                continue
            child = Workspace()
            target[name] = child
            count += 1
            stack.append((iter(child_rows), child, path))
            continue
        reader = reader_for(obj_class, obj_type)
        if reader is None:
            report(
                TypeError(
                    f"Object class {obj_class} type {obj_type} is a native-only encoding "
                    "with no Julia or Python loader."
                ),
                path,
                False,
            )
            continue
        try:
            _verify_identity(db, path, child_id)
            value = getattr(db, reader)(path)
        except _REPORTED as exc:
            report(exc, path, False)
            continue
        target[name] = value
        count += 1
    return LoadedWorkspace(root, WorkspaceReport(report_path, count, tuple(skipped)))
