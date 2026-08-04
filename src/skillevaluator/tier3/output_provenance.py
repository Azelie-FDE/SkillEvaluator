# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Path-bound provenance for evaluator-owned generated output trees."""

from __future__ import annotations

import base64
import contextlib
import hmac
import os
import secrets
import stat
import sys
import time
from pathlib import Path

from skillevaluator.tier3.case_ids import validate_output_directory_path

GENERATED_OUTPUT_MARKER = ".skillevaluator-generated-output"
OUTPUT_PROVENANCE_KEY_ENV = "SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE"

_KEY_BYTES = 32
_MAX_STORED_KEY_BYTES = 4096
_KEY_TEMP_PREFIX = ".output-provenance.key.tmp-"
_MARKER_PREFIX = b"SkillEvaluator generated output v2\n"
_MARKER_CONTEXT = b"skillevaluator.generated-output.v2\0"
_MARKER_SIZE = len(_MARKER_PREFIX) + len(base64.urlsafe_b64encode(bytes(_KEY_BYTES)).rstrip(b"=")) + 1


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)


def _node_fingerprint(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_nlink, metadata.st_size


def _protect_key_for_storage(key: bytes) -> bytes:
    """Bind key bytes to the current Windows user with DPAPI."""
    if os.name != "nt":
        return key
    import ctypes
    from ctypes import wintypes

    class _DataBlob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

    source = ctypes.create_string_buffer(key)
    source_blob = _DataBlob(len(key), ctypes.cast(source, ctypes.POINTER(ctypes.c_ubyte)))
    protected_blob = _DataBlob()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    crypt32.CryptProtectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL
    if not crypt32.CryptProtectData(
        ctypes.byref(source_blob),
        None,
        None,
        None,
        None,
        0x1,
        ctypes.byref(protected_blob),
    ):
        raise ValueError("Cannot protect output provenance key for the current Windows user") from ctypes.WinError(
            ctypes.get_last_error()
        )
    try:
        return ctypes.string_at(protected_blob.pbData, protected_blob.cbData)
    finally:
        kernel32.LocalFree(ctypes.cast(protected_blob.pbData, wintypes.HLOCAL))


def _unprotect_stored_key(payload: bytes) -> bytes:
    """Decode DPAPI-protected Windows bytes or return POSIX bytes unchanged."""
    if os.name != "nt":
        return payload
    import ctypes
    from ctypes import wintypes

    class _DataBlob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

    source = ctypes.create_string_buffer(payload)
    source_blob = _DataBlob(len(payload), ctypes.cast(source, ctypes.POINTER(ctypes.c_ubyte)))
    key_blob = _DataBlob()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL
    if not crypt32.CryptUnprotectData(
        ctypes.byref(source_blob),
        None,
        None,
        None,
        None,
        0x1,
        ctypes.byref(key_blob),
    ):
        raise ValueError("Output provenance key is not protected for the current Windows user") from ctypes.WinError(
            ctypes.get_last_error()
        )
    try:
        key = ctypes.string_at(key_blob.pbData, key_blob.cbData)
    finally:
        kernel32.LocalFree(ctypes.cast(key_blob.pbData, wintypes.HLOCAL))
    if len(key) != _KEY_BYTES:
        raise ValueError("Output provenance key has an invalid decrypted size")
    return key


def _stored_key_size_is_valid(size: int) -> bool:
    return 0 < size <= _MAX_STORED_KEY_BYTES if os.name == "nt" else size == _KEY_BYTES


def _default_key_path() -> Path:
    override = os.environ.get(OUTPUT_PROVENANCE_KEY_ENV)
    if override:
        return Path(override).expanduser().absolute()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "skillevaluator" / "output-provenance.key"


def output_provenance_key_path() -> Path:
    """Return the configured private-key path without creating the key."""
    return _default_key_path()


def _validate_key_metadata(path: Path, metadata: os.stat_result) -> None:
    if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"Output provenance key must be a single-link regular file: {path}")
    if not _stored_key_size_is_valid(metadata.st_size):
        raise ValueError(f"Output provenance key has an invalid size: {path}")
    if os.name == "posix":
        if metadata.st_uid != os.getuid():
            raise ValueError(f"Output provenance key is not owned by the current user: {path}")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ValueError(f"Output provenance key must not be accessible by group or other users: {path}")


def _read_key(path: Path) -> bytes:
    before = path.lstat()
    _validate_key_metadata(path, before)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        _validate_key_metadata(path, opened)
        if _node_fingerprint(opened) != _node_fingerprint(before):
            raise ValueError(f"Output provenance key changed while it was opened: {path}")
        payload = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
        _validate_key_metadata(path, after)
        if _node_fingerprint(after) != _node_fingerprint(before):
            raise ValueError(f"Output provenance key changed while it was read: {path}")
    finally:
        os.close(descriptor)
    if len(payload) != before.st_size:
        raise ValueError(f"Output provenance key has an invalid size: {path}")
    return _unprotect_stored_key(payload)


def _validate_key_directory(path: Path, metadata: os.stat_result) -> None:
    if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"Output provenance key directory must be a real directory: {path}")
    if os.name == "posix":
        if metadata.st_uid != os.getuid():
            raise ValueError(f"Output provenance key directory is not owned by the current user: {path}")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ValueError(f"Output provenance key directory must not be accessible by group or other users: {path}")


def _load_existing_key() -> bytes | None:
    path = _default_key_path()
    try:
        parent_metadata = path.parent.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"Cannot inspect output provenance key directory: {path.parent}") from exc
    _validate_key_directory(path.parent, parent_metadata)
    try:
        return _read_key(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"Cannot safely read output provenance key: {path}") from exc


def _recover_interrupted_key_publish(path: Path) -> bool:
    """Remove only same-inode private temp links left by an interrupted publish."""
    try:
        target = path.lstat()
    except OSError:
        return False
    if _is_link_or_reparse(target) or not stat.S_ISREG(target.st_mode) or not _stored_key_size_is_valid(target.st_size):
        return False
    if os.name == "posix" and target.st_uid != os.getuid():
        return False
    recovered = False
    for candidate in path.parent.iterdir():
        if not candidate.name.startswith(_KEY_TEMP_PREFIX):
            continue
        try:
            metadata = candidate.lstat()
            if (
                not _is_link_or_reparse(metadata)
                and stat.S_ISREG(metadata.st_mode)
                and metadata.st_dev == target.st_dev
                and metadata.st_ino == target.st_ino
            ):
                candidate.unlink()
                recovered = True
        except OSError:
            continue
    return recovered


def _read_key_after_concurrent_publish(path: Path) -> bytes:
    last_error: OSError | ValueError | None = None
    for _ in range(50):
        try:
            return _read_key(path)
        except (OSError, ValueError) as exc:
            last_error = exc
            _recover_interrupted_key_publish(path)
            time.sleep(0.002)
    if last_error is not None:
        raise last_error
    raise ValueError(f"Cannot safely read concurrently published output provenance key: {path}")


def _load_or_create_key() -> bytes:
    path = _default_key_path()
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    try:
        parent_metadata = path.parent.lstat()
    except OSError as exc:
        raise ValueError(f"Cannot inspect output provenance key directory: {path.parent}") from exc
    _validate_key_directory(path.parent, parent_metadata)
    try:
        return _read_key(path)
    except FileNotFoundError:
        pass
    except ValueError:
        return _read_key_after_concurrent_publish(path)
    except OSError as exc:
        raise ValueError(f"Cannot safely read output provenance key: {path}") from exc

    key = secrets.token_bytes(_KEY_BYTES)
    stored_key = _protect_key_for_storage(key)
    temporary = path.parent / f"{_KEY_TEMP_PREFIX}{os.getpid()}-{secrets.token_hex(8)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(stored_key)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return _read_key_after_concurrent_publish(path)
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    return _read_key(path)


def _canonical_output_binding(path: Path) -> bytes:
    try:
        resolved = path.expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Cannot resolve generated output path for provenance: {path}") from exc
    normalized = os.path.normcase(os.path.normpath(os.fspath(resolved)))
    return os.fsencode(normalized)


def _path_is_contained(path: Path, root: Path) -> bool:
    for candidate, candidate_root in (
        (path.expanduser().absolute(), root.expanduser().absolute()),
        (path.expanduser().resolve(strict=False), root.expanduser().resolve(strict=False)),
    ):
        normalized = Path(os.path.normcase(os.path.normpath(os.fspath(candidate))))
        normalized_root = Path(os.path.normcase(os.path.normpath(os.fspath(candidate_root))))
        try:
            normalized.relative_to(normalized_root)
        except ValueError:
            continue
        return True
    return False


def validate_provenance_key_outside(workspace_root: Path) -> None:
    """Reject a private-key location that could enter an evaluated or generated tree."""
    try:
        inside_workspace = _path_is_contained(_default_key_path(), workspace_root)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Cannot validate output provenance key location against: {workspace_root}") from exc
    if inside_workspace:
        raise ValueError(f"Output provenance key must be outside evaluated and generated trees: {workspace_root}")


def _marker_payload(destination: Path, key: bytes) -> bytes:
    digest = hmac.digest(key, _MARKER_CONTEXT + _canonical_output_binding(destination), "sha256")
    signature = base64.urlsafe_b64encode(digest).rstrip(b"=")
    return _MARKER_PREFIX + signature + b"\n"


def generated_output_marker_payload(destination: Path) -> bytes:
    """Return marker bytes signed for one canonical public destination."""
    validate_provenance_key_outside(destination)
    return _marker_payload(destination, _load_or_create_key())


def _read_marker(path: Path, expected_size: int) -> bytes | None:
    try:
        before = path.lstat()
    except OSError:
        return None
    if (
        _is_link_or_reparse(before)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size != expected_size
    ):
        return None
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        opened = os.fstat(descriptor)
        if (
            _is_link_or_reparse(opened)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != expected_size
            or _node_fingerprint(opened) != _node_fingerprint(before)
        ):
            return None
        payload = os.read(descriptor, expected_size + 1)
        after = os.fstat(descriptor)
        if _node_fingerprint(after) != _node_fingerprint(before):
            return None
        return payload
    except OSError:
        return None
    finally:
        os.close(descriptor)


def is_generated_output_root(path: Path) -> bool:
    """Return whether *path* has a valid marker bound to its canonical location."""
    observed = _read_marker(path / GENERATED_OUTPUT_MARKER, _MARKER_SIZE)
    if observed is None or not observed.startswith(_MARKER_PREFIX):
        return False
    validate_provenance_key_outside(path)
    key = _load_existing_key()
    if key is None:
        return False
    return hmac.compare_digest(observed, _marker_payload(path, key))


def validate_generated_output_replacement(path: Path) -> None:
    """Allow replacement only for absent, empty, or authentically owned roots."""
    validate_output_directory_path(path)
    if not os.path.lexists(path):
        return
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"Cannot inspect generated output root before replacement: {path}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"Generated output root must be a real directory: {path}")
    try:
        has_contents = next(path.iterdir(), None) is not None
    except OSError as exc:
        raise ValueError(f"Cannot inspect generated output root before replacement: {path}") from exc
    if has_contents and not is_generated_output_root(path):
        raise ValueError(f"Generated output marker is missing, invalid, or bound to another path: {path}")


def write_generated_output_marker(root: Path, *, destination: Path | None = None) -> None:
    """Write provenance into a caller-owned tree, optionally for a later public path."""
    validate_provenance_key_outside(root)
    validate_output_directory_path(root)
    root.mkdir(parents=True, exist_ok=True)
    validate_output_directory_path(root)
    expected = generated_output_marker_payload(destination or root)
    marker = root / GENERATED_OUTPUT_MARKER
    try:
        descriptor = os.open(
            marker,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError:
        observed = _read_marker(marker, len(expected))
        if observed is None or not hmac.compare_digest(observed, expected):
            raise ValueError(f"Generated output marker is invalid or unsafe: {marker}") from None
        return
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(expected)
        handle.flush()
        os.fsync(handle.fileno())


def mark_generated_output_root(path: Path) -> None:
    """Claim or validate an output root before any generated child is replaced."""
    validate_generated_output_replacement(path)
    path.mkdir(parents=True, exist_ok=True)
    validate_output_directory_path(path)
    write_generated_output_marker(path)
