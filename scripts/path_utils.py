"""
Shared path-validation helpers used across CI scripts.

All functions enforce containment within a trusted root directory to
prevent path-traversal vulnerabilities.  Absolute paths that are
already resolved by the caller are passed through without a root check
(the caller is trusted to supply them securely).
"""

from pathlib import Path
from typing import Optional, FrozenSet


def validate_path(
    path: str,
    root: Path,
    allowed_suffixes: Optional[FrozenSet[str]] = None,
) -> Path:
    """
    Resolve *path* and assert it remains inside *root*.

    - Relative paths are joined under *root* and must not escape it
      (path-traversal guard via ``Path.relative_to``).
    - Absolute paths are resolved as-is (containment check skipped;
      the caller is responsible for supplying a trusted path).

    Args:
        path: Relative or absolute path string.
        root: The trusted root directory for containment checks.
        allowed_suffixes: Optional set of lower-cased file extensions
            (e.g. ``frozenset({".json", ".py"})``) that the resolved
            path must have.  Pass ``None`` to skip the check.

    Returns:
        The resolved ``Path`` object.

    Raises:
        ValueError: If the path escapes *root* (for relative paths) or
            if the suffix is not in *allowed_suffixes*.
    """
    p = Path(path)
    if p.is_absolute():
        candidate = p.resolve()
    else:
        candidate = (root / path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            raise ValueError(f"Path escapes allowed directory: {path}")

    if allowed_suffixes is not None and candidate.suffix.lower() not in allowed_suffixes:
        raise ValueError(
            f"Invalid file type '{candidate.suffix}'. "
            f"Expected one of {sorted(allowed_suffixes)}."
        )

    return candidate


def validate_input_path(
    path: str,
    root: Path,
    allowed_suffixes: Optional[FrozenSet[str]] = None,
) -> Path:
    """
    Validate a read-only input path and assert it exists on disk.

    Delegates containment and suffix checks to :func:`validate_path`,
    then additionally verifies the file is present.

    Args:
        path: Relative or absolute path string.
        root: The trusted root directory for containment checks.
        allowed_suffixes: Optional lower-cased extension whitelist.

    Returns:
        The resolved, existing ``Path`` object.

    Raises:
        ValueError: If the path escapes *root*, has a disallowed suffix,
            or does not exist.
    """
    candidate = validate_path(path, root, allowed_suffixes)
    if not candidate.exists():
        raise ValueError(f"Input path does not exist: {path}")
    return candidate


def validate_output_path(
    path: str,
    root: Path,
    allowed_suffixes: Optional[FrozenSet[str]] = None,
) -> Path:
    """
    Validate a writable output path (the file need not exist yet).

    Delegates containment and suffix checks to :func:`validate_path`.
    The parent directory is *not* created here; callers are responsible
    for calling ``parent.mkdir(parents=True, exist_ok=True)`` before
    writing.

    Args:
        path: Relative or absolute path string.
        root: The trusted root directory for containment checks.
        allowed_suffixes: Optional lower-cased extension whitelist.

    Returns:
        The resolved ``Path`` object.

    Raises:
        ValueError: If the path escapes *root* or has a disallowed suffix.
    """
    return validate_path(path, root, allowed_suffixes)
