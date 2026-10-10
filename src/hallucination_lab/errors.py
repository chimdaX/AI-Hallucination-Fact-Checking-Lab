"""Exception hierarchy.

Callers (CLI, Streamlit UI) catch ``HallucinationLabError`` to show a clean
message instead of a traceback, while unexpected bugs still surface.
"""


class HallucinationLabError(Exception):
    """Base class for all expected, user-reportable errors."""


class ExtractionError(HallucinationLabError):
    """Claim extraction could not run on the given input."""


class ClaimEditError(HallucinationLabError):
    """A requested manual claim edit (split, merge, remove...) is invalid."""


class StorageError(HallucinationLabError):
    """Base class for persistence problems."""


class SessionNotFoundError(StorageError):
    """No stored session exists for the requested identifier."""


class SessionLoadError(StorageError):
    """A stored session exists but is unreadable or fails schema validation."""


class InvalidIdentifierError(StorageError):
    """An identifier is malformed (and could be unsafe to use in a file path)."""
