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


class SessionConflictError(StorageError):
    """The session file changed on disk after it was loaded (e.g. another process saved)."""


class AuditTrailError(StorageError):
    """A save would modify or remove an existing verification or evidence record."""


class InvalidIdentifierError(StorageError):
    """An identifier is malformed (and could be unsafe to use in a file path)."""


# ------------------------------------------------------------- reference collections


class ReferenceCollectionError(HallucinationLabError):
    """Base class for reference collection, ingestion and retrieval problems."""


class DocumentIngestionError(ReferenceCollectionError):
    """A document could not be ingested (unsupported, empty, too large, undecodable)."""


class DuplicateDocumentError(ReferenceCollectionError):
    """The document's content is already registered in the collection."""


class DocumentModifiedError(ReferenceCollectionError):
    """A registered document's bytes no longer match the hash recorded when it was added."""


class CollectionNotFoundError(ReferenceCollectionError):
    """No reference collection exists with the requested name."""


class ReadOnlyCollectionError(ReferenceCollectionError):
    """The packaged demonstration collection cannot be modified through the tool."""


class IndexNotFoundError(ReferenceCollectionError):
    """The collection has not been indexed yet."""


class StaleIndexError(ReferenceCollectionError):
    """The saved index no longer matches the collection's documents or settings."""


class QueryError(ReferenceCollectionError):
    """A search query is empty or contains no searchable terms."""


# -------------------------------------------------------------------- verification


class VerificationError(HallucinationLabError):
    """A verification, draft or revision request is invalid."""
