"""Custom exceptions for the SQL analysis pipeline."""


class PipelineBaseError(Exception):
    """Base class for all pipeline exceptions."""


class ParsingError(PipelineBaseError):
    """Raised when sqlglot cannot parse the input SQL."""


class SerializationError(PipelineBaseError):
    """Raised when AST serialization encounters an unexpected structure."""


class LineageExtractionError(PipelineBaseError):
    """Raised when column lineage cannot be derived."""


class LLMCommunicationError(PipelineBaseError):
    """Raised when the LLM API or chain invocation fails."""


class InvalidResponseError(PipelineBaseError):
    """Raised when the LLM response is malformed or cannot be validated."""
